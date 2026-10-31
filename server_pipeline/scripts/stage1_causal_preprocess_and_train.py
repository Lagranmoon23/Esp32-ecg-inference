"""Rebuild the dataset with a deployment-realistic causal filter chain and retrain Stage 1.

The training pipeline used sosfiltfilt (forward + backward), which is impossible
on a live device. The measured cost of swapping in a causal filter while keeping
the old weights is 0.05-0.09 macro-F1 (see stage1_causal_filter_study.py).

The fix is not a cleverer filter, it is consistency: preprocess with exactly the
chain the firmware will run, then retrain. This script does that.

Deployment chain (implemented as second-order sections on the device):
    bandpass  Butterworth order 4, 0.5-40 Hz, forward only
    notch     50 Hz, Q=30, forward only
    normalize median / (1.4826 * MAD)

RR features are derived from annotation peaks and do not depend on the filter, so
they are reused from the zero-phase split file after a shape check.

Run with mcu_env.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import wfdb
from scipy import signal
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras import layers


DATA = ("/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/"
        "MIT-BIH/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0")
SRC_SPLITS = "patient_independent_hierarchical_splits.npz"
OUT_SPLITS = "patient_independent_hierarchical_causal_splits.npz"
OUT_WEIGHTS = "tiny_transformer_stage1_n_s_vf_causal.weights.h5"

FS = 360
HALF = 99
NUM_HEADS, KEY_DIM, EMBEDDING, MLP_DIM, KERNEL = 8, 2, 16, 128, 3
RR_FEATURE_COUNT = 6
CLASS_TYPES = ["N", "S", "V", "F"]

DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
       '122', '124', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
       '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
VALIDATION_RECORDS = ['118', '205', '207', '223']
TRAIN_RECORDS = [r for r in DS1 if r not in VALIDATION_RECORDS]
TEST_RECORDS = DS2

AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", ".": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V", "F": "F",
        "/": "Q", "f": "Q", "Q": "Q", "?": "Q"}


# --------------------------------------------------------------------------- #
# causal preprocessing: this is the chain the firmware must implement
# --------------------------------------------------------------------------- #
def build_sos():
    bp = signal.butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")
    b, a = signal.iirnotch(50.0, 30.0, fs=FS)
    notch = signal.tf2sos(b, a)
    return bp, notch


def causal_preprocess(x: np.ndarray, sos_bp, sos_notch) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    y, _ = signal.sosfilt(sos_bp, x, zi=signal.sosfilt_zi(sos_bp) * x[0])
    y, _ = signal.sosfilt(sos_notch, y, zi=signal.sosfilt_zi(sos_notch) * y[0])
    center = np.median(y)
    scale = 1.4826 * np.median(np.abs(y - center))
    if scale < 1e-6:
        scale = np.std(y) + 1e-8
    return ((y - center) / scale).astype(np.float32)


def pick_lead(record) -> int:
    names = [str(n).upper().replace(" ", "") for n in record.sig_name]
    for preferred in ("MLII", "II"):
        if preferred in names:
            return names.index(preferred)
    return 0


def extract_beats(record_id: str, sos_bp, sos_notch):
    path = str(Path(DATA) / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    y = causal_preprocess(rec.p_signal[:, pick_lead(rec)], sos_bp, sos_notch)

    mapped = [(s, AAMI[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
    r_peaks = np.asarray([m[0] for m in mapped], dtype=np.int64)
    symbols = [m[1] for m in mapped]
    rr_seconds = np.diff(r_peaks) / float(FS)

    beats, labels = [], []
    for i in range(1, len(r_peaks) - 1):
        if symbols[i] not in CLASS_TYPES:
            continue
        start, end = r_peaks[i] - HALF, r_peaks[i] + HALF
        if start < 0 or end >= len(y):
            continue
        beats.append(y[start:end])
        labels.append(symbols[i])
    return beats, labels


def build_split(records, sos_bp, sos_notch, name):
    beats, labels = [], []
    for rid in records:
        b, l = extract_beats(rid, sos_bp, sos_notch)
        if not b:
            raise RuntimeError(f"no beats from {rid}")
        beats.extend(b)
        labels.extend(l)
    label_map = {c: i for i, c in enumerate(CLASS_TYPES)}
    y = np.asarray([label_map[s] for s in labels], dtype=np.int64)
    print(f"  {name}: {len(beats)} beats {Counter(labels)}", flush=True)
    return np.asarray(beats, dtype=np.float32), y


# --------------------------------------------------------------------------- #
# model + training, identical to the notebook recipe
# --------------------------------------------------------------------------- #
@tf.keras.utils.register_keras_serializable(package="ecg")
class AddPositionEmbs(layers.Layer):
    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = posemb_init

    def build(self, inputs_shape):
        init = self.posemb_init
        if init is None or isinstance(init, dict):
            init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.pos_embedding = self.add_weight(
            name="pos_embedding", shape=(1, inputs_shape[1], inputs_shape[2]),
            initializer=init, trainable=True,
        )

    def call(self, inputs, inputs_positions=None):
        return inputs + tf.cast(self.pos_embedding, inputs.dtype)


def build_model() -> tf.keras.Model:
    x1 = layers.Input(shape=(198, 1, 1), name="x1")
    x2 = layers.Input(shape=(RR_FEATURE_COUNT,), name="x2")
    x = layers.Conv2D(EMBEDDING, (KERNEL, 1), strides=(KERNEL, 1), padding="valid")(x1)
    x = layers.Reshape((66, EMBEDDING))(x)
    x = AddPositionEmbs(name="posembed_input")(x)
    x = layers.Dropout(0.2)(x)
    n = layers.LayerNormalization(dtype=x.dtype)(x)
    a = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM, dropout=0.1)(n, n)
    x = layers.Add()([a, x])
    n = layers.LayerNormalization(dtype=x.dtype)(x)
    y = layers.Dense(MLP_DIM, activation=tf.nn.gelu)(n)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(EMBEDDING, activation=tf.nn.gelu)(y)
    y = layers.Dropout(0.1)(y)
    x = layers.Add()([y, x])
    enc = layers.LayerNormalization(name="encoder_norm")(x)
    morph = layers.Concatenate()([layers.GlobalAveragePooling1D()(enc),
                                  layers.GlobalMaxPooling1D()(enc)])
    rr = layers.LayerNormalization(name="rr_norm")(x2)
    rr = layers.Dense(16, activation=tf.nn.gelu, name="rr_embedding")(rr)
    rr = layers.Dropout(0.15)(rr)
    f = layers.Concatenate()([morph, rr])
    f = layers.Dense(32, activation=tf.nn.gelu, name="fusion")(f)
    f = layers.Dropout(0.25)(f)
    logits = layers.Dense(3, name="stage1_n_s_vf_head",
                          kernel_initializer=tf.keras.initializers.Zeros())(f)
    return tf.keras.Model(inputs=[x1, x2], outputs=logits, name="stage1_n_s_vf")


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=(0.20, 0.70, 0.45), gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = gamma

    def call(self, y_true, logits):
        probs = tf.nn.softmax(logits, axis=-1)
        p_t = tf.reduce_sum(y_true * probs, axis=-1)
        a_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
        return a_t * tf.pow(1.0 - p_t, self.gamma) * ce


def augment(x, y):
    x1 = x["x1"]
    batch = tf.shape(x1)[0]
    shape = tf.stack([batch, 1, 1, 1])
    amp = tf.random.uniform(shape, 0.90, 1.10)
    noise = tf.random.normal(tf.shape(x1), stddev=0.015, dtype=x1.dtype)
    phase = tf.random.uniform(shape, 0.0, 2.0 * np.pi, dtype=x1.dtype)
    t = tf.reshape(tf.linspace(0.0, 2.0 * np.pi, 198), [1, 198, 1, 1])
    return {"x1": x1 * amp + noise + 0.02 * tf.sin(t + phase), "x2": x["x2"]}, y


def make_ds(X, RR, y, batch, train: bool, seed=42):
    X = X.reshape(-1, 198, 1, 1)
    y = np.where(y >= 2, 2, y)  # Stage 1 merges V and F into one ventricular class
    yc = tf.keras.utils.to_categorical(y, num_classes=3)
    ds = tf.data.Dataset.from_tensor_slices(({"x1": X, "x2": RR}, yc))
    if train:
        ds = ds.shuffle(10000, seed=seed, reshuffle_each_iteration=True).batch(batch).map(
            augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE)


class ValMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds

    def on_epoch_end(self, epoch, logs=None):
        yt, yp = [], []
        for inputs, labels in self.val_ds:
            logits = self.model(inputs, training=False)
            yt.extend(np.argmax(labels.numpy(), 1))
            yp.extend(np.argmax(logits.numpy(), 1))
        score = f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0)
        if logs is not None:
            logs["val_macro_f1"] = score
        print(f" - val_macro_f1: {score:.4f}", end="")


def evaluate(model, X, RR, y, tag, outdir):
    logits = model.predict({"x1": X.reshape(-1, 198, 1, 1), "x2": RR}, batch_size=512, verbose=0)
    preds = np.argmax(logits, 1)
    y3 = np.where(y >= 2, 2, y)
    acc = float(accuracy_score(y3, preds))
    f1 = float(f1_score(y3, preds, labels=[0, 1, 2], average="macro"))
    pc = f1_score(y3, preds, labels=[0, 1, 2], average=None).tolist()
    cm = confusion_matrix(y3, preds, labels=[0, 1, 2])
    print(f"  [{tag}] acc={acc:.4f} macro-F1={f1:.4f} "
          f"N={pc[0]:.4f} S={pc[1]:.4f} VF={pc[2]:.4f}")
    print(classification_report(y3, preds, labels=[0, 1, 2],
                                target_names=["N", "S", "VF"], digits=4, zero_division=0))
    return {"accuracy": acc, "macro_f1": f1, "per_class_f1": pc, "confusion_matrix": cm.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="deployment_artifacts")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--splits-only", action="store_true", help="build the causal splits and exit")
    ap.add_argument("--init-weights", default=None,
                    help="warm-start from existing weights instead of random init (domain adaptation)")
    ap.add_argument("--out-weights", default=OUT_WEIGHTS)
    ap.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    tf.keras.utils.set_random_seed(42)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # ---- 1. splits ------------------------------------------------------
    ref = np.load(SRC_SPLITS)
    rr_train, rr_val, rr_test = ref["RR_train"], ref["RR_val"], ref["RR_test"]
    ref_y_train, ref_y_val, ref_y_test = ref["y_train"], ref["y_val"], ref["y_test"]

    if Path(OUT_SPLITS).is_file() and not args.rebuild_splits:
        print(f"loading cached causal splits {OUT_SPLITS}")
        d = np.load(OUT_SPLITS)
        X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
        y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
    else:
        print("rebuilding splits with the causal filter chain")
        sos_bp, sos_notch = build_sos()
        X_train, y_train = build_split(TRAIN_RECORDS, sos_bp, sos_notch, "train")
        X_val, y_val = build_split(VALIDATION_RECORDS, sos_bp, sos_notch, "val")
        X_test, y_test = build_split(TEST_RECORDS, sos_bp, sos_notch, "test")
        assert np.array_equal(y_train, ref_y_train), "train labels differ from reference split"
        assert np.array_equal(y_val, ref_y_val), "val labels differ from reference split"
        assert np.array_equal(y_test, ref_y_test), "test labels differ from reference split"
        np.savez(OUT_SPLITS, X_train=X_train, X_val=X_val, X_test=X_test,
                 y_train=y_train, y_val=y_val, y_test=y_test,
                 RR_train=rr_train, RR_val=rr_val, RR_test=rr_test,
                 rr_mean=ref["rr_mean"], rr_std=ref["rr_std"],
                 train_records=np.asarray(TRAIN_RECORDS),
                 validation_records=np.asarray(VALIDATION_RECORDS),
                 test_records=np.asarray(TEST_RECORDS),
                 preprocessing=np.asarray("causal butter4 0.5-40 Hz + notch50 Q30, forward only"))
        print(f"saved {OUT_SPLITS}")

    print(f"splits: train {X_train.shape} val {X_val.shape} test {X_test.shape}")
    if args.splits_only:
        print("splits-only mode: stopping before training")
        return

    # ---- 2. train -------------------------------------------------------
    model = build_model()
    print(f"parameters: {model.count_params():,}")
    if args.init_weights:
        model.load_weights(args.init_weights)
        print(f"warm start from {args.init_weights}")
    train_ds = make_ds(X_train, rr_train, y_train, args.batch, True)
    val_ds = make_ds(X_val, rr_val, y_val, args.batch, False)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4, clipnorm=1.0),
        loss=FocalLoss(),
        metrics=["accuracy"],
    )
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        args.out_weights, monitor="val_macro_f1", save_best_only=True,
        save_weights_only=True, mode="max", verbose=1,
    )
    hist = model.fit(
        train_ds, epochs=args.epochs, validation_data=val_ds,
        callbacks=[ValMacroF1(val_ds), ckpt,
                   tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", patience=args.patience,
                                                    restore_best_weights=True, mode="max", verbose=1),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                        patience=args.lr_patience, min_lr=1e-7,
                                                        verbose=1)],
        verbose=2,
    )
    model.load_weights(args.out_weights)

    # ---- 3. evaluate ----------------------------------------------------
    print("=" * 78)
    print("evaluation with the causally trained weights")
    res_causal = evaluate(model, X_test, rr_test, y_test, "causal DS2", outdir)
    x_zp = ref["X_test"].astype(np.float32)
    res_zero = evaluate(model, x_zp, rr_test, ref_y_test, "zero-phase DS2", outdir)

    report = {
        "weights": args.out_weights,
        "init_weights": args.init_weights,
        "splits": OUT_SPLITS,
        "preprocessing": "causal butter4 0.5-40 Hz + notch 50 Hz Q30, forward only, median/MAD norm",
        "parameters": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "best_val_macro_f1": float(max(hist.history.get("val_macro_f1", [0.0]))),
        "causal_test": res_causal,
        "zero_phase_test": res_zero,
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
    }
    (outdir / (Path(args.out_weights).stem + "_report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {outdir / (Path(args.out_weights).stem + '_report.json')}")


if __name__ == "__main__":
    main()
