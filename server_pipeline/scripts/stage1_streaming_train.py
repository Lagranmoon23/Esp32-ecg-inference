"""Fine-tune Stage-1 on the exact causal rolling-normalized input pipeline.

Safety against test leakage:
  * only train_records and validation_records are loaded;
  * no DS2 waveform or label is read;
  * model selection uses validation macro-F1 only.

Conservative training augmentation models deployment error:
  * most beats are shifted by up to 3 samples around the annotated R peak;
  * a small minority are shifted by up to 10 samples;
  * RR features receive small timing noise;
  * waveform amplitude/noise/baseline perturbations are deliberately weak.

The fine-tuned candidate never silently replaces the initial model.  A
separate selected-weights file is written from whichever model has the better
validation macro-F1 (subject to --min-improvement).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import wfdb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from stage1_deployment_audit import build_stage1_model
from stage1_streaming_pipeline_study import (
    AAMI,
    CLASS_NAMES,
    FS,
    HALF_WINDOW,
    OUTPUT_CLASSES,
    build_filter_chain,
    causal_filter_continuous,
    choose_lead,
    class_id,
    normalize_rolling_robust,
    rr_features,
)


DEFAULT_DATA = (
    "/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/"
    "wym_mcu/MIT-BIH/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0"
)
DEFAULT_REFERENCE_SPLITS = "patient_independent_hierarchical_splits.npz"
DEFAULT_STREAMING_SPLITS = "patient_independent_hierarchical_streaming_train_val.npz"
DEFAULT_INIT_WEIGHTS = "tiny_transformer_stage1_n_s_vf.weights.h5"
DEFAULT_CANDIDATE_WEIGHTS = "tiny_transformer_stage1_n_s_vf_streaming_candidate.weights.h5"
DEFAULT_SELECTED_WEIGHTS = "tiny_transformer_stage1_n_s_vf_streaming_selected.weights.h5"
MAX_PEAK_SHIFT = 15
CONTEXT_LENGTH = 2 * HALF_WINDOW + 2 * MAX_PEAK_SHIFT
NARROW_PEAK_SHIFT = 3
WIDE_PEAK_SHIFT = 10
WIDE_SHIFT_PROBABILITY = 0.10
RR_NOISE_RAW_STD = np.asarray(
    [0.005, 0.005, 0.005, 0.005, 0.008, 0.008], dtype=np.float32
)
TRAINABLE_HEAD_LAYERS = {
    "encoder_norm",
    "rr_norm",
    "rr_embedding",
    "fusion",
    "stage1_n_s_vf_head",
}


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=(0.20, 0.70, 0.45), gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = float(gamma)

    def call(self, y_true, logits):
        probabilities = tf.nn.softmax(logits, axis=-1)
        p_t = tf.reduce_sum(y_true * probabilities, axis=-1)
        alpha_t = tf.reduce_sum(y_true * self.alpha, axis=-1)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=logits
        )
        return alpha_t * tf.pow(1.0 - p_t, self.gamma) * cross_entropy


def extract_record(
    data_path: Path,
    record_id: str,
    bandpass: np.ndarray,
    notch: np.ndarray,
    rolling_window_seconds: float,
    rolling_update_seconds: float,
    rolling_warmup_seconds: float,
    clip: float,
    training_context: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    record = wfdb.rdrecord(str(data_path / record_id))
    annotation = wfdb.rdann(str(data_path / record_id), "atr")
    if int(round(float(record.fs))) != FS:
        raise ValueError(f"Record {record_id} has fs={record.fs}; expected {FS}")

    raw = record.p_signal[:, choose_lead(record)]
    filtered = causal_filter_continuous(raw, bandpass, notch)
    normalized = normalize_rolling_robust(
        filtered,
        FS,
        rolling_window_seconds,
        rolling_update_seconds,
        rolling_warmup_seconds,
        clip,
    )

    mapped = [
        (int(sample), AAMI[symbol])
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in AAMI
    ]
    peaks = np.asarray([item[0] for item in mapped], dtype=np.int64)
    symbols = [item[1] for item in mapped]
    margin = HALF_WINDOW + (MAX_PEAK_SHIFT if training_context else 0)

    beats: list[np.ndarray] = []
    rhythm: list[np.ndarray] = []
    labels: list[int] = []
    for index in range(1, len(peaks) - 1):
        symbol = symbols[index]
        if symbol not in OUTPUT_CLASSES:
            continue
        start = int(peaks[index]) - margin
        end = int(peaks[index]) + margin
        if start < 0 or end >= len(normalized):
            continue
        beat = normalized[start:end]
        if not np.all(np.isfinite(beat)):
            continue
        expected = CONTEXT_LENGTH if training_context else 2 * HALF_WINDOW
        if len(beat) != expected:
            raise AssertionError(f"Unexpected beat length {len(beat)} != {expected}")
        beats.append(beat.astype(np.float32))
        rhythm.append(rr_features(peaks, index, FS))
        labels.append(class_id(symbol))
    return beats, rhythm, labels


def build_split(
    records: list[str],
    split_name: str,
    data_path: Path,
    bandpass: np.ndarray,
    notch: np.ndarray,
    args: argparse.Namespace,
    training_context: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beats: list[np.ndarray] = []
    rhythm: list[np.ndarray] = []
    labels: list[int] = []
    for record_id in records:
        record_beats, record_rr, record_labels = extract_record(
            data_path,
            record_id,
            bandpass,
            notch,
            args.rolling_window_seconds,
            args.rolling_update_seconds,
            args.rolling_warmup_seconds,
            args.beat_clip,
            training_context,
        )
        beats.extend(record_beats)
        rhythm.extend(record_rr)
        labels.extend(record_labels)
        print(
            f"  {split_name} {record_id}: {len(record_labels)} beats "
            f"{dict(Counter(record_labels))}",
            flush=True,
        )
    return (
        np.asarray(beats, dtype=np.float32),
        np.asarray(rhythm, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def build_or_load_splits(args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache_path = Path(args.streaming_splits).resolve()
    if cache_path.is_file() and not args.rebuild_splits:
        print(f"Loading streaming train/validation cache: {cache_path}")
        with np.load(cache_path) as cached:
            expected = {
                "rolling_window_seconds": args.rolling_window_seconds,
                "rolling_update_seconds": args.rolling_update_seconds,
                "rolling_warmup_seconds": args.rolling_warmup_seconds,
                "beat_clip": args.beat_clip,
                "max_peak_shift": MAX_PEAK_SHIFT,
            }
            for key, value in expected.items():
                if key not in cached or not np.isclose(float(cached[key]), float(value)):
                    raise ValueError(
                        f"Cached {key} does not match requested value {value}; "
                        "rerun with --rebuild-splits."
                    )
            return {key: cached[key] for key in cached.files}

    reference_path = Path(args.reference_splits).resolve()
    with np.load(reference_path) as reference:
        train_records = [str(item) for item in reference["train_records"]]
        validation_records = [str(item) for item in reference["validation_records"]]
        rr_mean = reference["rr_mean"].astype(np.float32).reshape(1, 6)
        rr_std = reference["rr_std"].astype(np.float32).reshape(1, 6)

    print("Building deployment-realistic train/validation data (DS2 is not read)")
    bandpass, notch = build_filter_chain(FS)
    data_path = Path(args.data).resolve()
    x_train_context, rr_train_raw, y_train = build_split(
        train_records, "train", data_path, bandpass, notch, args, True
    )
    x_val, rr_val_raw, y_val = build_split(
        validation_records, "validation", data_path, bandpass, notch, args, False
    )
    rr_train = ((rr_train_raw - rr_mean) / rr_std).astype(np.float32)
    rr_val = ((rr_val_raw - rr_mean) / rr_std).astype(np.float32)

    np.savez(
        cache_path,
        X_train_context=x_train_context,
        X_val=x_val,
        RR_train=rr_train,
        RR_val=rr_val,
        y_train=y_train,
        y_val=y_val,
        rr_mean=rr_mean,
        rr_std=rr_std,
        train_records=np.asarray(train_records),
        validation_records=np.asarray(validation_records),
        preprocessing=np.asarray(
            "causal butter4 0.5-40 Hz + notch50 Q30; rolling median/MAD"
        ),
        rolling_window_seconds=np.asarray(args.rolling_window_seconds),
        rolling_update_seconds=np.asarray(args.rolling_update_seconds),
        rolling_warmup_seconds=np.asarray(args.rolling_warmup_seconds),
        beat_clip=np.asarray(args.beat_clip),
        max_peak_shift=np.asarray(MAX_PEAK_SHIFT),
    )
    print(f"Saved streaming cache: {cache_path}")
    return {
        "X_train_context": x_train_context,
        "X_val": x_val,
        "RR_train": rr_train,
        "RR_val": rr_val,
        "y_train": y_train,
        "y_val": y_val,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "train_records": np.asarray(train_records),
        "validation_records": np.asarray(validation_records),
    }


def augment_batch(
    inputs: dict[str, tf.Tensor],
    labels: tf.Tensor,
    rr_noise_std: tf.Tensor,
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    context = inputs["x1"]
    batch_size = tf.shape(context)[0]
    narrow_shift = tf.random.uniform(
        [batch_size],
        minval=-NARROW_PEAK_SHIFT,
        maxval=NARROW_PEAK_SHIFT + 1,
        dtype=tf.int32,
    )
    wide_shift = tf.random.uniform(
        [batch_size], minval=-WIDE_PEAK_SHIFT, maxval=WIDE_PEAK_SHIFT + 1,
        dtype=tf.int32,
    )
    use_wide = (
        tf.random.uniform([batch_size], 0.0, 1.0) < WIDE_SHIFT_PROBABILITY
    )
    shifts = tf.where(use_wide, wide_shift, narrow_shift)
    starts = MAX_PEAK_SHIFT + shifts

    def crop_one(values: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        beat_context, start = values
        return beat_context[start:start + 2 * HALF_WINDOW]

    beats = tf.map_fn(
        crop_one,
        (context, starts),
        fn_output_signature=tf.TensorSpec((2 * HALF_WINDOW, 1, 1), tf.float32),
    )
    shape = tf.stack([batch_size, 1, 1, 1])
    amplitude = tf.random.uniform(shape, 0.96, 1.04)
    noise = tf.random.normal(tf.shape(beats), stddev=0.008, dtype=tf.float32)
    phase = tf.random.uniform(shape, 0.0, 2.0 * np.pi, dtype=tf.float32)
    time_axis = tf.reshape(
        tf.linspace(0.0, 2.0 * np.pi, 2 * HALF_WINDOW),
        [1, 2 * HALF_WINDOW, 1, 1],
    )
    beats = beats * amplitude + noise + 0.010 * tf.sin(time_axis + phase)
    beats = tf.clip_by_value(beats, -5.0, 5.0)

    rr_noise = tf.random.normal(tf.shape(inputs["x2"]), dtype=tf.float32) * rr_noise_std
    rhythm = inputs["x2"] + rr_noise
    return {"x1": beats, "x2": rhythm}, labels


def make_train_dataset(
    x_context: np.ndarray,
    rr: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rr_std: np.ndarray,
    seed: int,
) -> tf.data.Dataset:
    x_context = x_context.reshape(-1, CONTEXT_LENGTH, 1, 1)
    categorical = tf.keras.utils.to_categorical(labels, num_classes=3)
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"x1": x_context, "x2": rr}, categorical)
    )
    # About 5 ms noise for RR times and small noise for RR ratios, converted
    # to the already-standardized feature space.
    standardized_noise = tf.constant(
        RR_NOISE_RAW_STD / rr_std.reshape(6).astype(np.float32), dtype=tf.float32
    )
    dataset = dataset.shuffle(
        min(10000, len(labels)), seed=seed, reshuffle_each_iteration=True
    ).batch(batch_size)
    dataset = dataset.map(
        lambda x, y: augment_batch(x, y, standardized_noise),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def make_validation_dataset(
    x: np.ndarray,
    rr: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    x = x.reshape(-1, 2 * HALF_WINDOW, 1, 1)
    categorical = tf.keras.utils.to_categorical(labels, num_classes=3)
    return (
        tf.data.Dataset.from_tensor_slices(({"x1": x, "x2": rr}, categorical))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


class ValidationMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset: tf.data.Dataset):
        super().__init__()
        self.validation_dataset = validation_dataset

    def on_epoch_end(self, epoch, logs=None):
        true_labels: list[int] = []
        predictions: list[int] = []
        for inputs, labels in self.validation_dataset:
            logits = self.model(inputs, training=False)
            true_labels.extend(np.argmax(labels.numpy(), axis=1).tolist())
            predictions.extend(np.argmax(logits.numpy(), axis=1).tolist())
        score = f1_score(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
        if logs is not None:
            logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}", end="")


def configure_trainable_layers(model: tf.keras.Model, mode: str) -> list[str]:
    """Freeze the feature extractor for conservative deployment adaptation."""
    if mode == "all":
        for layer in model.layers:
            layer.trainable = True
    elif mode == "head":
        for layer in model.layers:
            layer.trainable = layer.name in TRAINABLE_HEAD_LAYERS
    else:
        raise ValueError(f"Unsupported trainable mode: {mode}")

    trainable_names = [
        layer.name for layer in model.layers if layer.trainable and layer.weights
    ]
    if not model.trainable_weights:
        raise RuntimeError("No trainable weights remain after freezing layers")
    return trainable_names


def evaluate(
    model: tf.keras.Model,
    x: np.ndarray,
    rr: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, Any]:
    logits = model.predict(
        {"x1": x.reshape(-1, 2 * HALF_WINDOW, 1, 1), "x2": rr},
        batch_size=batch_size,
        verbose=0,
    )
    predictions = np.argmax(logits, axis=1)
    per_class = f1_score(
        labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "per_class_f1": per_class.tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--reference-splits", default=DEFAULT_REFERENCE_SPLITS)
    parser.add_argument("--streaming-splits", default=DEFAULT_STREAMING_SPLITS)
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--splits-only", action="store_true")
    parser.add_argument("--init-weights", default=DEFAULT_INIT_WEIGHTS)
    parser.add_argument(
        "--candidate-weights",
        "--out-weights",
        dest="candidate_weights",
        default=DEFAULT_CANDIDATE_WEIGHTS,
        help="Best fine-tuned candidate; kept even when it loses to the initial model.",
    )
    parser.add_argument("--selected-weights", default=DEFAULT_SELECTED_WEIGHTS)
    parser.add_argument(
        "--report", default="deployment_artifacts/stage1_streaming_train_report.json"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--trainable-mode", choices=("head", "all"), default="head")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.002,
        help="Required validation macro-F1 gain before selecting the candidate.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rolling-window-seconds", type=float, default=10.0)
    parser.add_argument("--rolling-update-seconds", type=float, default=1.0)
    parser.add_argument("--rolling-warmup-seconds", type=float, default=10.0)
    parser.add_argument("--beat-clip", type=float, default=5.0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    data = build_or_load_splits(args)
    print(
        f"Streaming data: train={data['X_train_context'].shape}, "
        f"validation={data['X_val'].shape}"
    )
    if args.splits_only:
        print("Splits-only mode; DS2 was not read and training was not started.")
        return

    train_dataset = make_train_dataset(
        data["X_train_context"],
        data["RR_train"],
        data["y_train"],
        args.batch_size,
        data["rr_std"],
        args.seed,
    )
    validation_dataset = make_validation_dataset(
        data["X_val"], data["RR_val"], data["y_val"], args.batch_size
    )

    model = build_stage1_model()
    model.load_weights(args.init_weights)
    baseline_validation = evaluate(
        model, data["X_val"], data["RR_val"], data["y_val"], args.batch_size
    )
    print(
        "Initial streaming validation: "
        f"accuracy={baseline_validation['accuracy']:.4f} "
        f"macro-F1={baseline_validation['macro_f1']:.4f}"
    )

    trainable_layers = configure_trainable_layers(model, args.trainable_mode)
    trainable_parameters = int(
        sum(np.prod(variable.shape) for variable in model.trainable_weights)
    )
    print(
        f"Trainable mode={args.trainable_mode}: {trainable_parameters:,}/"
        f"{model.count_params():,} parameters; layers={trainable_layers}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=1e-5, clipnorm=1.0
        ),
        loss=FocalLoss(),
        metrics=["accuracy"],
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.candidate_weights,
        monitor="val_macro_f1",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1,
    )
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset,
        callbacks=[
            ValidationMacroF1(validation_dataset),
            checkpoint,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                patience=args.patience,
                restore_best_weights=True,
                mode="max",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=args.lr_patience,
                min_lr=1e-7,
                verbose=1,
            ),
        ],
        verbose=2,
    )
    model.load_weights(args.candidate_weights)
    candidate_validation = evaluate(
        model, data["X_val"], data["RR_val"], data["y_val"], args.batch_size
    )
    print(
        "Candidate streaming validation: "
        f"accuracy={candidate_validation['accuracy']:.4f} "
        f"macro-F1={candidate_validation['macro_f1']:.4f} "
        f"per-class={candidate_validation['per_class_f1']}"
    )

    improvement = (
        candidate_validation["macro_f1"] - baseline_validation["macro_f1"]
    )
    candidate_selected = improvement >= args.min_improvement
    selected_source = (
        Path(args.candidate_weights) if candidate_selected else Path(args.init_weights)
    ).resolve()
    selected_path = Path(args.selected_weights).resolve()
    if selected_source != selected_path:
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_source, selected_path)
    selection_reason = (
        f"candidate improved validation macro-F1 by {improvement:+.4f}"
        if candidate_selected
        else (
            f"candidate gain {improvement:+.4f} was below required "
            f"{args.min_improvement:+.4f}; retained initial model"
        )
    )
    print(f"Selection: {selection_reason}")

    report = {
        "model": "stage1_n_s_vf_streaming",
        "test_split_read": False,
        "init_weights": str(Path(args.init_weights).resolve()),
        "candidate_weights": str(Path(args.candidate_weights).resolve()),
        "selected_weights": str(selected_path),
        "selected_source": str(selected_source),
        "candidate_selected": bool(candidate_selected),
        "selection_reason": selection_reason,
        "validation_macro_f1_improvement": float(improvement),
        "minimum_required_improvement": float(args.min_improvement),
        "streaming_splits": str(Path(args.streaming_splits).resolve()),
        "parameters": int(model.count_params()),
        "preprocessing": {
            "filter": "causal butter4 0.5-40 Hz + notch50 Q30",
            "rolling_window_seconds": args.rolling_window_seconds,
            "rolling_update_seconds": args.rolling_update_seconds,
            "rolling_warmup_seconds": args.rolling_warmup_seconds,
            "clip": args.beat_clip,
        },
        "augmentation": {
            "peak_shift_narrow_samples": NARROW_PEAK_SHIFT,
            "peak_shift_wide_samples": WIDE_PEAK_SHIFT,
            "wide_probability": WIDE_SHIFT_PROBABILITY,
            "rr_noise_raw_std": RR_NOISE_RAW_STD.tolist(),
            "amplitude_range": [0.96, 1.04],
            "waveform_noise_std": 0.008,
            "baseline_sine_amplitude": 0.010,
        },
        "training": {
            "trainable_mode": args.trainable_mode,
            "trainable_layers": trainable_layers,
            "trainable_parameters": trainable_parameters,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-5,
        },
        "epochs_run": int(len(history.history["loss"])),
        "best_val_macro_f1_callback": float(
            max(history.history.get("val_macro_f1", [0.0]))
        ),
        "initial_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "selected_validation": (
            candidate_validation if candidate_selected else baseline_validation
        ),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Candidate weights: {Path(args.candidate_weights).resolve()}")
    print(f"Selected weights:  {selected_path}")
    print(f"Report:  {report_path}")
    print("DS2 was not read. Do not evaluate it until the model is frozen.")


if __name__ == "__main__":
    main()
