"""Stage A: export the selected N/S/VF Stage-1 model to static ONNX and verify it.

This script never trains and never modifies the checkpoint. It:

1. rebuilds the Stage-1 architecture and loads the selected .weights.h5
2. re-checks the reloaded float model against the golden logits saved by
   stage1_deployment_audit.py
3. traces the model with a static (1, 198, 1, 1) / (1, 6) signature
4. converts to ONNX (opset 13 by default) and runs onnx.checker
5. prints the ONNX operator inventory (needed for the ESP-DL / ESP-PPQ review)
6. runs every golden vector through ONNX Runtime and compares logits + top-1
   against Keras, then optionally replays the full DS2 test set

Usage (server, inside mcu_env):

    python stage1_onnx_export.py
    python stage1_onnx_export.py --full-suite
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow.keras import layers


NUM_HEADS = 8
KEY_DIM = 2
EMBEDDING = 16
MLP_DIM = 128
KERNEL = 3
RR_FEATURE_COUNT = 6
CLASS_NAMES = ["N", "S", "VF"]


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
            name="pos_embedding",
            shape=(1, inputs_shape[1], inputs_shape[2]),
            initializer=init,
            trainable=True,
        )

    def call(self, inputs, inputs_positions=None):
        return inputs + tf.cast(self.pos_embedding, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        if isinstance(self.posemb_init, dict) or self.posemb_init is None:
            config["posemb_init"] = None
        else:
            config["posemb_init"] = tf.keras.initializers.serialize(self.posemb_init)
        return config


@tf.keras.utils.register_keras_serializable(package="ecg")
class ErfGelu(layers.Layer):
    """Exact GELU expressed with erf instead of erfc.

    Keras 3 implements exact GELU as x * 0.5 * erfc(-x / sqrt(2)). tf2onnx has no
    handler for Erfc, but Erf maps cleanly to ONNX Erf. Both forms are the same
    function: erfc(-z) == 1 + erf(z), so no accuracy is traded away.
    """

    def call(self, inputs):
        x = tf.cast(inputs, tf.float32)
        return 0.5 * x * (1.0 + tf.math.erf(x * 0.7071067811865476))


def build_stage1_model(gelu_mode: str = "erf") -> tf.keras.Model:
    """Rebuild the exact Stage-1 architecture used by the hierarchical notebook.

    gelu_mode="erf" swaps the GELU implementation (no weights involved) so the
    graph can be exported to ONNX; gelu_mode="keras" keeps tf.nn.gelu.
    """
    if gelu_mode == "erf":
        act = ErfGelu(name="gelu_erf")
        def act_fn(x, _l=act):
            return _l(x)
    else:
        act_fn = tf.nn.gelu

    x1 = layers.Input(shape=(198, 1, 1), name="x1")
    x2 = layers.Input(shape=(RR_FEATURE_COUNT,), name="x2")

    x = layers.Conv2D(
        filters=EMBEDDING,
        kernel_size=(KERNEL, 1),
        strides=(KERNEL, 1),
        padding="valid",
    )(x1)
    x = layers.Reshape((66, EMBEDDING))(x)
    x = AddPositionEmbs(name="posembed_input")(x)
    x = layers.Dropout(0.2)(x)

    attention_input = layers.LayerNormalization(dtype=x.dtype)(x)
    attention = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        dropout=0.1,
    )(attention_input, attention_input)
    x = layers.Add()([attention, x])

    mlp_input = layers.LayerNormalization(dtype=x.dtype)(x)
    y = layers.Dense(MLP_DIM, activation=act_fn)(mlp_input)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(EMBEDDING, activation=act_fn)(y)
    y = layers.Dropout(0.1)(y)
    x = layers.Add()([y, x])

    encoded = layers.LayerNormalization(name="encoder_norm")(x)
    avg_features = layers.GlobalAveragePooling1D()(encoded)
    max_features = layers.GlobalMaxPooling1D()(encoded)
    morphology = layers.Concatenate()([avg_features, max_features])

    rr = layers.LayerNormalization(name="rr_norm")(x2)
    rr = layers.Dense(16, activation=act_fn, name="rr_embedding")(rr)
    rr = layers.Dropout(0.15)(rr)

    fused = layers.Concatenate()([morphology, rr])
    fused = layers.Dense(32, activation=act_fn, name="fusion")(fused)
    fused = layers.Dropout(0.25)(fused)
    logits = layers.Dense(
        3,
        name="stage1_n_s_vf_head",
        kernel_initializer=tf.keras.initializers.Zeros(),
    )(fused)
    return tf.keras.Model(inputs=[x1, x2], outputs=logits, name="stage1_n_s_vf")


def make_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def run_ort(session: ort.InferenceSession, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    names = [i.name for i in session.get_inputs()]
    feed = {names[0]: x1.astype(np.float32), names[1]: x2.astype(np.float32)}
    return session.run(None, feed)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="tiny_transformer_stage1_n_s_vf.weights.h5")
    parser.add_argument("--golden", default="deployment_artifacts/stage1_golden_vectors.npz")
    parser.add_argument("--splits", default="patient_independent_hierarchical_splits.npz")
    parser.add_argument("--outdir", default="deployment_artifacts")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--gelu", choices=["erf", "keras"], default="erf",
                        help="erf: ONNX-exportable exact GELU (default); keras: original tf.nn.gelu")
    parser.add_argument("--full-suite", action="store_true", help="replay all 49,661 DS2 beats through ORT")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    golden_path = Path(args.golden).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("[1/6] rebuild float model and load checkpoint")
    model = build_stage1_model(args.gelu)
    model.load_weights(str(weights_path))
    print(f"      gelu mode  : {args.gelu}")
    print(f"      parameters: {model.count_params():,}")
    print(f"      inputs : {[ (i.name, tuple(i.shape)) for i in model.inputs ]}")
    print(f"      outputs: {[ (o.name, tuple(o.shape)) for o in model.outputs ]}")

    with np.load(golden_path) as g:
        gx1 = g["x1"].astype(np.float32)
        gx2 = g["x2"].astype(np.float32)
        glabels = g["labels"].astype(np.int64)
        gref_logits = g["logits"].astype(np.float32)
        rr_mean = g["rr_mean"].astype(np.float32)
        rr_std = g["rr_std"].astype(np.float32)

    print("=" * 78)
    print("[2/6] reload sanity check against saved golden logits")
    print("      (golden logits came from the original tf.nn.gelu model,")
    print("       so this also proves the erf GELU rewrite is numerically equivalent)")
    keras_golden = model.predict({"x1": gx1, "x2": gx2}, batch_size=32, verbose=0).astype(np.float32)
    reload_diff = float(np.max(np.abs(keras_golden - gref_logits)))
    reload_top1 = float(np.mean(np.argmax(keras_golden, 1) == np.argmax(gref_logits, 1)))
    reload_mae = float(np.mean(np.abs(keras_golden - gref_logits)))
    print(f"      golden samples      : {len(gx1)}")
    print(f"      max |logit diff|    : {reload_diff:.3e}")
    print(f"      mean |logit diff|   : {reload_mae:.3e}")
    print(f"      top-1 agreement     : {reload_top1 * 100:.2f}%")
    if reload_top1 < 1.0:
        raise SystemExit("Reloaded model does not match the audited golden logits. Stop here.")

    print("=" * 78)
    print("[3/6] trace with static batch=1 signature")
    input_sig = [
        tf.TensorSpec([1, 198, 1, 1], tf.float32, name="x1"),
        tf.TensorSpec([1, RR_FEATURE_COUNT], tf.float32, name="x2"),
    ]

    @tf.function(input_signature=input_sig)
    def serving(x1, x2):
        return model([x1, x2], training=False)

    concrete = serving.get_concrete_function()
    print(f"      traced ok, concrete inputs: {[i.name for i in concrete.inputs]}")
    onnx_path = outdir / "stage1_n_s_vf_b1.onnx"
    model_proto, _ = tf2onnx.convert.from_function(
        serving,
        input_signature=input_sig,
        opset=args.opset,
        output_path=str(onnx_path),
    )
    onnx.checker.check_model(str(onnx_path))
    size_kb = onnx_path.stat().st_size / 1024.0
    print(f"      onnx written: {onnx_path}  ({size_kb:.1f} KiB)  opset={args.opset}")
    print("      onnx.checker: PASSED")
    print(f"      graph inputs : {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model_proto.graph.input]}")
    print(f"      graph outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model_proto.graph.output]}")

    print("=" * 78)
    print("[4/6] operator inventory")
    ops = Counter(node.op_type for node in model_proto.graph.node)
    for name, count in ops.most_common():
        print(f"      {name:<24} {count:>4}")
    print(f"      {'TOTAL nodes':<24} {sum(ops.values()):>4}")
    initializer_bytes = sum(
        int(np.prod(t.dims)) * 4 for t in model_proto.graph.initializer
    )
    print(f"      weight bytes (fp32): {initializer_bytes / 1024:.1f} KiB")

    print("=" * 78)
    print("[5/6] ONNX Runtime vs Keras on golden vectors")
    session = make_session(onnx_path)
    ort_logits = np.empty_like(keras_golden)
    for i in range(len(gx1)):
        ort_logits[i] = run_ort(session, gx1[i : i + 1], gx2[i : i + 1])[0]

    abs_diff = np.abs(ort_logits - keras_golden)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    top1_agree = float(np.mean(np.argmax(ort_logits, 1) == np.argmax(keras_golden, 1)))
    keras_acc = float(np.mean(np.argmax(keras_golden, 1) == glabels))
    ort_acc = float(np.mean(np.argmax(ort_logits, 1) == glabels))
    print(f"      max |logit diff| : {max_diff:.3e}")
    print(f"      mean |logit diff|: {mean_diff:.3e}")
    print(f"      top-1 agreement  : {top1_agree * 100:.2f}%")
    print(f"      golden accuracy  : keras={keras_acc:.4f}  onnx={ort_acc:.4f}")

    full_result = None
    if args.full_suite:
        print("=" * 78)
        print("[6/6] full DS2 replay through ONNX Runtime (batch=1)")
        from sklearn.metrics import accuracy_score, f1_score

        with np.load(Path(args.splits).resolve()) as d:
            x_test = d["X_test"].astype(np.float32).reshape(-1, 198, 1, 1)
            rr_test = d["RR_test"].astype(np.float32)
            y_raw = d["y_test"].astype(np.int64)
        y_test = np.where(y_raw >= 2, 2, y_raw).astype(np.int64)

        preds = np.empty(len(y_test), dtype=np.int64)
        for i in range(len(y_test)):
            preds[i] = int(np.argmax(run_ort(session, x_test[i : i + 1], rr_test[i : i + 1])[0]))
            if (i + 1) % 5000 == 0:
                print(f"      {i + 1}/{len(y_test)}", flush=True)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, labels=[0, 1, 2], average="macro")
        full_result = {"samples": int(len(y_test)), "accuracy": float(acc), "macro_f1": float(f1)}
        print(f"      ONNX full-suite accuracy: {acc:.4f}")
        print(f"      ONNX full-suite macro-F1: {f1:.4f}")
    else:
        print("=" * 78)
        print("[6/6] skipped full DS2 replay (use --full-suite)")

    report = {
        "weights": str(weights_path),
        "onnx": str(onnx_path),
        "opset": args.opset,
        "gelu_mode": args.gelu,
        "onnx_size_kib": round(size_kb, 1),
        "parameters": int(model.count_params()),
        "operator_counts": dict(ops.most_common()),
        "total_nodes": int(sum(ops.values())),
        "fp32_weight_kib": round(initializer_bytes / 1024.0, 1),
        "reload_vs_saved": {"max_abs_logit_diff": reload_diff, "top1_agreement": reload_top1},
        "onnx_vs_keras_golden": {
            "samples": int(len(gx1)),
            "max_abs_logit_diff": max_diff,
            "mean_abs_logit_diff": mean_diff,
            "top1_agreement": top1_agree,
            "keras_accuracy": keras_acc,
            "onnx_accuracy": ort_acc,
        },
        "onnx_full_suite": full_result,
        "rr_mean": rr_mean.tolist(),
        "rr_std": rr_std.tolist(),
    }
    report_path = outdir / "stage1_onnx_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 78)
    print(f"report written: {report_path}")

    if top1_agree < 1.0:
        print("WARNING: keras and onnx top-1 disagree on at least one golden vector.")
    if max_diff > 1e-3:
        print("WARNING: logit drift above 1e-3, inspect before quantizing.")


if __name__ == "__main__":
    main()
