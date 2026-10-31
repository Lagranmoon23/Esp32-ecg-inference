"""Audit the selected N/S/(V+F) model and create deployment test vectors.

This script does not train or change the model. Run it in the directory that
contains the hierarchical split file and the selected Stage-1 weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras import layers


NUM_HEADS = 8
KEY_DIM = 2
EMBEDDING = 16
MLP_DIM = 128
KERNEL = 3
RR_FEATURE_COUNT = 6
CLASS_NAMES = np.asarray(["N", "S", "VF"])


@tf.keras.utils.register_keras_serializable(package="ecg")
class AddPositionEmbs(layers.Layer):
    def __init__(self, posemb_init=None, **kwargs):
        super().__init__(**kwargs)
        self.posemb_init = posemb_init or tf.keras.initializers.RandomNormal(stddev=0.02)

    def build(self, inputs_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, inputs_shape[1], inputs_shape[2]),
            initializer=self.posemb_init,
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pos_embedding, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config["posemb_init"] = tf.keras.initializers.serialize(self.posemb_init)
        return config


def build_stage1_model() -> tf.keras.Model:
    """Rebuild the exact Stage-1 architecture used by the hierarchical notebook."""
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
    y = layers.Dense(MLP_DIM, activation=tf.nn.gelu)(mlp_input)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(EMBEDDING, activation=tf.nn.gelu)(y)
    y = layers.Dropout(0.1)(y)
    x = layers.Add()([y, x])

    encoded = layers.LayerNormalization(name="encoder_norm")(x)
    avg_features = layers.GlobalAveragePooling1D()(encoded)
    max_features = layers.GlobalMaxPooling1D()(encoded)
    morphology = layers.Concatenate()([avg_features, max_features])

    rr = layers.LayerNormalization(name="rr_norm")(x2)
    rr = layers.Dense(16, activation=tf.nn.gelu, name="rr_embedding")(rr)
    rr = layers.Dropout(0.15)(rr)

    fused = layers.Concatenate()([morphology, rr])
    fused = layers.Dense(32, activation=tf.nn.gelu, name="fusion")(fused)
    fused = layers.Dropout(0.25)(fused)
    logits = layers.Dense(
        3,
        name="stage1_n_s_vf_head",
        kernel_initializer=tf.keras.initializers.Zeros(),
    )(fused)
    return tf.keras.Model(inputs=[x1, x2], outputs=logits, name="stage1_n_s_vf")


def choose_golden_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(3):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < per_class:
            raise ValueError(
                f"Class {CLASS_NAMES[class_id]} has only {len(candidates)} test samples; "
                f"cannot select {per_class}."
            )
        selected.extend(rng.choice(candidates, size=per_class, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="patient_independent_hierarchical_splits.npz")
    parser.add_argument("--weights", default="tiny_transformer_stage1_n_s_vf.weights.h5")
    parser.add_argument("--output", default="deployment_artifacts")
    parser.add_argument("--golden-per-class", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_path = Path(args.splits).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {weights_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(split_path) as data:
        x_test = data["X_test"].astype(np.float32).reshape(-1, 198, 1, 1)
        rr_test = data["RR_test"].astype(np.float32)
        y_original = data["y_test"].astype(np.int64)
        rr_mean = data["rr_mean"].astype(np.float32)
        rr_std = data["rr_std"].astype(np.float32)

    y_test = np.where(y_original >= 2, 2, y_original).astype(np.int64)
    model = build_stage1_model()
    model.load_weights(weights_path)
    logits = model.predict({"x1": x_test, "x2": rr_test}, batch_size=args.batch_size, verbose=1)
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, labels=[0, 1, 2], average="macro")
    weighted_f1 = f1_score(y_test, predictions, labels=[0, 1, 2], average="weighted")
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES.tolist(),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])

    results = {
        "model": "stage1_n_s_vf",
        "weights": str(weights_path),
        "splits": str(split_path),
        "parameters": int(model.count_params()),
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    result_path = output_dir / "stage1_float_audit.json"
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    golden_indices = choose_golden_indices(y_test, args.golden_per_class, args.seed)
    golden_path = output_dir / "stage1_golden_vectors.npz"
    np.savez(
        golden_path,
        x1=x_test[golden_indices],
        x2=rr_test[golden_indices],
        labels=y_test[golden_indices],
        logits=logits[golden_indices].astype(np.float32),
        predictions=predictions[golden_indices].astype(np.int64),
        source_indices=golden_indices,
        rr_mean=rr_mean,
        rr_std=rr_std,
        class_names=CLASS_NAMES,
    )

    keras_path = output_dir / "stage1_n_s_vf.keras"
    model.save(keras_path)

    print("\nStage-1 deployment audit complete")
    print(f"Parameters:  {model.count_params():,}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Results:     {result_path}")
    print(f"Golden set:  {golden_path}")
    print(f"Full model:  {keras_path}")


if __name__ == "__main__":
    main()
