"""Stage B: quantize the deploy ONNX graph for ESP32-S3 with ESP-PPQ.

Rules enforced here:
  * calibration data comes from X_train/X_val (DS1) only
  * the DS2 test split is used for evaluation only, never for calibration
  * the float reference metrics (0.9202 acc / 0.6845 macro-F1) stay printed next
    to the quantized numbers so the drift is explicit

Outputs:
  deployment_artifacts/stage1_n_s_vf_<quant>.espdl   model for ESP-IDF
  deployment_artifacts/stage1_<quant>_report.json    metrics + config
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

FLOAT_ACC = 0.9202
FLOAT_F1 = 0.6845
CLASS_NAMES = ["N", "S", "VF"]


class CalibDataset(Dataset):
    """Yields [x1, x2] pairs so the default collate gives a two-tensor batch."""

    def __init__(self, x1: np.ndarray, x2: np.ndarray) -> None:
        self.x1 = torch.from_numpy(np.ascontiguousarray(x1, dtype=np.float32))
        self.x2 = torch.from_numpy(np.ascontiguousarray(x2, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.x1)

    def __getitem__(self, idx: int):
        return [self.x1[idx], self.x2[idx]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="deployment_artifacts/stage1_n_s_vf_deploy.onnx")
    ap.add_argument("--splits", default="patient_independent_hierarchical_splits.npz")
    ap.add_argument("--golden", default="deployment_artifacts/stage1_golden_vectors.npz")
    ap.add_argument("--outdir", default="deployment_artifacts")
    ap.add_argument("--target", default="esp32s3")
    ap.add_argument("--quant-type", default="w8a8", choices=["w8a8", "w8a16", "w16a16"])
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--eval-samples", type=int, default=0, help="0 = full DS2 test set")
    ap.add_argument("--per-class", type=int, default=1500,
                    help="stratified evaluation size per class (ignored when --eval-samples > 0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    onnx_in = Path(args.onnx).resolve()
    # espdl_quantize_onnx runs onnxsim in place, so quantize a copy
    onnx_work = outdir / f"_{args.quant_type}_work.onnx"
    shutil.copyfile(onnx_in, onnx_work)
    espdl_out = outdir / f"stage1_n_s_vf_{args.quant_type}.espdl"

    with np.load(Path(args.splits).resolve()) as d:
        x_train = d["X_train"].astype(np.float32).reshape(-1, 198, 1, 1)
        rr_train = d["RR_train"].astype(np.float32)
        x_test = d["X_test"].astype(np.float32).reshape(-1, 198, 1, 1)
        rr_test = d["RR_test"].astype(np.float32)
        y_raw = d["y_test"].astype(np.int64)
        train_records = d["train_records"] if "train_records" in d.files else None
        test_records = d["test_records"] if "test_records" in d.files else None
    y_test = np.where(y_raw >= 2, 2, y_raw).astype(np.int64)
    if train_records is not None:
        assert not set(map(str, train_records)) & set(map(str, test_records)), "DS1/DS2 overlap!"

    rng = np.random.default_rng(args.seed)
    calib_idx = rng.choice(len(x_train), size=min(args.calib_samples, len(x_train)), replace=False)
    calib_loader = DataLoader(
        CalibDataset(x_train[calib_idx], rr_train[calib_idx]), batch_size=1, shuffle=False
    )
    print("=" * 78)
    print(f"calibration: {len(calib_idx)} beats from DS1 train "
          f"(records {sorted(set(map(str, train_records.tolist())))[:5]}...)" if train_records is not None else
          f"calibration: {len(calib_idx)} beats from DS1 train")
    print(f"quant      : target={args.target} type={args.quant_type}")

    from esp_ppq.api import espdl_quantize_onnx

    t0 = time.time()
    graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_work),
        espdl_export_file=str(espdl_out),
        calib_dataloader=calib_loader,
        calib_steps=len(calib_idx),
        input_shape=[[1, 198, 1, 1], [1, 6]],
        target=args.target,
        quant_type=args.quant_type,
        device="cpu",
        error_report=True,
        export_config=True,
    )
    print(f"quantization finished in {time.time() - t0:.1f}s -> {espdl_out} "
          f"({espdl_out.stat().st_size / 1024:.1f} KiB)")

    # ESP-PPQ may fuse the primitive self-attention subgraph into an
    # `Attention` op. ESP-DL 3.3.14 has no corresponding runtime module on
    # ESP32-S3, even though the exporter accepts it. Reject such an artifact
    # before it can be copied into firmware.
    espdl_bytes = espdl_out.read_bytes()
    if b"Attention" in espdl_bytes:
        raise RuntimeError(
            "Generated ESP-DL still contains unsupported Attention. "
            "Rebuild the deploy ONNX with the primitive-attention guard."
        )
    print("runtime operator guard: PASS (no fused Attention)")

    # ---- evaluate: stratified subset, float and quantized on the same rows --
    from esp_ppq.executor import TorchExecutor
    import onnxruntime as ort

    if args.eval_samples > 0:
        eval_idx = np.arange(min(args.eval_samples, len(y_test)))
    else:
        eval_idx = []
        for c in range(3):
            cand = np.flatnonzero(y_test == c)
            take = min(args.per_class, len(cand))
            eval_idx.extend(rng.choice(cand, size=take, replace=False).tolist())
        eval_idx = np.sort(np.asarray(eval_idx, dtype=np.int64))

    yt = y_test[eval_idx]
    print("=" * 78)
    print(f"evaluating {len(eval_idx)} DS2 beats "
          f"(class counts {np.bincount(yt, minlength=3).tolist()})")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    fsess = ort.InferenceSession(str(onnx_in), sess_options=so, providers=["CPUExecutionProvider"])
    float_preds = np.empty(len(eval_idx), dtype=np.int64)
    for k, i in enumerate(eval_idx):
        float_preds[k] = int(np.argmax(
            fsess.run(["logits"], {"x1": x_test[i:i+1], "x2": rr_test[i:i+1]})[0][0]
        ))

    executor = TorchExecutor(graph=graph, device="cpu")
    quant_preds = np.empty(len(eval_idx), dtype=np.int64)
    quant_eval_logits = np.empty((len(eval_idx), 3), dtype=np.float32)
    t0 = time.time()
    for k, i in enumerate(eval_idx):
        out = executor.forward(
            inputs=[torch.from_numpy(x_test[i:i+1]), torch.from_numpy(rr_test[i:i+1])]
        )
        quant_eval_logits[k] = np.asarray(out[0]).reshape(3)
        quant_preds[k] = int(np.argmax(quant_eval_logits[k]))
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(eval_idx)}  ({time.time()-t0:.0f}s)", flush=True)

    # Export reference logits from the quantized PPQ graph itself. Firmware
    # must be compared with these values, not with the pre-quantization Keras
    # logits stored in the original golden file.
    golden_in = Path(args.golden).resolve()
    with np.load(golden_in) as golden:
        golden_payload = {name: golden[name] for name in golden.files}
    golden_x1 = np.ascontiguousarray(
        golden_payload["x1"], dtype=np.float32
    ).reshape(-1, 198, 1, 1)
    golden_x2 = np.ascontiguousarray(
        golden_payload["x2"], dtype=np.float32
    ).reshape(-1, 6)
    quant_golden_logits = np.empty((len(golden_x1), 3), dtype=np.float32)
    for i in range(len(golden_x1)):
        out = executor.forward(
            inputs=[torch.from_numpy(golden_x1[i:i+1]),
                    torch.from_numpy(golden_x2[i:i+1])]
        )
        quant_golden_logits[i] = np.asarray(out[0]).reshape(3)

    if "logits" in golden_payload:
        golden_payload["prequant_logits"] = golden_payload["logits"]
    if "predictions" in golden_payload:
        golden_payload["prequant_predictions"] = golden_payload["predictions"]
    golden_payload["logits"] = quant_golden_logits
    golden_payload["predictions"] = np.argmax(
        quant_golden_logits, axis=1
    ).astype(np.int64)
    golden_payload["reference_quant_type"] = np.asarray(args.quant_type)
    golden_payload["reference_espdl"] = np.asarray(str(espdl_out))
    golden_reference_path = outdir / f"stage1_golden_vectors_{args.quant_type}.npz"
    np.savez(golden_reference_path, **golden_payload)
    print(f"quantized golden reference: {golden_reference_path}")

    def metrics(preds):
        return (accuracy_score(yt, preds),
                f1_score(yt, preds, labels=[0, 1, 2], average="macro"),
                f1_score(yt, preds, labels=[0, 1, 2], average=None),
                confusion_matrix(yt, preds, labels=[0, 1, 2]))

    f_acc, f_f1, f_pc, _ = metrics(float_preds)
    q_acc, q_f1, q_pc, cm = metrics(quant_preds)

    print("=" * 78)
    print(f"  {'':<10}{'acc':>9}{'macro-F1':>11}{'N':>9}{'S':>9}{'VF':>9}")
    print(f"  {'float':<10}{f_acc:>9.4f}{f_f1:>11.4f}{f_pc[0]:>9.4f}{f_pc[1]:>9.4f}{f_pc[2]:>9.4f}")
    print(f"  {args.quant_type:<10}{q_acc:>9.4f}{q_f1:>11.4f}{q_pc[0]:>9.4f}{q_pc[1]:>9.4f}{q_pc[2]:>9.4f}")
    print(f"  {'delta':<10}{q_acc-f_acc:>+9.4f}{q_f1-f_f1:>+11.4f}"
          f"{q_pc[0]-f_pc[0]:>+9.4f}{q_pc[1]-f_pc[1]:>+9.4f}{q_pc[2]-f_pc[2]:>+9.4f}")
    print(f"  prediction agreement float vs quant : "
          f"{np.mean(float_preds == quant_preds) * 100:.2f}%")
    print(f"  confusion matrix (rows=true N,S,VF):\n{cm}")

    ok = (q_acc - f_acc) > -0.02 and (q_f1 - f_f1) > -0.02
    print(f"  acceptance (drop < 0.02 on both): {'PASS' if ok else 'FAIL'}")

    # Create a compact binary board-evaluation package. It contains the exact
    # balanced DS2 rows evaluated above and can be embedded without compiling a
    # multi-megabyte C++ initializer.
    board_eval_path = outdir / f"stage1_board_eval_{args.quant_type}_{len(eval_idx)}.bin"
    magic = b"ECGEV001"
    with board_eval_path.open("wb") as stream:
        stream.write(struct.pack("<8sIIIII", magic, 1, len(eval_idx), 198, 6, 3))
        for k, source_index in enumerate(eval_idx):
            stream.write(struct.pack("<ii", int(yt[k]), int(quant_preds[k])))
            stream.write(np.asarray(x_test[source_index], dtype="<f4").reshape(198).tobytes())
            stream.write(np.asarray(rr_test[source_index], dtype="<f4").reshape(6).tobytes())
            stream.write(np.asarray(quant_eval_logits[k], dtype="<f4").reshape(3).tobytes())
    board_eval_md5 = hashlib.md5(board_eval_path.read_bytes()).hexdigest()
    print(f"board evaluation package: {board_eval_path}")
    print(f"board evaluation MD5: {board_eval_md5}")

    report = {
        "onnx": str(onnx_in),
        "espdl": str(espdl_out),
        "target": args.target,
        "quant_type": args.quant_type,
        "calib_samples": int(len(calib_idx)),
        "eval_samples": int(len(eval_idx)),
        "eval_class_counts": np.bincount(yt, minlength=3).tolist(),
        "float": {"accuracy": float(f_acc), "macro_f1": float(f_f1), "per_class_f1": f_pc.tolist()},
        "quant": {"accuracy": float(q_acc), "macro_f1": float(q_f1), "per_class_f1": q_pc.tolist()},
        "prediction_agreement": float(np.mean(float_preds == quant_preds)),
        "golden_reference": str(golden_reference_path),
        "board_evaluation_package": str(board_eval_path),
        "board_evaluation_md5": board_eval_md5,
        "confusion_matrix": cm.tolist(),
        "accepted": bool(ok),
    }
    (outdir / f"stage1_{args.quant_type}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"report: {outdir / f'stage1_{args.quant_type}_report.json'}")
    onnx_work.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
