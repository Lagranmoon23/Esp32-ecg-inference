"""Can the device run the *exact* training preprocessing with a bounded lookahead?

sosfiltfilt needs future samples, but only a bounded number of them matter: the
0.5 Hz high-pass settles within a few hundred samples. So process the raw stream
in blocks, filter each block forward+backward together with a margin M of
neighbouring samples, and keep only the block centre. The processor then needs
M samples (M/360 seconds) of lookahead instead of the whole future.

If the stitched signal matches the record-level sosfiltfilt, the model keeps its
float baseline accuracy with NO retraining, at the cost of a fixed latency.

Variants:
  full            record-level sosfiltfilt + record median/MAD   (training pipeline)
  blk150/300/600  block filtfilt, margin M samples, record median/MAD
  blk300_runnorm  block filtfilt + running 10 s median/MAD (what firmware can do)

Run with mcu_env.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf

import stage1_build_deploy_onnx as B

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)


DATA = ("/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/"
        "MIT-BIH/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0")
SPLITS = "patient_independent_hierarchical_splits.npz"
WEIGHTS = "tiny_transformer_stage1_n_s_vf.weights.h5"
FS, HALF = 360, 99
CLASS = {"N": 0, "S": 1, "V": 2, "F": 3}
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", ".": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V", "F": "F",
        "/": "Q", "f": "Q", "Q": "Q", "?": "Q"}


def sos_bp():
    return signal.butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")


def block_zerophase(x: np.ndarray, sos: np.ndarray, block: int, margin: int) -> np.ndarray:
    """Forward+backward filtering per block with overlap, keeping block centres."""
    n = len(x)
    y = np.empty(n, dtype=np.float64)
    for start in range(0, n, block):
        end = min(start + block, n)
        lo, hi = max(0, start - margin), min(n, end + margin)
        seg = signal.sosfiltfilt(sos, x[lo:hi], padlen=min(27, (hi - lo) // 4))
        y[start:end] = seg[start - lo : end - lo]
    return y


def picks(record) -> int:
    names = [str(n).upper().replace(" ", "") for n in record.sig_name]
    for pref in ("MLII", "II"):
        if pref in names:
            return names.index(pref)
    return 0


def beats_from(y: np.ndarray, r_peaks, symbols, runnorm_seconds: float | None):
    beats, labels = [], []
    run = int(runnorm_seconds * FS) if runnorm_seconds else 0
    for i in range(1, len(r_peaks) - 1):
        if symbols[i] not in CLASS:
            continue
        r = int(r_peaks[i])
        s, e = r - HALF, r + HALF
        if s < 0 or e >= len(y):
            continue
        seg = y[s:e]
        if run:  # statistics over what is already available: 10 s before the window end
            lo = max(0, e - run)
            ref = y[lo:e]
            center = np.median(ref)
            scale = 1.4826 * np.median(np.abs(ref - center))
            if scale < 1e-6:
                scale = np.std(ref) + 1e-8
        else:
            center = np.median(y)
            scale = 1.4826 * np.median(np.abs(y - center))
            if scale < 1e-6:
                scale = np.std(y) + 1e-8
        beats.append(((seg - center) / scale).astype(np.float32))
        labels.append(symbols[i])
    return beats, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="deployment_artifacts")
    ap.add_argument("--block", type=int, default=396, help="samples per output block (1.1 s at 360 Hz)")
    ap.add_argument("--margins", default="150,300,600")
    ap.add_argument("--variants", default=None, help="comma separated subset, e.g. blk300")
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()

    ref = np.load(SPLITS)
    x_ref = ref["X_test"].astype(np.float32)
    rr_test, y_raw = ref["RR_test"], ref["y_test"]
    records = [str(r) for r in ref["test_records"]]
    y3 = np.where(y_raw >= 2, 2, y_raw)
    model = B.build_reference_model()
    model.load_weights(WEIGHTS)
    sos = sos_bp()

    if args.variants:
        variants = args.variants.split(",")
    else:
        variants = ["full"] + [f"blk{m}" for m in args.margins.split(",")] + [
            f"blk{args.margins.split(',')[1]}_runnorm"]
    results = {}

    # cache per record to avoid re-reading
    cache = {}
    for rid in records:
        path = str(Path(DATA) / rid)
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, "atr")
        mapped = [(s, AAMI[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
        cache[rid] = (np.asarray(rec.p_signal[:, picks(rec)], np.float64),
                      np.asarray([m[0] for m in mapped], np.int64),
                      [m[1] for m in mapped])

    for variant in variants:
        margin = 0
        runnorm = None
        if variant.startswith("blk"):
            body = variant[3:]
            if body.endswith("_runnorm"):
                runnorm = 10.0
                body = body[: -len("_runnorm")]
            margin = int(body)

        all_beats, all_labels, diffs = [], [], []
        for k, rid in enumerate(records):
            raw, rp, sym = cache[rid]
            if variant == "full":
                y = signal.sosfiltfilt(sos, raw)
            else:
                y = block_zerophase(raw, sos, args.block, margin)
                if margin:
                    diffs.append(float(np.max(np.abs(y - signal.sosfiltfilt(sos, raw)))))
            b, l = beats_from(y, rp, sym, runnorm)
            all_beats.extend(b)
            all_labels.extend(l)
            if (k + 1) % 8 == 0:
                print(f"    {variant}: {k+1}/{len(records)} records", flush=True)

        x = np.asarray(all_beats, np.float32)
        lab = np.asarray([CLASS[s] for s in all_labels], np.int64)
        entry = {"beats": int(x.shape[0])}
        if x.shape[0] != x_ref.shape[0] or not np.array_equal(lab, y_raw):
            entry["status"] = "MISMATCH"
            results[variant] = entry
            print(f"{variant:<16} MISMATCH {x.shape[0]}")
            continue

        if variant == "full":
            entry["max_abs_diff_vs_cache"] = float(np.max(np.abs(x - x_ref)))
        if margin:
            entry["latency_ms"] = margin / FS * 1000.0
            if diffs:
                entry["stitch_max_abs_diff_vs_full"] = max(diffs)

        logits = model.predict({"x1": x.reshape(-1, 198, 1, 1), "x2": rr_test},
                               batch_size=512, verbose=0)
        p = np.argmax(logits, 1)
        acc = float(accuracy_score(y3, p))
        f1m = float(f1_score(y3, p, labels=[0, 1, 2], average="macro"))
        pc = f1_score(y3, p, labels=[0, 1, 2], average=None).tolist()
        per_beat = np.abs(x - x_ref).reshape(x.shape[0], -1).max(axis=1)
        entry.update({"accuracy": acc, "macro_f1": f1m, "per_class_f1": pc,
                      "max_abs_diff_vs_cache": float(per_beat.max()),
                      "beat_diff_p50": float(np.percentile(per_beat, 50)),
                      "beat_diff_p99": float(np.percentile(per_beat, 99)),
                      "beats_diff_gt_0.5": int((per_beat > 0.5).sum())})
        results[variant] = entry
        print(f"{variant:<16} acc={acc:.4f} macro-F1={f1m:.4f} "
              f"N={pc[0]:.4f} S={pc[1]:.4f} VF={pc[2]:.4f} "
              f"|x-cache| p50={entry['beat_diff_p50']:.3f} p99={entry['beat_diff_p99']:.3f} "
              f"n>0.5={entry['beats_diff_gt_0.5']}"
              + (f" latency={entry['latency_ms']:.0f}ms" if margin else ""))

    (outdir / "stage1_windowed_zerophase_study.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    print(f"{'variant':<18}{'acc':>8}{'macroF1':>9}{'N':>8}{'S':>8}{'VF':>8}"
          f"{'dF1':>9}{'latency':>9}")
    base = results.get("full", {}).get("macro_f1", 0.0)
    for k, v in results.items():
        if "macro_f1" not in v:
            continue
        print(f"{k:<18}{v['accuracy']:>8.4f}{v['macro_f1']:>9.4f}"
              f"{v['per_class_f1'][0]:>8.4f}{v['per_class_f1'][1]:>8.4f}{v['per_class_f1'][2]:>8.4f}"
              f"{v['macro_f1']-base:>+9.4f}{v.get('latency_ms', 0):>8.0f}ms")
    print("=" * 88)
    print(f"report: {outdir / 'stage1_windowed_zerophase_study.json'}")


if __name__ == "__main__":
    main()
