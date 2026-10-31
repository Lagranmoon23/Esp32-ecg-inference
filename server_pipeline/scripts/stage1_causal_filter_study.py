"""Quantify what breaks when the zero-phase preprocessing becomes causal.

Training used scipy.signal.sosfiltfilt (forward + backward), which needs future
samples and cannot run on a live ESP32. This script rebuilds the DS2 beats with
several causal alternatives and scores them with the UNCHANGED model, so the
damage is measured rather than assumed.

Variants (all use the annotation R peaks for RR features, so only the waveform
changes):

  zerophase        sosfiltfilt, slice at annotated R      <- training pipeline
  zerophase_align  sosfiltfilt, slice at local max        <- controls peak jitter
  causal_iir       4th-order Butterworth, forward only, slice at annotated R
  causal_iir_align forward only, slice at local max       <- what firmware does
  fir361_align     linear-phase FIR 0.8-40 Hz, 361 taps
  fir181_align     linear-phase FIR 1.2-40 Hz, 181 taps

Run with mcu_env (needs tensorflow + wfdb).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import wfdb
from scipy import signal

import stage1_build_deploy_onnx as B


DATA = ("/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/wym_mcu/"
        "MIT-BIH/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0")
SPLITS = "patient_independent_hierarchical_splits.npz"
WEIGHTS = "tiny_transformer_stage1_n_s_vf.weights.h5"
FS = 360
HALF = 99
SEARCH = 40  # +-40 samples (~111 ms) when locating the peak of the filtered beat

AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N", ".": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}


def sos_bandpass() -> np.ndarray:
    return signal.butter(4, [0.5, 40.0], btype="bandpass", fs=FS, output="sos")


def fir_bandpass(numtaps: int, low: float) -> np.ndarray:
    return signal.firwin(numtaps, [low, 40.0], pass_zero=False, fs=FS, window="hamming")


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    center = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - center))
    if scale < 1e-6:
        scale = np.std(x) + 1e-8
    return ((x - center) / scale).astype(np.float32)


def apply_filter(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zerophase":
        return signal.sosfiltfilt(sos_bandpass(), x)
    if mode == "causal_iir":
        sos = sos_bandpass()
        zi = signal.sosfilt_zi(sos) * x[0]  # start settled, no startup step
        y, _ = signal.sosfilt(sos, x, zi=zi)
        return y
    if mode == "fir361":
        return signal.lfilter(fir_bandpass(361, 0.8), [1.0], x)
    if mode == "fir181":
        return signal.lfilter(fir_bandpass(181, 1.2), [1.0], x)
    raise ValueError(mode)


def pick_lead(record) -> int:
    names = [str(n).upper().replace(" ", "") for n in record.sig_name]
    for preferred in ("MLII", "II"):
        if preferred in names:
            return names.index(preferred)
    return 0


def record_beats(record_id: str, mode: str, align: bool):
    """Return (beats, labels, peak_offsets) for one record under one variant."""
    path = str(Path(DATA) / record_id)
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    x = rec.p_signal[:, pick_lead(rec)]
    y = normalize(apply_filter(np.asarray(x, np.float64), mode))

    mapped = [(s, AAMI[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
    r_peaks = np.asarray([m[0] for m in mapped], dtype=np.int64)
    symbols = [m[1] for m in mapped]

    beats, labels, offs = [], [], []
    for i in range(1, len(r_peaks) - 1):
        if symbols[i] not in ("N", "S", "V", "F"):
            continue
        r = int(r_peaks[i])
        centre = r
        if align:
            lo = max(0, r - SEARCH)
            hi = min(len(y), r + SEARCH + 1)
            centre = lo + int(np.argmax(y[lo:hi]))
            offs.append(centre - r)
        start, end = centre - HALF, centre + HALF
        if start < 0 or end >= len(y):
            continue
        beats.append(y[start:end])
        labels.append(symbols[i])
    return np.asarray(beats, np.float32), labels, np.asarray(offs, np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="deployment_artifacts")
    ap.add_argument("--variants", default="zerophase,zerophase_align,causal_iir,causal_iir_align,fir361_align,fir181_align")
    args = ap.parse_args()
    outdir = Path(args.outdir).resolve()

    with np.load(SPLITS) as d:
        x_test_ref = d["X_test"].astype(np.float32)
        rr_test = d["RR_test"].astype(np.float32)
        y_raw = d["y_test"].astype(np.int64)
        test_records = [str(r) for r in d["test_records"]]
    y_test = np.where(y_raw >= 2, 2, y_raw).astype(np.int64)
    print(f"reference DS2: {x_test_ref.shape[0]} beats, records {len(test_records)}")

    model = B.build_reference_model()
    model.load_weights(WEIGHTS)

    label_map = {"N": 0, "S": 1, "V": 2, "F": 3}
    results = {}
    for variant in args.variants.split(","):
        mode, align = variant, variant.endswith("_align")
        if mode.endswith("_align"):
            mode = mode[: -len("_align")]
        store = "fir361" if mode == "fir361" else ("fir181" if mode == "fir181" else mode)

        all_beats, all_labels, all_offs = [], [], []
        for rid in test_records:
            beats, labels, offs = record_beats(rid, store, align)
            all_beats.append(beats)
            all_labels.extend(labels)
            all_offs.append(offs)
        x = np.concatenate(all_beats, 0)
        lab = np.asarray([label_map[s] for s in all_labels], dtype=np.int64)
        offs = np.concatenate(all_offs, 0) if align else np.zeros(0, dtype=np.int64)

        entry = {"beats": int(x.shape[0]), "label_match": bool(np.array_equal(lab, y_raw))}
        if x.shape[0] != x_test_ref.shape[0] or not entry["label_match"]:
            entry["status"] = "MISMATCH vs reference split"
            results[variant] = entry
            print(f"{variant:<20} MISMATCH: {x.shape[0]} beats, labels_equal={entry['label_match']}")
            continue

        if align and offs.size:
            entry["peak_offset_median"] = float(np.median(offs))
            entry["peak_offset_p95"] = float(np.percentile(np.abs(offs), 95))

        if variant == "zerophase":
            diff = float(np.max(np.abs(x - x_test_ref)))
            entry["max_abs_diff_vs_cache"] = diff
            entry["reproduces_cache"] = bool(diff < 1e-3)

        logits = model.predict({"x1": x.reshape(-1, 198, 1, 1), "x2": rr_test},
                               batch_size=512, verbose=0)
        preds = np.argmax(logits, 1)
        from sklearn.metrics import accuracy_score, f1_score
        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, labels=[0, 1, 2], average="macro"))
        pc = f1_score(y_test, preds, labels=[0, 1, 2], average=None).tolist()
        entry.update({"accuracy": acc, "macro_f1": f1, "per_class_f1": pc})

        if variant != "zerophase" and "zerophase" in results and "macro_f1" in results["zerophase"]:
            entry["macro_f1_delta_vs_zerophase"] = f1 - results["zerophase"]["macro_f1"]

        results[variant] = entry
        print(f"{variant:<20} acc={acc:.4f}  macro-F1={f1:.4f}  "
              f"N={pc[0]:.4f} S={pc[1]:.4f} VF={pc[2]:.4f}"
              + (f"  offset_med={entry.get('peak_offset_median', 0):+.0f}" if align else ""))

    (outdir / "stage1_causal_filter_study.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 92)
    print(f"{'variant':<20}{'acc':>8}{'macroF1':>9}{'N':>8}{'S':>8}{'VF':>8}"
          f"{'dF1':>9}{'peak_off':>10}")
    for k, v in results.items():
        if "macro_f1" not in v:
            continue
        print(f"{k:<20}{v['accuracy']:>8.4f}{v['macro_f1']:>9.4f}"
              f"{v['per_class_f1'][0]:>8.4f}{v['per_class_f1'][1]:>8.4f}{v['per_class_f1'][2]:>8.4f}"
              f"{v.get('macro_f1_delta_vs_zerophase', 0.0):>+9.4f}"
              f"{v.get('peak_offset_median', 0.0):>+10.1f}")
    print("=" * 92)
    print(f"report: {outdir / 'stage1_causal_filter_study.json'}")


if __name__ == "__main__":
    main()
