"""Evaluate a deployment-realistic streaming ECG pipeline on MIT-BIH DS2.

The script does not train a model. It separates four sources of deployment
shift while keeping the patient-independent DS2 test protocol unchanged:

  A  cached zero-phase preprocessing + annotated R peaks
  B  causal IIR preprocessing + annotated R peaks + record-wide normalization
  C  causal IIR preprocessing + annotated R peaks + per-beat normalization
  D  causal IIR preprocessing + independent R detector + per-beat normalization

D reports both classification on matched beats and an end-to-end score in
which missed annotated beats count as wrong. False detections are reported by
the detector PPV/F1 and are not silently discarded.

The detector is deliberately implemented from basic causal operations so it
can later be ported to ESP32. Its adaptive threshold is initialized from a
short startup period; no annotation is used to locate its peaks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import wfdb
from scipy import signal
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from stage1_deployment_audit import build_stage1_model


FS = 360
HALF_WINDOW = 99
CLASS_NAMES = ["N", "S", "VF"]
OUTPUT_CLASSES = {"N", "S", "V", "F"}

AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N", ".": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V", "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "?": "Q",
}

DEFAULT_DATA = (
    "/media/ecglab506/c91df7e1-f50c-468a-9741-41700680ea81/"
    "wym_mcu/MIT-BIH/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0"
)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def class_id(symbol: str) -> int:
    if symbol == "N":
        return 0
    if symbol == "S":
        return 1
    if symbol in {"V", "F"}:
        return 2
    raise ValueError(f"Unsupported output class: {symbol}")


def choose_lead(record: wfdb.Record) -> int:
    names = [str(name).upper().replace(" ", "") for name in record.sig_name]
    for preferred in ("MLII", "II"):
        if preferred in names:
            return names.index(preferred)
    return 0


def build_filter_chain(fs: int) -> tuple[np.ndarray, np.ndarray]:
    bandpass = signal.butter(
        4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos"
    )
    notch_b, notch_a = signal.iirnotch(50.0, 30.0, fs=fs)
    notch = signal.tf2sos(notch_b, notch_a)
    return bandpass.astype(np.float64), notch.astype(np.float64)


def causal_filter_continuous(
    samples: np.ndarray,
    bandpass: np.ndarray,
    notch: np.ndarray,
) -> np.ndarray:
    """Filter a whole record once while preserving continuous filter state."""
    x = np.nan_to_num(np.asarray(samples, dtype=np.float64))
    bp_state = signal.sosfilt_zi(bandpass) * x[0]
    y, _ = signal.sosfilt(bandpass, x, zi=bp_state)
    notch_state = signal.sosfilt_zi(notch) * y[0]
    y, _ = signal.sosfilt(notch, y, zi=notch_state)
    return y.astype(np.float32)


def robust_parameters(samples: np.ndarray) -> tuple[float, float]:
    center = float(np.median(samples))
    scale = float(1.4826 * np.median(np.abs(samples - center)))
    if scale < 1e-6:
        scale = float(np.std(samples) + 1e-8)
    return center, scale


def normalize_record(samples: np.ndarray) -> np.ndarray:
    center, scale = robust_parameters(samples)
    return ((samples - center) / scale).astype(np.float32)


def normalize_beat(beat: np.ndarray, clip: float) -> np.ndarray:
    center, scale = robust_parameters(beat)
    normalized = (beat - center) / scale
    return np.clip(normalized, -clip, clip).astype(np.float32)


def normalize_rolling_robust(
    samples: np.ndarray,
    fs: int,
    window_seconds: float,
    update_seconds: float,
    warmup_seconds: float,
    clip: float,
) -> np.ndarray:
    """Causal rolling median/MAD normalization for a continuous stream.

    Statistics used for each output block are calculated exclusively from
    samples that arrived before that block. Values before warm-up are NaN and
    are excluded from rolling-normalized classification experiments.
    """
    x = np.asarray(samples, dtype=np.float32)
    output = np.full(x.shape, np.nan, dtype=np.float32)
    window = max(1, int(round(window_seconds * fs)))
    update = max(1, int(round(update_seconds * fs)))
    warmup = max(window, int(round(warmup_seconds * fs)))
    for block_start in range(warmup, len(x), update):
        history_start = max(0, block_start - window)
        history = x[history_start:block_start]
        center, scale = robust_parameters(history)
        block_end = min(len(x), block_start + update)
        normalized = (x[block_start:block_end] - center) / scale
        output[block_start:block_end] = np.clip(normalized, -clip, clip)
    return output


def moving_average_causal(values: np.ndarray, width: int) -> np.ndarray:
    return signal.lfilter(
        np.ones(width, dtype=np.float64) / float(width),
        [1.0],
        values.astype(np.float64),
    )


def detect_r_peaks_streaming(
    filtered: np.ndarray,
    fs: int,
    startup_seconds: float,
    threshold_weight: float,
    refractory_seconds: float,
) -> np.ndarray:
    """Pan-Tompkins-inspired detector without annotation-guided peak search.

    The detector uses a derivative-energy envelope, a causal moving integrator,
    adaptive signal/noise levels, and a refractory period. A short look-ahead
    is used only to refine the R location; this delay is acceptable because the
    classifier already waits for post-RR.
    """
    x = np.asarray(filtered, dtype=np.float64)
    derivative = signal.lfilter(
        np.asarray([1.0, 2.0, 0.0, -2.0, -1.0]) * (fs / 8.0),
        [1.0],
        x,
    )
    integration_width = max(1, int(round(0.150 * fs)))
    envelope = moving_average_causal(derivative * derivative, integration_width)

    # Candidate extraction itself only needs a one-sample look-ahead and a
    # refractory-like distance; threshold decisions below remain sequential.
    candidates, _ = signal.find_peaks(
        envelope,
        distance=max(1, int(round(0.080 * fs))),
    )
    startup = max(int(round(startup_seconds * fs)), integration_width)
    initial_values = envelope[candidates[candidates < startup]]
    if initial_values.size < 4:
        initial_values = envelope[:startup]
    noise_level = float(np.percentile(initial_values, 50))
    signal_level = float(np.percentile(initial_values, 95))
    if signal_level <= noise_level:
        signal_level = float(np.max(initial_values) + 1e-12)

    refractory = max(1, int(round(refractory_seconds * fs)))
    refine_before = int(round(0.200 * fs))
    refine_after = int(round(0.080 * fs))
    accepted: list[int] = []
    accepted_strength: list[float] = []

    for candidate in candidates:
        strength = float(envelope[candidate])
        threshold = noise_level + threshold_weight * (signal_level - noise_level)
        if candidate < startup:
            # Startup is used only to initialize the adaptive levels.
            continue

        if strength >= threshold:
            lo = max(0, int(candidate) - refine_before)
            hi = min(len(x), int(candidate) + refine_after + 1)
            # abs() also supports records whose dominant R deflection is negative.
            r_peak = lo + int(np.argmax(np.abs(x[lo:hi])))

            if not accepted or r_peak - accepted[-1] >= refractory:
                accepted.append(r_peak)
                accepted_strength.append(strength)
            elif strength > accepted_strength[-1]:
                # Two envelope maxima may represent one QRS. Keep the stronger.
                accepted[-1] = r_peak
                accepted_strength[-1] = strength
            signal_level = 0.125 * strength + 0.875 * signal_level
        else:
            noise_level = 0.125 * strength + 0.875 * noise_level

    if not accepted:
        return np.empty(0, dtype=np.int64)
    peaks = np.asarray(accepted, dtype=np.int64)
    # Refinement windows can map adjacent candidates to the same location.
    return np.unique(peaks)


def detect_r_peaks_streaming_v2(
    filtered: np.ndarray,
    fs: int,
    startup_seconds: float,
    threshold_weight: float,
    secondary_ratio: float,
    refractory_seconds: float,
    t_wave_end_seconds: float,
    t_wave_slope_ratio: float,
    searchback_rr_factor: float,
) -> np.ndarray:
    """Adaptive QRS detector with search-back and T-wave rejection.

    A separate causal 5-20 Hz channel is used only for detection. The wider
    0.5-40 Hz signal remains unchanged for model input and final R refinement.
    """
    model_signal = np.asarray(filtered, dtype=np.float64)
    detector_sos = signal.butter(
        2, [5.0, 20.0], btype="bandpass", fs=fs, output="sos"
    )
    detector_state = signal.sosfilt_zi(detector_sos) * model_signal[0]
    qrs_band, _ = signal.sosfilt(detector_sos, model_signal, zi=detector_state)

    derivative = signal.lfilter(
        np.asarray([1.0, 2.0, 0.0, -2.0, -1.0]) * (fs / 8.0),
        [1.0],
        qrs_band,
    )
    integration_width = max(1, int(round(0.150 * fs)))
    envelope = moving_average_causal(derivative * derivative, integration_width)
    candidates, _ = signal.find_peaks(
        envelope,
        distance=max(1, int(round(0.080 * fs))),
    )

    startup = max(int(round(startup_seconds * fs)), integration_width)
    initial_values = envelope[candidates[candidates < startup]]
    if initial_values.size < 4:
        initial_values = envelope[:startup]
    noise_level = float(np.percentile(initial_values, 50))
    signal_level = float(np.percentile(initial_values, 95))
    if signal_level <= noise_level:
        signal_level = float(np.max(initial_values) + 1e-12)

    refractory = max(1, int(round(refractory_seconds * fs)))
    t_wave_end = max(refractory + 1, int(round(t_wave_end_seconds * fs)))
    # The energy envelope is delayed. A bounded backward refinement avoids
    # selecting the following T wave while tolerating variable QRS width.
    refine_before = int(round(0.200 * fs))
    refine_after = int(round(0.020 * fs))
    slope_radius = max(2, int(round(0.040 * fs)))

    accepted: list[int] = []
    accepted_strength: list[float] = []
    accepted_slope: list[float] = []
    rr_history: list[int] = []
    # Entries are (r_peak, envelope strength, local slope, secondary threshold).
    searchback_pool: list[tuple[int, float, float, float]] = []

    def refine(candidate: int) -> tuple[int, float]:
        lo = max(0, int(candidate) - refine_before)
        hi = min(len(model_signal), int(candidate) + refine_after + 1)
        r_peak = lo + int(np.argmax(np.abs(model_signal[lo:hi])))
        slope_lo = max(1, r_peak - slope_radius)
        slope_hi = min(len(qrs_band), r_peak + slope_radius + 1)
        local_slope = float(np.max(np.abs(np.diff(qrs_band[slope_lo:slope_hi]))))
        return r_peak, local_slope

    def is_t_wave(r_peak: int, local_slope: float) -> bool:
        if not accepted:
            return False
        interval = r_peak - accepted[-1]
        return (
            refractory <= interval <= t_wave_end
            and accepted_slope[-1] > 0.0
            and local_slope < t_wave_slope_ratio * accepted_slope[-1]
        )

    def accept(r_peak: int, strength: float, local_slope: float) -> None:
        if accepted:
            rr_history.append(r_peak - accepted[-1])
            if len(rr_history) > 8:
                del rr_history[0]
        accepted.append(r_peak)
        accepted_strength.append(strength)
        accepted_slope.append(local_slope)

    for candidate in candidates:
        strength = float(envelope[candidate])
        primary = noise_level + threshold_weight * (signal_level - noise_level)
        secondary = secondary_ratio * primary
        r_peak, local_slope = refine(int(candidate))
        if candidate < startup:
            continue

        # If the current candidate exposes a long gap, recover the strongest
        # secondary-threshold candidate before evaluating the current one.
        if accepted and rr_history:
            expected_rr = float(np.median(rr_history))
            if r_peak - accepted[-1] > searchback_rr_factor * expected_rr:
                eligible = [
                    item for item in searchback_pool
                    if item[0] - accepted[-1] >= refractory
                    and r_peak - item[0] >= refractory
                    and item[1] >= item[3]
                ]
                if eligible:
                    recovered = max(eligible, key=lambda item: item[1])
                    if not is_t_wave(recovered[0], recovered[2]):
                        accept(recovered[0], recovered[1], recovered[2])
                        signal_level = 0.25 * recovered[1] + 0.75 * signal_level
                searchback_pool = [item for item in searchback_pool if item[0] > accepted[-1]]

        interval_ok = not accepted or r_peak - accepted[-1] >= refractory
        t_wave = is_t_wave(r_peak, local_slope)
        if strength >= primary and interval_ok and not t_wave:
            accept(r_peak, strength, local_slope)
            signal_level = 0.125 * strength + 0.875 * signal_level
            searchback_pool = [item for item in searchback_pool if item[0] > r_peak]
        else:
            noise_level = 0.125 * strength + 0.875 * noise_level
            if interval_ok and not t_wave:
                searchback_pool.append((r_peak, strength, local_slope, secondary))
                if len(searchback_pool) > 64:
                    del searchback_pool[:-64]

    if not accepted:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.asarray(accepted, dtype=np.int64))


def match_peaks(
    reference: np.ndarray,
    detected: np.ndarray,
    tolerance: int,
) -> tuple[dict[int, int], np.ndarray]:
    """Return annotation-index -> detected-index matches and signed errors."""
    possible: list[tuple[int, int, int]] = []
    for detected_index, peak in enumerate(detected):
        pos = int(np.searchsorted(reference, peak))
        for reference_index in (pos - 1, pos):
            if 0 <= reference_index < len(reference):
                distance = abs(int(peak) - int(reference[reference_index]))
                if distance <= tolerance:
                    possible.append((distance, reference_index, detected_index))

    matches: dict[int, int] = {}
    used_detected: set[int] = set()
    for _, reference_index, detected_index in sorted(possible):
        if reference_index in matches or detected_index in used_detected:
            continue
        matches[reference_index] = detected_index
        used_detected.add(detected_index)

    errors = np.asarray(
        [int(detected[d]) - int(reference[r]) for r, d in matches.items()],
        dtype=np.int64,
    )
    return matches, errors


def rr_features(peaks: np.ndarray, index: int, fs: int) -> np.ndarray:
    intervals = np.diff(peaks).astype(np.float64) / float(fs)
    pre_rr = float(intervals[index - 1])
    post_rr = float(intervals[index])
    local_history = intervals[max(0, index - 10):index]
    global_history = intervals[max(0, index - 60):index]
    local_rr = float(np.median(local_history)) if local_history.size else pre_rr
    global_rr = float(np.median(global_history)) if global_history.size else local_rr
    local_rr = max(local_rr, 1e-3)
    return np.asarray(
        [
            np.clip(pre_rr, 0.2, 2.0),
            np.clip(post_rr, 0.2, 2.0),
            np.clip(local_rr, 0.2, 2.0),
            np.clip(global_rr, 0.2, 2.0),
            np.clip(pre_rr / local_rr, 0.3, 2.0),
            np.clip(post_rr / local_rr, 0.3, 2.0),
        ],
        dtype=np.float32,
    )


def annotation_rows(
    filtered: np.ndarray,
    annotation_peaks: np.ndarray,
    annotation_symbols: list[str],
    fs: int,
    beat_clip: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """Create B/C inputs at annotated peaks using identical RR context."""
    record_normalized = normalize_record(filtered)
    beats_record: list[np.ndarray] = []
    beats_per_beat: list[np.ndarray] = []
    rr_rows: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(1, len(annotation_peaks) - 1):
        symbol = annotation_symbols[i]
        if symbol not in OUTPUT_CLASSES:
            continue
        start = int(annotation_peaks[i]) - HALF_WINDOW
        end = int(annotation_peaks[i]) + HALF_WINDOW
        if start < 0 or end >= len(filtered):
            continue
        raw_beat = filtered[start:end]
        beats_record.append(record_normalized[start:end].astype(np.float32))
        beats_per_beat.append(normalize_beat(raw_beat, beat_clip))
        rr_rows.append(rr_features(annotation_peaks, i, fs))
        labels.append(class_id(symbol))
    return beats_record, beats_per_beat, rr_rows, labels


def normalized_annotation_rows(
    normalized_stream: np.ndarray,
    annotation_peaks: np.ndarray,
    annotation_symbols: list[str],
    fs: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Extract annotated beats from an already normalized causal stream."""
    beats: list[np.ndarray] = []
    rr_rows: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(1, len(annotation_peaks) - 1):
        symbol = annotation_symbols[i]
        if symbol not in OUTPUT_CLASSES:
            continue
        start = int(annotation_peaks[i]) - HALF_WINDOW
        end = int(annotation_peaks[i]) + HALF_WINDOW
        if start < 0 or end >= len(normalized_stream):
            continue
        beat = normalized_stream[start:end]
        if not np.all(np.isfinite(beat)):
            continue
        beats.append(beat.astype(np.float32))
        rr_rows.append(rr_features(annotation_peaks, i, fs))
        labels.append(class_id(symbol))
    return beats, rr_rows, labels


def detected_rows(
    filtered: np.ndarray,
    detected_peaks: np.ndarray,
    annotation_peaks: np.ndarray,
    annotation_symbols: list[str],
    matches: dict[int, int],
    fs: int,
    beat_clip: float,
    minimum_sample: int,
    normalized_stream: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int], list[int]]:
    """Create D inputs and map each prediction back to an eligible annotation."""
    eligible_annotation_indices: list[int] = []
    eligible_labels: list[int] = []
    annotation_to_output: dict[int, int] = {}
    for annotation_index in range(1, len(annotation_peaks) - 1):
        symbol = annotation_symbols[annotation_index]
        if symbol not in OUTPUT_CLASSES:
            continue
        if int(annotation_peaks[annotation_index]) < minimum_sample:
            continue
        start = int(annotation_peaks[annotation_index]) - HALF_WINDOW
        end = int(annotation_peaks[annotation_index]) + HALF_WINDOW
        if start < 0 or end >= len(filtered):
            continue
        annotation_to_output[annotation_index] = len(eligible_labels)
        eligible_annotation_indices.append(annotation_index)
        eligible_labels.append(class_id(symbol))

    beats: list[np.ndarray] = []
    rr_rows: list[np.ndarray] = []
    matched_labels: list[int] = []
    output_positions: list[int] = []
    for annotation_index, detected_index in sorted(matches.items()):
        if annotation_index not in annotation_to_output:
            continue
        # post-RR requires a following detected beat.
        if detected_index < 1 or detected_index >= len(detected_peaks) - 1:
            continue
        center = int(detected_peaks[detected_index])
        start, end = center - HALF_WINDOW, center + HALF_WINDOW
        if start < 0 or end >= len(filtered):
            continue
        if normalized_stream is None:
            beat = normalize_beat(filtered[start:end], beat_clip)
        else:
            beat = normalized_stream[start:end]
            if not np.all(np.isfinite(beat)):
                continue
            beat = beat.astype(np.float32)
        beats.append(beat)
        rr_rows.append(rr_features(detected_peaks, detected_index, fs))
        matched_labels.append(class_id(annotation_symbols[annotation_index]))
        output_positions.append(annotation_to_output[annotation_index])

    return beats, rr_rows, matched_labels, output_positions, eligible_labels


def predict(
    model: tf.keras.Model,
    beats: np.ndarray,
    rr: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.asarray(beats, dtype=np.float32).reshape(-1, 198, 1, 1)
    x2 = np.asarray(rr, dtype=np.float32).reshape(-1, 6)
    logits = model.predict(
        {"x1": x1, "x2": x2}, batch_size=batch_size, verbose=0
    )
    return logits.astype(np.float32), np.argmax(logits, axis=1).astype(np.int64)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    return {
        "samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(labels, predictions, labels=[0, 1, 2], average="weighted", zero_division=0)
        ),
        "per_class_f1": f1_score(
            labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
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


def end_to_end_metrics(
    all_labels: np.ndarray,
    matched_predictions: np.ndarray,
    output_positions: np.ndarray,
) -> dict[str, Any]:
    predictions = np.full(len(all_labels), -1, dtype=np.int64)
    predictions[np.asarray(output_positions, dtype=np.int64)] = matched_predictions
    metrics = classification_metrics(all_labels, predictions)
    metrics["coverage"] = float(len(matched_predictions) / max(1, len(all_labels)))
    metrics["missed_output_beats"] = int(np.sum(predictions < 0))
    metrics["confusion_matrix_with_missed_column"] = confusion_matrix(
        all_labels, predictions, labels=[0, 1, 2, -1]
    ).tolist()
    return metrics


def load_model(weights: Path) -> tf.keras.Model:
    model = build_stage1_model()
    model.load_weights(weights)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--splits", default="patient_independent_hierarchical_splits.npz")
    parser.add_argument(
        "--record-split",
        choices=["train", "validation", "test"],
        default="test",
        help="Tune detector settings only on train/validation; reserve test for final reporting.",
    )
    parser.add_argument("--weights", default="tiny_transformer_stage1_n_s_vf.weights.h5")
    parser.add_argument(
        "--streaming-weights",
        default="tiny_transformer_stage1_n_s_vf_causal_ft.weights.h5",
        help="Optional diagnostic weights; use 'none' to skip.",
    )
    parser.add_argument("--output", default="deployment_artifacts/stage1_streaming_pipeline_study.json")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--startup-seconds", type=float, default=2.0)
    parser.add_argument("--detector-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--match-ms", type=float, default=50.0)
    parser.add_argument("--threshold-weight", type=float, default=0.25)
    parser.add_argument("--secondary-threshold-ratio", type=float, default=0.50)
    parser.add_argument("--refractory-ms", type=float, default=250.0)
    parser.add_argument("--t-wave-end-ms", type=float, default=360.0)
    parser.add_argument("--t-wave-slope-ratio", type=float, default=0.50)
    parser.add_argument("--searchback-rr-factor", type=float, default=1.66)
    parser.add_argument("--beat-clip", type=float, default=5.0)
    parser.add_argument("--rolling-window-seconds", type=float, default=10.0)
    parser.add_argument("--rolling-update-seconds", type=float, default=1.0)
    parser.add_argument("--rolling-warmup-seconds", type=float, default=10.0)
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Debug only; 0 processes the complete DS2.",
    )
    args = parser.parse_args()

    split_path = Path(args.splits).resolve()
    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(split_path) as cached:
        split_suffix = {"train": "train", "validation": "val", "test": "test"}[
            args.record_split
        ]
        cached_x = cached[f"X_{split_suffix}"].astype(np.float32)
        cached_rr = cached[f"RR_{split_suffix}"].astype(np.float32)
        cached_y_raw = cached[f"y_{split_suffix}"].astype(np.int64)
        rr_mean = cached["rr_mean"].astype(np.float32).reshape(1, 6)
        rr_std = cached["rr_std"].astype(np.float32).reshape(1, 6)
        record_key = {
            "train": "train_records",
            "validation": "validation_records",
            "test": "test_records",
        }[args.record_split]
        all_test_records = [str(item) for item in cached[record_key]]

    test_records = list(all_test_records)
    if args.max_records:
        test_records = test_records[: args.max_records]
    complete_ds2 = len(test_records) == len(all_test_records)

    model = load_model(weights_path)
    results: dict[str, Any] = {
        "configuration": {
            "data": str(data_path),
            "splits": str(split_path),
            "weights": str(weights_path),
            "record_split": args.record_split,
            "records": test_records,
            "complete_ds2": complete_ds2,
            "sampling_hz": FS,
            "filter": "causal Butterworth order=4 bandpass 0.5-40 Hz + notch 50 Hz Q=30",
            "per_beat_normalization": f"median/MAD, clip +/-{args.beat_clip}",
            "rolling_normalization": {
                "window_seconds": args.rolling_window_seconds,
                "update_seconds": args.rolling_update_seconds,
                "warmup_seconds": args.rolling_warmup_seconds,
                "clip": args.beat_clip,
                "causal": True,
            },
            "detector": {
                "version": args.detector_version,
                "startup_seconds": args.startup_seconds,
                "threshold_weight": args.threshold_weight,
                "secondary_threshold_ratio": args.secondary_threshold_ratio,
                "refractory_ms": args.refractory_ms,
                "t_wave_end_ms": args.t_wave_end_ms,
                "t_wave_slope_ratio": args.t_wave_slope_ratio,
                "searchback_rr_factor": args.searchback_rr_factor,
                "match_ms": args.match_ms,
            },
        }
    }

    # A is evaluated directly from the known cache only for a complete DS2 run.
    if complete_ds2:
        cached_y = np.where(cached_y_raw >= 2, 2, cached_y_raw).astype(np.int64)
        _, predictions_a = predict(model, cached_x, cached_rr, args.batch_size)
        results["A_zero_phase_cached_annotation"] = classification_metrics(
            cached_y, predictions_a
        )

    bandpass, notch = build_filter_chain(FS)
    b_beats: list[np.ndarray] = []
    c_beats: list[np.ndarray] = []
    bc_rr_raw: list[np.ndarray] = []
    bc_labels: list[int] = []
    rolling_annotation_beats: list[np.ndarray] = []
    rolling_annotation_rr_raw: list[np.ndarray] = []
    rolling_annotation_labels: list[int] = []
    d_beats: list[np.ndarray] = []
    d_rr_raw: list[np.ndarray] = []
    d_labels_matched: list[int] = []
    d_labels_all: list[int] = []
    d_output_positions_global: list[int] = []
    rolling_detected_beats: list[np.ndarray] = []
    rolling_detected_rr_raw: list[np.ndarray] = []
    rolling_detected_labels_matched: list[int] = []
    rolling_detected_labels_all: list[int] = []
    rolling_detected_positions_global: list[int] = []
    record_detection: dict[str, Any] = {}
    total_reference = total_detected = total_matched = 0
    all_timing_errors: list[np.ndarray] = []

    for record_id in test_records:
        record = wfdb.rdrecord(str(data_path / record_id))
        annotation = wfdb.rdann(str(data_path / record_id), "atr")
        if int(round(float(record.fs))) != FS:
            raise ValueError(f"Record {record_id} has fs={record.fs}; expected {FS}")
        channel = choose_lead(record)
        raw = record.p_signal[:, channel]
        filtered = causal_filter_continuous(raw, bandpass, notch)
        rolling_normalized = normalize_rolling_robust(
            filtered,
            FS,
            args.rolling_window_seconds,
            args.rolling_update_seconds,
            args.rolling_warmup_seconds,
            args.beat_clip,
        )

        mapped = [
            (int(sample), AAMI[symbol])
            for sample, symbol in zip(annotation.sample, annotation.symbol)
            if symbol in AAMI
        ]
        annotation_peaks = np.asarray([item[0] for item in mapped], dtype=np.int64)
        annotation_symbols = [item[1] for item in mapped]

        record_b, record_c, record_rr, record_labels = annotation_rows(
            filtered,
            annotation_peaks,
            annotation_symbols,
            FS,
            args.beat_clip,
        )
        b_beats.extend(record_b)
        c_beats.extend(record_c)
        bc_rr_raw.extend(record_rr)
        bc_labels.extend(record_labels)

        record_rolling, record_rolling_rr, record_rolling_labels = normalized_annotation_rows(
            rolling_normalized,
            annotation_peaks,
            annotation_symbols,
            FS,
        )
        rolling_annotation_beats.extend(record_rolling)
        rolling_annotation_rr_raw.extend(record_rolling_rr)
        rolling_annotation_labels.extend(record_rolling_labels)

        if args.detector_version == "v1":
            detected = detect_r_peaks_streaming(
                filtered,
                FS,
                args.startup_seconds,
                args.threshold_weight,
                args.refractory_ms / 1000.0,
            )
        else:
            detected = detect_r_peaks_streaming_v2(
                filtered,
                FS,
                args.startup_seconds,
                args.threshold_weight,
                args.secondary_threshold_ratio,
                args.refractory_ms / 1000.0,
                args.t_wave_end_ms / 1000.0,
                args.t_wave_slope_ratio,
                args.searchback_rr_factor,
            )
        startup_sample = int(round(args.startup_seconds * FS))
        reference_eval_mask = annotation_peaks >= startup_sample
        detected_eval_mask = detected >= startup_sample
        reference_eval = annotation_peaks[reference_eval_mask]
        detected_eval = detected[detected_eval_mask]
        # Convert matches back to full annotation/detection indices.
        matches_eval, errors = match_peaks(
            reference_eval,
            detected_eval,
            int(round(args.match_ms * FS / 1000.0)),
        )
        reference_indices = np.flatnonzero(reference_eval_mask)
        detected_indices = np.flatnonzero(detected_eval_mask)
        matches_full = {
            int(reference_indices[r]): int(detected_indices[d])
            for r, d in matches_eval.items()
        }

        offset_base = len(d_labels_all)
        record_d, record_d_rr, record_d_labels, record_positions, record_all_labels = detected_rows(
            filtered,
            detected,
            annotation_peaks,
            annotation_symbols,
            matches_full,
            FS,
            args.beat_clip,
            startup_sample,
        )
        d_beats.extend(record_d)
        d_rr_raw.extend(record_d_rr)
        d_labels_matched.extend(record_d_labels)
        d_output_positions_global.extend([offset_base + p for p in record_positions])
        d_labels_all.extend(record_all_labels)

        rolling_minimum_sample = max(
            startup_sample,
            int(round(args.rolling_warmup_seconds * FS)) + HALF_WINDOW,
        )
        rolling_offset_base = len(rolling_detected_labels_all)
        (
            record_dr,
            record_dr_rr,
            record_dr_labels,
            record_dr_positions,
            record_dr_all_labels,
        ) = detected_rows(
            filtered,
            detected,
            annotation_peaks,
            annotation_symbols,
            matches_full,
            FS,
            args.beat_clip,
            rolling_minimum_sample,
            normalized_stream=rolling_normalized,
        )
        rolling_detected_beats.extend(record_dr)
        rolling_detected_rr_raw.extend(record_dr_rr)
        rolling_detected_labels_matched.extend(record_dr_labels)
        rolling_detected_positions_global.extend(
            [rolling_offset_base + p for p in record_dr_positions]
        )
        rolling_detected_labels_all.extend(record_dr_all_labels)

        tp = len(matches_eval)
        fp = len(detected_eval) - tp
        fn = len(reference_eval) - tp
        total_reference += len(reference_eval)
        total_detected += len(detected_eval)
        total_matched += tp
        if errors.size:
            all_timing_errors.append(errors)
        record_detection[record_id] = {
            "lead": str(record.sig_name[channel]),
            "reference": int(len(reference_eval)),
            "detected": int(len(detected_eval)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "sensitivity": float(tp / max(1, tp + fn)),
            "ppv": float(tp / max(1, tp + fp)),
            "timing_abs_median_samples": float(np.median(np.abs(errors))) if errors.size else None,
            "timing_abs_p95_samples": float(np.percentile(np.abs(errors), 95)) if errors.size else None,
        }
        print(
            f"{record_id}: ref={len(reference_eval)} det={len(detected_eval)} "
            f"TP={tp} FP={fp} FN={fn}",
            flush=True,
        )

    labels_bc = np.asarray(bc_labels, dtype=np.int64)
    rr_bc = (np.asarray(bc_rr_raw, dtype=np.float32) - rr_mean) / rr_std
    _, pred_b = predict(model, np.asarray(b_beats), rr_bc, args.batch_size)
    _, pred_c = predict(model, np.asarray(c_beats), rr_bc, args.batch_size)
    results["B_causal_annotation_record_norm"] = classification_metrics(labels_bc, pred_b)
    results["C_causal_annotation_per_beat_norm"] = classification_metrics(labels_bc, pred_c)

    labels_rolling_annotation = np.asarray(rolling_annotation_labels, dtype=np.int64)
    rr_rolling_annotation = (
        np.asarray(rolling_annotation_rr_raw, dtype=np.float32) - rr_mean
    ) / rr_std
    _, pred_rolling_annotation = predict(
        model,
        np.asarray(rolling_annotation_beats),
        rr_rolling_annotation,
        args.batch_size,
    )
    results["C2_causal_annotation_rolling_norm"] = classification_metrics(
        labels_rolling_annotation, pred_rolling_annotation
    )

    labels_d_matched = np.asarray(d_labels_matched, dtype=np.int64)
    labels_d_all = np.asarray(d_labels_all, dtype=np.int64)
    rr_d = (np.asarray(d_rr_raw, dtype=np.float32) - rr_mean) / rr_std
    _, pred_d = predict(model, np.asarray(d_beats), rr_d, args.batch_size)
    results["D_streaming_matched_beats_original_model"] = classification_metrics(
        labels_d_matched, pred_d
    )
    results["D_streaming_end_to_end_original_model"] = end_to_end_metrics(
        labels_d_all,
        pred_d,
        np.asarray(d_output_positions_global, dtype=np.int64),
    )

    labels_dr_matched = np.asarray(rolling_detected_labels_matched, dtype=np.int64)
    labels_dr_all = np.asarray(rolling_detected_labels_all, dtype=np.int64)
    rr_dr = (np.asarray(rolling_detected_rr_raw, dtype=np.float32) - rr_mean) / rr_std
    _, pred_dr = predict(
        model, np.asarray(rolling_detected_beats), rr_dr, args.batch_size
    )
    results["D2_streaming_rolling_norm_matched_original_model"] = classification_metrics(
        labels_dr_matched, pred_dr
    )
    results["D2_streaming_rolling_norm_end_to_end_original_model"] = end_to_end_metrics(
        labels_dr_all,
        pred_dr,
        np.asarray(rolling_detected_positions_global, dtype=np.int64),
    )

    timing_errors = (
        np.concatenate(all_timing_errors)
        if all_timing_errors
        else np.empty(0, dtype=np.int64)
    )
    sensitivity = total_matched / max(1, total_reference)
    ppv = total_matched / max(1, total_detected)
    results["r_peak_detection"] = {
        "reference": int(total_reference),
        "detected": int(total_detected),
        "tp": int(total_matched),
        "fp": int(total_detected - total_matched),
        "fn": int(total_reference - total_matched),
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),
        "f1": float(2 * sensitivity * ppv / max(1e-12, sensitivity + ppv)),
        "timing_signed_mean_samples": float(np.mean(timing_errors)) if timing_errors.size else None,
        "timing_abs_median_samples": float(np.median(np.abs(timing_errors))) if timing_errors.size else None,
        "timing_abs_p95_samples": float(np.percentile(np.abs(timing_errors), 95)) if timing_errors.size else None,
        "per_record": record_detection,
    }

    streaming_weights = str(args.streaming_weights).strip()
    if streaming_weights.lower() not in {"", "none", "null"}:
        streaming_path = Path(streaming_weights).resolve()
        if streaming_path.is_file():
            streaming_model = load_model(streaming_path)
            _, streaming_pred = predict(
                streaming_model, np.asarray(d_beats), rr_d, args.batch_size
            )
            results["D_streaming_matched_beats_causal_ft_model"] = classification_metrics(
                labels_d_matched, streaming_pred
            )
            results["D_streaming_end_to_end_causal_ft_model"] = end_to_end_metrics(
                labels_d_all,
                streaming_pred,
                np.asarray(d_output_positions_global, dtype=np.int64),
            )
            _, streaming_pred_rolling = predict(
                streaming_model,
                np.asarray(rolling_detected_beats),
                rr_dr,
                args.batch_size,
            )
            results["D2_streaming_rolling_norm_matched_causal_ft_model"] = (
                classification_metrics(labels_dr_matched, streaming_pred_rolling)
            )
            results["D2_streaming_rolling_norm_end_to_end_causal_ft_model"] = (
                end_to_end_metrics(
                    labels_dr_all,
                    streaming_pred_rolling,
                    np.asarray(rolling_detected_positions_global, dtype=np.int64),
                )
            )
        else:
            results["streaming_weights_warning"] = f"File not found: {streaming_path}"

    results["counts"] = {
        "B_C_labels": dict(Counter(labels_bc.tolist())),
        "D_matched_labels": dict(Counter(labels_d_matched.tolist())),
        "D_all_labels": dict(Counter(labels_d_all.tolist())),
        "C2_rolling_annotation_labels": dict(Counter(labels_rolling_annotation.tolist())),
        "D2_rolling_matched_labels": dict(Counter(labels_dr_matched.tolist())),
        "D2_rolling_all_labels": dict(Counter(labels_dr_all.tolist())),
    }
    output_path.write_text(
        json.dumps(json_ready(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== STREAMING PIPELINE STUDY ===")
    for key, value in results.items():
        if isinstance(value, dict) and "macro_f1" in value:
            print(
                f"{key}: accuracy={value['accuracy']:.4f} "
                f"macro-F1={value['macro_f1']:.4f} samples={value['samples']}"
            )
    detector = results["r_peak_detection"]
    print(
        "R detector: "
        f"sensitivity={detector['sensitivity']:.4f} "
        f"PPV={detector['ppv']:.4f} F1={detector['f1']:.4f} "
        f"median|error|={detector['timing_abs_median_samples']} samples "
        f"p95|error|={detector['timing_abs_p95_samples']} samples"
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
