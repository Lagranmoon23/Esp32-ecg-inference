"""Real-time Windows viewer for the ESP32-S3 ECG inference firmware.

Expected serial rows:
  ECG,sample_index,raw,filtered,normalized,lead_off
  PRED,r_sample,class,bpm,logit_n,logit_s,logit_vf,pre_rr,post_rr,local_rr
"""

from __future__ import annotations

import math
import sys
from collections import deque

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import serial
from serial import SerialException


PORT = sys.argv[1] if len(sys.argv) > 1 else "COM9"
BAUD_RATE = 115200
SAMPLE_RATE_HZ = 360.0
WINDOW_SECONDS = 10.0
MAX_WAVE_POINTS = 1600
MAX_PREDICTIONS = 256
CLASS_NAMES = ("N", "S", "VF")
CLASS_COLORS = {"N": "#009E73", "S": "#E69F00", "VF": "#D55E00"}

sample_indices: deque[int] = deque(maxlen=MAX_WAVE_POINTS)
sample_times: deque[float] = deque(maxlen=MAX_WAVE_POINTS)
raw_values: deque[int] = deque(maxlen=MAX_WAVE_POINTS)
filtered_values: deque[float] = deque(maxlen=MAX_WAVE_POINTS)
normalized_values: deque[float] = deque(maxlen=MAX_WAVE_POINTS)

prediction_times: deque[float] = deque(maxlen=MAX_PREDICTIONS)
prediction_classes: deque[str] = deque(maxlen=MAX_PREDICTIONS)
prediction_values: deque[float] = deque(maxlen=MAX_PREDICTIONS)
prediction_probabilities: deque[tuple[float, float, float]] = deque(
    maxlen=MAX_PREDICTIONS
)

latest_lead_off = True
latest_bpm: float | None = None
latest_class: str | None = None
latest_probabilities = (0.0, 0.0, 0.0)
last_sample_index: int | None = None


def clear_session() -> None:
    sample_indices.clear()
    sample_times.clear()
    raw_values.clear()
    filtered_values.clear()
    normalized_values.clear()
    prediction_times.clear()
    prediction_classes.clear()
    prediction_values.clear()
    prediction_probabilities.clear()


def softmax(logits: tuple[float, float, float]) -> tuple[float, float, float]:
    maximum = max(logits)
    values = [math.exp(value - maximum) for value in logits]
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def nearest_wave_value(sample_index: int) -> float:
    if not sample_indices:
        return 0.0
    indices = list(sample_indices)
    position = min(
        range(len(indices)), key=lambda index: abs(indices[index] - sample_index)
    )
    return list(filtered_values)[position]


try:
    serial_port = serial.Serial(PORT, BAUD_RATE, timeout=0)
except SerialException as error:
    raise SystemExit(
        f"Cannot open {PORT}: {error}\n"
        "Close ESP-IDF Monitor before starting this viewer."
    ) from error


figure, (wave_axis, probability_axis) = plt.subplots(
    2,
    1,
    figsize=(13, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1]},
)
figure.canvas.manager.set_window_title(f"ESP32-S3 ECG inference - {PORT}")

wave_line, = wave_axis.plot([], [], color="#0072B2", linewidth=1.1)
prediction_markers = {
    name: wave_axis.plot(
        [], [], linestyle="none", marker="o", markersize=7,
        color=CLASS_COLORS[name], label=name,
    )[0]
    for name in CLASS_NAMES
}

probability_lines = {
    name: probability_axis.plot(
        [], [], color=CLASS_COLORS[name], linewidth=1.5,
        marker="o", markersize=3, label=f"P({name})",
    )[0]
    for name in CLASS_NAMES
}

status_text = wave_axis.text(
    0.01, 0.97, "Electrodes: waiting", transform=wave_axis.transAxes,
    va="top", fontsize=12, fontweight="bold",
)
prediction_text = wave_axis.text(
    0.99, 0.97, "Prediction: waiting", transform=wave_axis.transAxes,
    va="top", ha="right", fontsize=12, fontweight="bold",
)
bpm_text = wave_axis.text(
    0.99, 0.89, "Heart rate: --", transform=wave_axis.transAxes,
    va="top", ha="right", fontsize=11,
)
adc_text = wave_axis.text(
    0.01, 0.89, "ADC: --", transform=wave_axis.transAxes,
    va="top", fontsize=10,
)

wave_axis.set_title("AD8232 real-time ECG with ESP32-S3 Tiny Transformer")
wave_axis.set_ylabel("Filtered ECG (ADC counts)")
wave_axis.set_xlim(-WINDOW_SECONDS, 0.0)
wave_axis.grid(True, alpha=0.25)
wave_axis.legend(loc="lower left", ncol=3)

probability_axis.set_xlabel("Time relative to latest sample (seconds)")
probability_axis.set_ylabel("Probability")
probability_axis.set_ylim(-0.05, 1.05)
probability_axis.grid(True, alpha=0.25)
probability_axis.legend(loc="upper left", ncol=3)


def parse_ecg(fields: list[str]) -> None:
    global latest_lead_off, last_sample_index

    if len(fields) != 6:
        return
    try:
        sample_index = int(fields[1])
        raw = int(fields[2])
        filtered = float(fields[3])
        normalized = float(fields[4])
        lead_off = bool(int(fields[5]))
    except ValueError:
        return

    latest_lead_off = lead_off
    if lead_off:
        return
    if last_sample_index is not None and sample_index < last_sample_index:
        clear_session()
    last_sample_index = sample_index
    sample_indices.append(sample_index)
    sample_times.append(sample_index / SAMPLE_RATE_HZ)
    raw_values.append(raw)
    filtered_values.append(filtered)
    normalized_values.append(normalized)


def parse_prediction(fields: list[str]) -> None:
    global latest_bpm, latest_class, latest_probabilities

    if len(fields) != 10:
        return
    try:
        r_sample = int(fields[1])
        class_name = fields[2]
        bpm = float(fields[3])
        logits = (float(fields[4]), float(fields[5]), float(fields[6]))
    except ValueError:
        return
    if class_name not in CLASS_NAMES:
        return

    probabilities = softmax(logits)
    prediction_times.append(r_sample / SAMPLE_RATE_HZ)
    prediction_classes.append(class_name)
    prediction_values.append(nearest_wave_value(r_sample))
    prediction_probabilities.append(probabilities)
    latest_bpm = bpm
    latest_class = class_name
    latest_probabilities = probabilities


def read_serial() -> None:
    for _ in range(1200):
        raw_line = serial_port.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if line.startswith("ECG,"):
            parse_ecg(line.split(","))
        elif line.startswith("PRED,"):
            parse_prediction(line.split(","))


def update_plot(_frame):
    read_serial()
    if not sample_times:
        status_text.set_text(
            "Electrodes: detached" if latest_lead_off else "Electrodes: warming up"
        )
        status_text.set_color("red" if latest_lead_off else "#E69F00")
        return tuple([wave_line, *prediction_markers.values(),
                      *probability_lines.values(), status_text,
                      prediction_text, bpm_text, adc_text])

    latest_time = sample_times[-1]
    relative_times = [value - latest_time for value in sample_times]
    values = list(filtered_values)
    wave_line.set_data(relative_times, values)

    minimum = min(values)
    maximum = max(values)
    margin = max(10.0, 0.12 * max(maximum - minimum, 1.0))
    wave_axis.set_ylim(minimum - margin, maximum + margin)

    for class_name, marker in prediction_markers.items():
        visible = [
            (time - latest_time, value)
            for time, predicted, value in zip(
                prediction_times, prediction_classes, prediction_values
            )
            if predicted == class_name and time >= latest_time - WINDOW_SECONDS
        ]
        marker.set_data(
            [item[0] for item in visible], [item[1] for item in visible]
        )

    visible_predictions = [
        (time - latest_time, probabilities)
        for time, probabilities in zip(
            prediction_times, prediction_probabilities
        )
        if time >= latest_time - WINDOW_SECONDS
    ]
    for class_index, class_name in enumerate(CLASS_NAMES):
        probability_lines[class_name].set_data(
            [item[0] for item in visible_predictions],
            [item[1][class_index] for item in visible_predictions],
        )

    status_text.set_text("Electrodes: connected")
    status_text.set_color("green")
    adc_text.set_text(f"ADC: {raw_values[-1]}")
    if latest_class is None:
        prediction_text.set_text("Prediction: warming up")
        prediction_text.set_color("#E69F00")
        bpm_text.set_text("Heart rate: acquiring...")
    else:
        confidence = max(latest_probabilities) * 100.0
        prediction_text.set_text(
            f"Prediction: {latest_class} ({confidence:.1f}%)"
        )
        prediction_text.set_color(CLASS_COLORS[latest_class])
        bpm_text.set_text(f"Heart rate: {latest_bpm:.1f} BPM")

    return tuple([wave_line, *prediction_markers.values(),
                  *probability_lines.values(), status_text,
                  prediction_text, bpm_text, adc_text])


def close_serial(_event) -> None:
    if serial_port.is_open:
        serial_port.close()


figure.canvas.mpl_connect("close_event", close_serial)
plot_animation = animation.FuncAnimation(
    figure, update_plot, interval=40, cache_frame_data=False,
)
plt.tight_layout()
plt.show()
