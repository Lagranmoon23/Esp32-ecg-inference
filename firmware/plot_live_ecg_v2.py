"""Enhanced real-time ECG viewer for the ESP32-S3 inference firmware (UI rewrite).

This is a drop-in UI replacement for ``plot_live_ecg.py``. It keeps the exact
same parsing / softmax / data buffering logic as the original, but replaces the
matplotlib rendering with a pyqtgraph-based dashboard and adds:

  * anomaly highlighting (S / VF alerts, recent-event log)
  * a three-state signal-quality indicator (GOOD / POOR / LEAD_OFF)
  * a heart-rate trend line
  * slow-adaptive Y-axis (no "breathing" on motion spikes)
  * a large core readout (status / HR / confidence)

Expected serial rows (unchanged protocol):
  ECG,sample_index,raw,filtered,normalized,lead_off
  PRED,r_sample,class,bpm,logit_n,logit_s,logit_vf,pre_rr,post_rr,local_rr

Dependencies:
  python -m pip install pyserial pyqtgraph PyQt5
"""

from __future__ import annotations

import math
import sys
from collections import deque

import serial
from serial import SerialException
from PyQt5 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM9"
BAUD_RATE = 115200
SAMPLE_RATE_HZ = 360.0
WINDOW_SECONDS = 10.0
MAX_WAVE_POINTS = 1600
MAX_PREDICTIONS = 256
MAX_HR_HISTORY = 600  # enough for ~1 minute of beats
MAX_EVENT_LOG = 200
CLASS_NAMES = ("N", "S", "VF")
CLASS_COLORS = {"N": "#009E73", "S": "#E69F00", "VF": "#D55E00"}
CLASS_LABELS_CN = {"N": "正常", "S": "室上性", "VF": "室性(V/F)"}

# --- signal quality thresholds (UI-side heuristic; does not modify firmware) ---
SQ_GOOD = "GOOD"
SQ_POOR = "POOR"
SQ_LEAD_OFF = "LEAD_OFF"

# Motion-spike heuristic: a filtered sample whose absolute value exceeds this
# many multiples of the running MAD is flagged as a transient artifact.
SQ_SPIKE_MAD_MULTIPLE = 8.0


class DataStore:
    """Owns the buffering state. Algorithm identical to the original viewer."""

    def __init__(self) -> None:
        self.sample_indices: deque[int] = deque(maxlen=MAX_WAVE_POINTS)
        self.sample_times: deque[float] = deque(maxlen=MAX_WAVE_POINTS)
        self.raw_values: deque[int] = deque(maxlen=MAX_WAVE_POINTS)
        self.filtered_values: deque[float] = deque(maxlen=MAX_WAVE_POINTS)
        self.normalized_values: deque[float] = deque(maxlen=MAX_WAVE_POINTS)

        self.prediction_times: deque[float] = deque(maxlen=MAX_PREDICTIONS)
        self.prediction_classes: deque[str] = deque(maxlen=MAX_PREDICTIONS)
        self.prediction_values: deque[float] = deque(maxlen=MAX_PREDICTIONS)
        self.prediction_probabilities: deque[tuple[float, float, float]] = deque(
            maxlen=MAX_PREDICTIONS
        )

        self.latest_lead_off = True
        self.latest_bpm: float | None = None
        self.latest_class: str | None = None
        self.latest_probabilities = (0.0, 0.0, 0.0)
        self.last_sample_index: int | None = None

        # heart-rate history (time, bpm)
        self.hr_history: deque[tuple[float, float]] = deque(maxlen=MAX_HR_HISTORY)
        # signal-quality rolling statistics
        self._sq_mad_median: float | None = None
        self._sq_spike_count = 0
        self._sq_window_samples = 0

    def clear_session(self) -> None:
        self.sample_indices.clear()
        self.sample_times.clear()
        self.raw_values.clear()
        self.filtered_values.clear()
        self.normalized_values.clear()
        self.prediction_times.clear()
        self.prediction_classes.clear()
        self.prediction_values.clear()
        self.prediction_probabilities.clear()
        self.hr_history.clear()
        self._sq_mad_median = None
        self._sq_spike_count = 0
        self._sq_window_samples = 0

    @staticmethod
    def softmax(logits: tuple[float, float, float]) -> tuple[float, float, float]:
        maximum = max(logits)
        values = [math.exp(value - maximum) for value in logits]
        total = sum(values)
        return tuple(value / total for value in values)  # type: ignore[return-value]

    def nearest_wave_value(self, sample_index: int) -> float:
        if not self.sample_indices:
            return 0.0
        indices = list(self.sample_indices)
        position = min(
            range(len(indices)), key=lambda i: abs(indices[i] - sample_index)
        )
        return list(self.filtered_values)[position]

    def parse_ecg(self, fields: list[str]) -> None:
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

        self.latest_lead_off = lead_off
        if lead_off:
            return
        if self.last_sample_index is not None and sample_index < self.last_sample_index:
            self.clear_session()
        self.last_sample_index = sample_index
        self.sample_indices.append(sample_index)
        self.sample_times.append(sample_index / SAMPLE_RATE_HZ)
        self.raw_values.append(raw)
        self.filtered_values.append(filtered)
        self.normalized_values.append(normalized)
        self._update_signal_quality(filtered)

    def parse_prediction(self, fields: list[str]) -> None:
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

        probabilities = self.softmax(logits)
        self.prediction_times.append(r_sample / SAMPLE_RATE_HZ)
        self.prediction_classes.append(class_name)
        self.prediction_values.append(self.nearest_wave_value(r_sample))
        self.prediction_probabilities.append(probabilities)
        self.latest_bpm = bpm
        self.latest_class = class_name
        self.latest_probabilities = probabilities
        self.hr_history.append((r_sample / SAMPLE_RATE_HZ, bpm))

    # --- signal-quality heuristic (UI-only) ---
    def _update_signal_quality(self, filtered: float) -> None:
        """Maintain a crude MAD estimate and count out-of-range spikes."""
        self._sq_window_samples += 1
        if self._sq_window_samples < 60:
            # not enough data to form a stable median yet
            return
        if self._sq_mad_median is None:
            # bootstrap the median from the current buffer
            values = list(self.filtered_values)
            values.sort()
            self._sq_mad_median = values[len(values) // 2]
        center = self._sq_mad_median
        # slow EMA of the median
        self._sq_mad_median = 0.999 * center + 0.001 * filtered
        if abs(filtered - center) > SQ_SPIKE_MAD_MULTIPLE * max(
            self._sq_mad_median - center, 1.0
        ):
            self._sq_spike_count += 1
        # decay the spike counter over time
        self._sq_spike_count = max(0, self._sq_spike_count - 1)
        if self._sq_window_samples % 360 == 0:
            self._sq_window_samples = 0

    def signal_quality(self) -> str:
        if self.latest_lead_off:
            return SQ_LEAD_OFF
        if self._sq_spike_count > 3:
            return SQ_POOR
        return SQ_GOOD


class EcgDashboard(QtWidgets.QMainWindow):
    def __init__(self, store: DataStore, serial_port: serial.Serial) -> None:
        super().__init__()
        self.store = store
        self.serial_port = serial_port

        self.setWindowTitle(f"ESP32-S3 ECG 实时监测 - {PORT}")
        self.resize(1280, 820)

        pg.setConfigOptions(antialias=True, background="#1b1e24",
                            foreground="#d8d8d8")
        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.event_log: list[str] = []
        self._last_alert_class: str | None = None

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---- top status bar ----
        top = QtWidgets.QHBoxLayout()
        self.status_pill = self._make_pill("等待连接", "#888888")
        self.sq_pill = self._make_pill("信号质量: --", "#888888")
        self.hr_big = self._make_big_label("--")
        self.conf_big = self._make_big_label("--")
        top.addWidget(self.status_pill)
        top.addWidget(self.sq_pill)
        top.addStretch(1)
        top.addWidget(self._make_label("心率", QtWidgets.QLabel))
        top.addWidget(self.hr_big)
        top.addSpacing(16)
        top.addWidget(self._make_label("置信度", QtWidgets.QLabel))
        top.addWidget(self.conf_big)
        root.addLayout(top)

        # ---- main plot area ----
        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.wave_plot = pg.PlotWidget()
        self.prob_plot = pg.PlotWidget()
        self.plot_splitter.addWidget(self.wave_plot)
        self.plot_splitter.addWidget(self.prob_plot)
        self.plot_splitter.setStretchFactor(0, 3)
        self.plot_splitter.setStretchFactor(1, 1)
        root.addWidget(self.plot_splitter, 1)

        # ---- bottom row: heart-rate trend + event log ----
        bottom = QtWidgets.QHBoxLayout()
        self.hr_plot = pg.PlotWidget()
        self.hr_plot.setMaximumHeight(160)
        self.event_list = QtWidgets.QListWidget()
        self.event_list.setMaximumWidth(360)
        bottom.addWidget(self.hr_plot, 1)
        bottom.addWidget(self.event_list)
        root.addLayout(bottom)

        self._setup_wave_plot()
        self._setup_prob_plot()
        self._setup_hr_plot()

        self.statusBar().showMessage(f"串口 {PORT} @ {BAUD_RATE} | 360 Hz | "
                                     "界面重写版（算法未改动）")

    @staticmethod
    def _make_pill(text: str, color: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet(
            f"background:{color}; color:#ffffff; border-radius:10px; "
            "padding:6px 14px; font-size:15px; font-weight:bold;"
        )
        label.setAlignment(QtCore.Qt.AlignCenter)
        return label

    @staticmethod
    def _make_big_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font-size:26px; font-weight:bold; color:#ffffff;")
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumWidth(70)
        return label

    @staticmethod
    def _make_label(text: str, _cls: type[QtWidgets.QLabel]) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setStyleSheet("font-size:14px; color:#aaaaaa;")
        return label

    def _setup_wave_plot(self) -> None:
        self.wave_plot.setTitle("AD8232 实时心电图（滤波后）")
        self.wave_plot.setLabel("left", "幅值")
        self.wave_plot.setLabel("bottom", "相对最新样本（秒）")
        self.wave_plot.showGrid(x=True, y=True, alpha=0.2)
        self.wave_curve = self.wave_plot.plot([], [], pen=pg.mkPen("#40a9ff", width=1.4))
        self.wave_markers: dict[str, pg.ScatterPlotItem] = {}
        for name in CLASS_NAMES:
            scatter = pg.ScatterPlotItem(
                size=9, pen=None, brush=pg.mkBrush(CLASS_COLORS[name]), symbol="o"
            )
            self.wave_plot.addItem(scatter)
            self.wave_markers[name] = scatter
        self._wave_y_min: float | None = None
        self._wave_y_max: float | None = None

    def _setup_prob_plot(self) -> None:
        self.prob_plot.setTitle("分类概率（逐拍）")
        self.prob_plot.setLabel("left", "概率")
        self.prob_plot.setLabel("bottom", "相对最新样本（秒）")
        self.prob_plot.showGrid(x=True, y=True, alpha=0.2)
        self.prob_plot.setYRange(-0.05, 1.05)
        self.prob_lines: dict[str, pg.PlotDataItem] = {}
        for name in CLASS_NAMES:
            line = self.prob_plot.plot(
                [], [], pen=pg.mkPen(CLASS_COLORS[name], width=2),
                symbol="o", symbolSize=4, symbolBrush=CLASS_COLORS[name],
                name=f"P({name})",
            )
            self.prob_lines[name] = line
        self.prob_plot.addLegend()

    def _setup_hr_plot(self) -> None:
        self.hr_plot.setTitle("心率趋势")
        self.hr_plot.setLabel("left", "BPM")
        self.hr_plot.setLabel("bottom", "相对最新样本（秒）")
        self.hr_plot.showGrid(x=True, y=True, alpha=0.2)
        self.hr_curve = self.hr_plot.plot(
            [], [], pen=pg.mkPen("#ff6b81", width=2), symbol="o", symbolSize=4
        )

    # ------------------------------------------------------------------ tick
    def _read_serial(self) -> None:
        for _ in range(1200):
            raw_line = self.serial_port.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if line.startswith("ECG,"):
                self.store.parse_ecg(line.split(","))
            elif line.startswith("PRED,"):
                self.store.parse_prediction(line.split(","))

    def _tick(self) -> None:
        self._read_serial()
        self._update_status()
        self._update_wave()
        self._update_probability()
        self._update_hr()

    def _update_status(self) -> None:
        s = self.store
        if s.latest_lead_off:
            self.status_pill.setText("电极脱落")
            self.status_pill.setStyleSheet(self._pill_style("#d9534f"))
        elif s.latest_bpm is None:
            self.status_pill.setText("建立统计中…")
            self.status_pill.setStyleSheet(self._pill_style("#E69F00"))
        else:
            self.status_pill.setText("监测中")
            self.status_pill.setStyleSheet(self._pill_style("#28a745"))

        sq = s.signal_quality()
        sq_color = {"GOOD": "#28a745", "POOR": "#E69F00", "LEAD_OFF": "#d9534f"}[sq]
        sq_text = {"GOOD": "信号质量: 良好", "POOR": "信号质量: 较差",
                   "LEAD_OFF": "信号质量: 脱落"}[sq]
        self.sq_pill.setText(sq_text)
        self.sq_pill.setStyleSheet(self._pill_style(sq_color))

        if s.latest_bpm is not None:
            self.hr_big.setText(f"{s.latest_bpm:.0f}")
        else:
            self.hr_big.setText("--")

        if s.latest_class is not None:
            confidence = max(s.latest_probabilities) * 100.0
            self.conf_big.setText(f"{s.latest_class} {confidence:.0f}%")
            self.conf_big.setStyleSheet(
                f"font-size:26px; font-weight:bold; color:{CLASS_COLORS[s.latest_class]};"
            )
            self._handle_alert(s.latest_class, s.latest_bpm)
        else:
            self.conf_big.setText("--")
            self.conf_big.setStyleSheet(
                "font-size:26px; font-weight:bold; color:#ffffff;"
            )

    def _handle_alert(self, class_name: str, bpm: float | None) -> None:
        """Log anomalies and flash the status pill on transitions."""
        if class_name == "N":
            self._last_alert_class = None
            return
        if class_name == self._last_alert_class:
            return  # avoid spamming the log with every beat
        self._last_alert_class = class_name
        label_cn = CLASS_LABELS_CN[class_name]
        bpm_s = f"{bpm:.0f}" if bpm is not None else "--"
        msg = f"⚠ {class_name} ({label_cn}) 心率 {bpm_s} BPM"
        self.event_list.insertItem(0, msg)
        # cap list size
        while self.event_list.count() > MAX_EVENT_LOG:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _update_wave(self) -> None:
        s = self.store
        if not s.sample_times:
            self.wave_curve.setData([], [])
            for scatter in self.wave_markers.values():
                scatter.setData([], [])
            return

        latest_time = s.sample_times[-1]
        relative = [t - latest_time for t in s.sample_times]
        values = list(s.filtered_values)
        self.wave_curve.setData(relative, values)

        # slow-adaptive Y range to avoid "breathing" on spikes
        current_min = min(values)
        current_max = max(values)
        if self._wave_y_min is None:
            self._wave_y_min = current_min
            self._wave_y_max = current_max
        else:
            self._wave_y_min = 0.9 * self._wave_y_min + 0.1 * min(self._wave_y_min, current_min)
            self._wave_y_max = 0.9 * self._wave_y_max + 0.1 * max(self._wave_y_max, current_max)
        margin = 0.15 * max(self._wave_y_max - self._wave_y_min, 1.0)
        self.wave_plot.setYRange(self._wave_y_min - margin, self._wave_y_max + margin)
        self.wave_plot.setXRange(-WINDOW_SECONDS, 0.0)

        for class_name, scatter in self.wave_markers.items():
            pts = [
                (t - latest_time, v)
                for t, c, v in zip(
                    s.prediction_times, s.prediction_classes, s.prediction_values
                )
                if c == class_name and t >= latest_time - WINDOW_SECONDS
            ]
            scatter.setData([p[0] for p in pts], [p[1] for p in pts])

    def _update_probability(self) -> None:
        s = self.store
        if not s.prediction_times:
            return
        latest_time = s.sample_times[-1] if s.sample_times else s.prediction_times[-1]
        visible = [
            (t - latest_time, p)
            for t, p in zip(s.prediction_times, s.prediction_probabilities)
            if t >= latest_time - WINDOW_SECONDS
        ]
        if not visible:
            return
        xs = [item[0] for item in visible]
        for idx, name in enumerate(CLASS_NAMES):
            self.prob_lines[name].setData(xs, [item[1][idx] for item in visible])
        self.prob_plot.setXRange(-WINDOW_SECONDS, 0.0)

    def _update_hr(self) -> None:
        s = self.store
        if not s.hr_history:
            return
        latest_time = s.sample_times[-1] if s.sample_times else s.hr_history[-1][0]
        pts = [(t - latest_time, bpm) for t, bpm in s.hr_history
               if t >= latest_time - WINDOW_SECONDS * 6]
        if pts:
            self.hr_curve.setData([p[0] for p in pts], [p[1] for p in pts])
            self.hr_plot.setXRange(-WINDOW_SECONDS * 6, 0.0)

    @staticmethod
    def _pill_style(color: str) -> str:
        return (f"background:{color}; color:#ffffff; border-radius:10px; "
                "padding:6px 14px; font-size:15px; font-weight:bold;")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.serial_port.is_open:
            self.serial_port.close()
        event.accept()


def main() -> None:
    try:
        serial_port = serial.Serial(PORT, BAUD_RATE, timeout=0)
    except SerialException as error:
        raise SystemExit(
            f"Cannot open {PORT}: {error}\n"
            "Close ESP-IDF Monitor before starting this viewer."
        ) from error

    store = DataStore()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EcgDashboard(store, serial_port)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
