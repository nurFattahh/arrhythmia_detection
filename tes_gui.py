# ecg_realtime.py
import sys
import collections
import time
import numpy as np

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import serial
import serial.tools.list_ports

# -----------------------
# Thread which reads serial
# -----------------------
class SerialReader(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(int)          # emits one integer value
    error_signal = QtCore.pyqtSignal(str)
    connected_signal = QtCore.pyqtSignal(bool)

    def __init__(self, port="COM5", baudrate=115200, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self._running = False
        self._ser = None

    def open_serial(self):
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.connected_signal.emit(True)
            return True
        except Exception as e:
            self._ser = None
            self.connected_signal.emit(False)
            self.error_signal.emit(f"Cannot open {self.port}: {e}")
            return False

    def close_serial(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self.connected_signal.emit(False)
        except Exception as e:
            self.error_signal.emit(f"Error closing serial: {e}")

    def run(self):
        if not self.open_serial():
            return
        self._running = True
        # Read loop
        while self._running:
            try:
                line = self._ser.readline()
                if not line:
                    continue
                # decode and parse integer
                try:
                    s = line.decode(errors='ignore').strip()
                    if s == "":
                        continue
                    # allow possible noise: remove non-numeric trailing/leading
                    # try parse int
                    val = int(float(s))
                    self.data_signal.emit(val)
                except Exception:
                    # ignore parse errors
                    continue
            except Exception as e:
                self.error_signal.emit(f"Serial read error: {e}")
                break
        self.close_serial()

    def stop(self):
        self._running = False
        # give some time to exit run loop
        self.wait(2000)


# -----------------------
# Main Window
# -----------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        from algorithm.pan_tompkins_plus_plus_v2 import Pan_Tompkins_Plus_Plus
        self.ptpp = Pan_Tompkins_Plus_Plus()
        self.latest_bpm = 0
        self.status_text = "Idle"
        


        self.setWindowTitle("ECG Realtime Viewer (PyQt + pyqtgraph)")
        self.resize(1000, 600)

        # Central widget
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout()
        w.setLayout(layout)

        # Top controls: port, baud, start/stop
        ctrl_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_layout)

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_ports()
        ctrl_layout.addWidget(QtWidgets.QLabel("COM Port:"))
        ctrl_layout.addWidget(self.port_combo)

        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.addItems(["9600","19200","38400","57600","115200","230400"])
        self.baud_combo.setCurrentText("115200")
        ctrl_layout.addWidget(QtWidgets.QLabel("Baud:"))
        ctrl_layout.addWidget(self.baud_combo)

        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_ports)
        ctrl_layout.addWidget(self.btn_refresh)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)

        ctrl_layout.addStretch()
        self.bpm_label = QtWidgets.QLabel("BPM: -")
        self.bpm_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        ctrl_layout.addWidget(self.bpm_label)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="ECG Signal (Realtime)")
        self.plot_widget.setLabel('left', 'ADC value')
        self.plot_widget.setLabel('bottom', 'time', units='s')
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget)

        # small status bar
        self.status = QtWidgets.QLabel("Status: Idle")
        layout.addWidget(self.status)

        # Plot data buffer: keep ~8 seconds default
        self.fs = 360  # default sampling rate suggested by you
        self.display_seconds = 8
        self.buffer_size = int(self.fs * self.display_seconds)
        self.buffer = collections.deque([0]*self.buffer_size, maxlen=self.buffer_size)
        self.time_buffer = collections.deque(np.linspace(-self.display_seconds, 0, self.buffer_size), maxlen=self.buffer_size)

        self.curve = self.plot_widget.plot(np.array(self.time_buffer), np.array(self.buffer), pen=pg.mkPen(width=1))
        self.ptr = 0

        # Serial reader thread placeholder
        self.reader = None

        # Timer to update plot (GUI thread)
        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(33)  # ~30 FPS
        self.update_timer.timeout.connect(self.update_plot)

        # Connections
        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

        # For BPM detection (simple)
        self.last_update = time.time()
        self.peak_times = collections.deque(maxlen=20)  # store times of last peaks

        self.peak_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None), brush='r', size=6)
        self.plot_widget.addItem(self.peak_scatter)

        # Signals for clean close
        self._is_running = False


    def try_detect_peak(self, value):
        # Jalankan Pan-Tompkins++ setiap 0.5 detik (biar tidak terlalu berat)
        now = time.time()
        if not hasattr(self, "last_detection_time"):
            self.last_detection_time = 0
        if (now - self.last_detection_time) < 0.5:
            return
        self.last_detection_time = now

        signal = np.array(self.buffer)
        fs = self.fs

        try:
            r_peaks = self.ptpp.rpeak_detection(ecg=signal, fs=fs)

            # Hilangkan duplikasi peak (<200 ms)
            corrected_peaks = []
            new_thresh = int(0.200 * fs)
            flag = 0
            for i in range(len(r_peaks)):
                if i > 0 and (r_peaks[i] - r_peaks[i-1]) < new_thresh:
                    if flag == 0:
                        flag = 1
                        continue
                corrected_peaks.append(r_peaks[i])
                flag = 0
            corrected_peaks = np.array(corrected_peaks)

            # Hitung BPM
            if len(corrected_peaks) > 1:
                rr_intervals = np.diff(corrected_peaks) / fs
                bpm = 60 / rr_intervals
                avg_bpm = np.mean(bpm)
                self.latest_bpm = avg_bpm

                if avg_bpm > 100:
                    self.status_text = "Takikardia"
                elif avg_bpm < 60:
                    self.status_text = "Bradikardia"
                else:
                    self.status_text = "Normal"

                self.bpm_label.setText(f"BPM: {avg_bpm:.0f} ({self.status_text})")

                if len(corrected_peaks) > 0:
                    corrected_peaks = corrected_peaks.astype(int)
                    corrected_peaks = corrected_peaks[corrected_peaks < len(signal)]

                    tb = np.array(self.time_buffer)
                    tb = tb - tb[-1]

                    x_peaks = tb[corrected_peaks]
                    y_peaks = signal[corrected_peaks]
                    self.peak_scatter.setData(x_peaks, y_peaks)



        except Exception as e:
            print("Detection error:", e)

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        names = [p.device for p in ports]
        # common defaults windows COM1..COM20
        if not names:
            # still offer COM5 and COM6 as defaults
            names = ["COM5","COM6"]
        self.port_combo.addItems(names)

    def start_stream(self):
        port = self.port_combo.currentText()
        baud = int(self.baud_combo.currentText())
        self.reader = SerialReader(port=port, baudrate=baud)
        self.reader.data_signal.connect(self.on_data)
        self.reader.error_signal.connect(self.on_error)
        self.reader.connected_signal.connect(self.on_connected)
        self.reader.start()
        self.update_timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status.setText(f"Status: Connecting to {port} @ {baud} ...")
        self._is_running = True

    def stop_stream(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        self.update_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText("Status: Stopped")
        self._is_running = False

    def on_connected(self, ok: bool):
        if ok:
            self.status.setText("Status: Connected - streaming")
        else:
            self.status.setText("Status: Connection failed")

    def on_error(self, msg: str):
        self.status.setText("Status: ERROR - " + msg)

    def on_data(self, value: int):
        # append value to buffer, update time buffer accordingly
        self.buffer.append(value)
        # increment time pointer: compute dt from fs
        dt = 1.0 / self.fs
        last_time = self.time_buffer[-1] if len(self.time_buffer) else 0.0
        new_time = last_time + dt
        self.time_buffer.append(new_time)
        # Keep time_buffer ending at 0 (relative): we will normalize in update_plot
        # Peak detection: simple threshold + local maxima
        self.try_detect_peak(value)

    # def try_detect_peak(self, value):
    #     # We'll use a very simple heuristic:
    #     # If value > mean + threshold and sufficient time since last peak -> count as peak
    #     arr = np.array(self.buffer)
    #     mean = arr.mean()
    #     std = arr.std() if arr.size>0 else 1.0
    #     threshold = mean + max(0.15 * (arr.max()-arr.min()), 0.5)  # adaptive
    #     now = time.time()
    #     min_interval = 0.35  # seconds between R-peaks (max 170 bpm)
    #     # check last recorded peak time
    #     last_peak = self.peak_times[-1] if self.peak_times else 0
    #     if value > threshold and (now - last_peak) > min_interval:
    #         # To reduce false positives, also ensure this is a local high in the buffer:
    #         # compare to previous few samples
    #         if len(self.buffer) >= 5:
    #             b = list(self.buffer)[-5:]
    #             # consider center sample as candidate (the newest one)
    #             if b[-1] == max(b):
    #                 self.peak_times.append(now)
    #                 self.update_bpm()

    def update_bpm(self):
        if len(self.peak_times) >= 2:
            # compute intervals and convert to BPM (average)
            intervals = np.diff(np.array(self.peak_times))
            if intervals.size == 0:
                return
            avg_interval = intervals.mean()
            bpm = 60.0 / avg_interval if avg_interval > 0 else 0
            self.bpm_label.setText(f"BPM: {bpm:.0f}")
        else:
            self.bpm_label.setText("BPM: -")

    def update_plot(self):
        if len(self.time_buffer) == 0:
            return
        # normalize time axis to end at 0 (seconds)
        tb = np.array(self.time_buffer)
        tb = tb - tb[-1]  # so last point at 0
        self.curve.setData(tb, np.array(self.buffer))
        # keep x-range to display_seconds window
        self.plot_widget.setXRange(-self.display_seconds, 0)

    def closeEvent(self, event):
        # clean up thread
        self.stop_stream()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    # set dark background for pyqtgraph
    pg.setConfigOption('background', 'w')   # white bg
    pg.setConfigOption('foreground', 'k')   # black lines
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
