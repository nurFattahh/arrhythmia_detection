import serial
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import time
from  .serial_reader import SerialReader

class StreamMixin:
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
                    self.bpm_label.setStyleSheet("font-size: 50px; font-weight: bold; color: red;")
                    self.status_label.setStyleSheet("font-size: 50px; font-weight: bold; color: red;")
                elif avg_bpm < 60:
                    self.status_text = "Bradikardia"
                    self.bpm_label.setStyleSheet("font-size: 50px; font-weight: bold; color: red;")
                    self.status_label.setStyleSheet("font-size: 50px; font-weight: bold; color: red;")
                else:
                    self.status_text = "Normal"
                    self.bpm_label.setStyleSheet("font-size: 50px; font-weight: bold; color: green;")
                    self.status_label.setStyleSheet("font-size: 50px; font-weight: bold; color: green;")

                self.bpm_label.setText(f"{avg_bpm:.0f}")
                self.status_label.setText(f"{self.status_text}")
                

                if len(corrected_peaks) > 0:
                    corrected_peaks = corrected_peaks.astype(int)
                    corrected_peaks = corrected_peaks[corrected_peaks < len(signal)]

                    tb = np.array(self.time_buffer)
                    tb = tb - tb[-1]

                    # x_peaks = tb[corrected_peaks]
                    # y_peaks = signal[corrected_peaks]
                    # self.peak_scatter.setData(x_peaks, y_peaks)



        except Exception as e:
            print("Detection error:", e)

    def start_stream(self):
        port = self.com_dropdown.currentText().strip()
        baud = 115200
        self.reader = SerialReader(port=port, baudrate=baud)
        self.reader.data_signal.connect(self.on_data)
        self.reader.error_signal.connect(self.on_error)
        self.reader.connected_signal.connect(self.on_connected)
        self.reader.start()
        self.update_timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText(f"Status: Connecting to {port} @ {baud} ...")
        self._is_running = True

    def stop_stream(self):
        if self.reader:
            self.reader.stop()
            self.reader = None
        self.update_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Status: Stopped")
        self._is_running = False

    def on_connected(self, ok: bool):
        if ok:
            print("Connected!")
        else:
            print("Failed to connect.")
            

    def on_error(self, msg: str):
        print("Error:", msg)

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


    def update_plot(self):
        if len(self.time_buffer) == 0:
            return

        # Gunakan waktu absolut (bukan offset)
        tb = np.array(self.time_buffer)
        signal = np.array(self.buffer)

        # Update kurva ECG
        self.pg_curve.setData(tb, signal)

        # Range sumbu X bergerak real-time seperti live monitor
        t_min = tb[-1] - self.display_seconds
        t_max = tb[-1]
        self.pg_plot.setXRange(t_min, t_max)
        self.pg_plot.setLabel('bottom', 'Time', units='s')

        # Range amplitudo (bisa disesuaikan)
        self.pg_plot.setYRange(min(signal), max(signal))
        self.pg_plot.setLabel('left', 'Amplitude')


    def closeEvent(self, event):
        # clean up thread
        self.stop_stream()
        event.accept()