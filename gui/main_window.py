import os
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import psutil
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QLineEdit, QPushButton, QComboBox, QTextEdit, QFileDialog, QMessageBox, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .plot_mixin import PlotMixin
from .record_mixin import RecordMixin
from .stream_mixin import StreamMixin
from .log_mixin import LogMixin
from .detail_mixin import DetailMixin

from algorithm.pan_tompkins_plus_plus_v2 import Pan_Tompkins_Plus_Plus

import collections
import time
import serial
import serial.tools.list_ports
from PyQt5.QtCore import pyqtSlot
from .serial_reader import SerialReader
from .serial_reader import ShimmerReader
from PyQt5 import QtCore

# dataset paths (dapat diletakkan di core/constants jika ingin)
MITBIH_PATH = "Dataset/mit-bih-arrhythmia-database"
ECGID_PATH = "Dataset/ECG-ID_Database"

try:
    from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
    SHIMMER_AVAILABLE = True
except ImportError:
    SHIMMER_AVAILABLE = False
    print("⚠️ pyshimmer not found. Shimmer mode will be disabled.")

class ECGApp(QMainWindow, PlotMixin, RecordMixin, StreamMixin, LogMixin, DetailMixin):
    def __init__(self):
        super().__init__()
        self.start_time_system = None  # waktu mulai sistem
        self.ptpp = Pan_Tompkins_Plus_Plus()
        self.setWindowTitle("Sistem Monitoring ECG – Pan-Tompkins++")
        self.resize(1100, 650)
        self.setStyleSheet("""
            QWidget { font-family: Segoe UI; background-color: #f9f9f9; }
            QLabel { color: #222; }
            QPushButton { background-color: #1976D2; color: white; border-radius: 6px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #1565C0; }
            QComboBox, QLineEdit { background-color: white; padding: 4px; border: 1px solid #ccc; border-radius: 4px; }
        """)

        # === Layout utama ===
        main_layout = QHBoxLayout()
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # === Panel kiri: kontrol ===
        left_panel = QVBoxLayout()
        left_panel.setSpacing(4)
        top_left = QVBoxLayout()
        middle_left = QVBoxLayout()
        bottom_left = QVBoxLayout()

        left_panel.addLayout(top_left)
        left_panel.addLayout(middle_left)
        left_panel.addLayout(bottom_left)
        bottom_left.addStretch()
        main_layout.addLayout(left_panel, 1)

        title = QLabel("🫀 Sistem Deteksi Aritmia")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2;")
        top_left.addWidget(title)

        # Sumber data
        top_left.addWidget(QLabel("Sumber Data:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Dataset", "Real-time (Shimmer)"])
        self.source_combo.currentTextChanged.connect(self.toggle_input_mode)
        top_left.addWidget(self.source_combo)

        # mode serial / shimmer
        self.mode_label = QLabel("")
        top_left.addWidget(self.mode_label)
        self.mode_label.hide()

        self.mode_dropdown = QComboBox()
        self.mode_dropdown.addItems(["Shimmer", "Serial"])
        self.mode_dropdown.hide()
        top_left.addWidget(self.mode_dropdown)

        self.input_label = QLabel("Pilih Dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["MIT-BIH", "ECG-ID"])
        top_left.addWidget(self.input_label)
        top_left.addWidget(self.dataset_combo)

        # Widget container untuk line edit + browse button
        self.record_input_container = QWidget()
        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)  # rapihkan layout
        self.record_input_container.setLayout(h_layout)

        # Line edit
        self.record_input = QLineEdit()
        self.record_input.setPlaceholderText("Record Name (contoh: 100 atau Person_25/rec_4)")
        h_layout.addWidget(self.record_input)

        # Tombol browse di dalam container
        self.browse_btn = QPushButton("📂")
        self.browse_btn.setFixedWidth(40)
        self.browse_btn.clicked.connect(self.browse_file)
        h_layout.addWidget(self.browse_btn)
        self.browse_btn.setStyleSheet("""
            QPushButton {
                background-color: 0000;  
                color: white;             
                border: 1px solid #555;   
                border-radius: 5px;      
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
        """)

        top_left.addWidget(self.record_input_container)


        # Load button
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.load_record)
        top_left.addWidget(self.load_btn)
        self.record_input.returnPressed.connect(self.load_btn.click)

        # Real-time controls (initially hidden)
        self.com_dropdown = QComboBox()
        self.com_dropdown.setPlaceholderText("Pilih COM Port")
        self.com_dropdown.hide()
        top_left.addWidget(self.com_dropdown)
        self.refresh_com_btn = QPushButton("🔄 Refresh COM")
        self.refresh_com_btn.clicked.connect(self.refresh_com_ports)
        self.refresh_com_btn.hide()
        top_left.addWidget(self.refresh_com_btn)

        self.start_btn = QPushButton("▶ Start Streaming")
        self.start_btn.clicked.connect(self.start_stream)
        self.start_btn.hide()
        top_left.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self.stop_stream)
        self.stop_btn.setEnabled(False)
        self.stop_btn.hide()
        top_left.addWidget(self.stop_btn)

        self.info_label = QLabel("Status Koneksi: -")
        self.info_label.setStyleSheet("font-size: 12px; color: gray;")
        top_left.addWidget(self.info_label)

        # Detail proses
        title_detail = QLabel("📊 Detail Proses")
        title_detail.setStyleSheet("font-weight: bold; font-size: 12pt; color: #1976D2; margin-bottom: 6px;")
        middle_left.addWidget(title_detail)
        middle_frame = QFrame()
        middle_frame.setStyleSheet("QFrame { border: 1px solid #ddd; border-radius: 10px; background-color: #ffffff; padding: 10px; }")
        middle_layout = QVBoxLayout(middle_frame)
        self.detail_fs = QLabel("Frekuensi Sampling: - Hz")
        self.detail_peak = QLabel("Jumlah Peak: -")
        self.detail_interval = QLabel("Rata-rata Interval: - s")
        self.detail_exec_time = QLabel("Waktu Eksekusi Algoritma: - s")
        self.system_exec_time = QLabel("Waktu Eksekusi Sistem: - s")
        self.system_exec_time.hide() 
        self.system_memory_usage = QLabel("Penggunaan Memory: - KB")

        for label in [self.detail_fs, self.detail_peak, self.detail_interval, self.detail_exec_time, self.system_exec_time, self.system_memory_usage]:
            label.setStyleSheet("font-size: 8pt; color: #333; font-weight: bold;")
            middle_layout.addWidget(label)
        middle_left.addWidget(middle_frame)

        # Log
        bottom_left.addWidget(QLabel("📝 Log Aktivitas:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #333;
                font-family: Consolas;
                font-size: 8pt;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        self.log_box.setFixedHeight(200)
        bottom_left.addWidget(self.log_box)

        # === Panel kanan: grafik + hasil ===
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 3)
        plot_frame = QFrame()
        plot_frame.setStyleSheet("QFrame { border: 1px solid #ddd; border-radius: 12px; background-color: #ffffff; padding: 10px; }")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)

        plot_header_layout = QHBoxLayout()
        plot_title = QLabel("📈 Sinyal ECG")
        plot_title.setStyleSheet("font-weight: bold; font-size: 11pt; color: #1976D2;")
        plot_header_layout.addWidget(plot_title)
        plot_header_layout.addStretch()

        clear_btn = QPushButton("🧹 Clear")
        clear_btn.clicked.connect(self.clear_plot)
        plot_header_layout.addWidget(clear_btn)
        plot_layout.addLayout(plot_header_layout)
        clear_btn.setToolTip("Bersihkan grafik dan reset data")
        clear_btn.setStyleSheet("QPushButton { background-color: #E53935; } QPushButton:hover { background-color: #D32F2F; }")

        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(6.5, 3.5))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("motion_notify_event", self.on_hover_peak)
        plot_layout.addWidget(self.canvas)
        right_panel.addWidget(plot_frame, stretch=3)

        # PyQtGraph real-time
        self.pg_plot = pg.PlotWidget(title="Sinyal ECG Real-time")
        self.pg_plot.setBackground('w')

        self.pg_plot.showGrid(x=True, y=True, alpha=0.1)
        
        self.pg_plot.setLabel('left', 'Amplitudo (mV)')
        self.pg_plot.setLabel('bottom', 'Waktu (s)')
        self.pg_plot.setYRange(-1.5, 1.5)
        self.pg_plot.setXRange(-10, 0)
        self.ecg_curve = self.pg_plot.plot(pen=pg.mkPen(color='#1976D2', width=2))
        self.peak_scatter = self.pg_plot.plot([], [], pen=None, symbol='o', symbolBrush='r', symbolSize=8)  # titik R-peak merah


        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(3)
        effect.setOffset(0,0)
        self.pg_plot.setGraphicsEffect(effect)
        plot_layout.addWidget(self.pg_plot)
        self.pg_plot.hide()

        # === FRAME UNTUK BPM DAN STATUS ===
        bpm_frame = QFrame()
        bpm_frame.setFixedHeight(200)
        bpm_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ddd;      /* border frame luar */
                border-radius: 12px;
                background-color: #ffffff;
                padding: 12px;
            }
        """)

        bpm_layout = QHBoxLayout(bpm_frame)
        bpm_layout.setSpacing(20)
        bpm_layout.setContentsMargins(12, 12, 12, 12)

        # ----- BPM Box -----
        bpm_box = QFrame()
        bpm_box.setStyleSheet("""
            QFrame {
                background-color: #fdfdfd;
                border-radius: 8px;
                border: none;          /* Hapus border internal */
                padding: 8px;
            }
        """)
        bpm_box_layout = QVBoxLayout(bpm_box)
        bpm_box_layout.setSpacing(8)

        bpm_label_small = QLabel("BPM")
        bpm_label_small.setStyleSheet("font-size: 10pt; color: #1976D2; font-weight: bold; border: none;")
        bpm_box_layout.addWidget(bpm_label_small, alignment=Qt.AlignLeft)

        self.bpm_label = QLabel("0")
        self.bpm_label.setAlignment(Qt.AlignCenter)
        self.bpm_label.setStyleSheet("font-size: 36px; font-weight: bold; color: #1976D2; border: none;")
        bpm_box_layout.addWidget(self.bpm_label, alignment=Qt.AlignCenter)

        bpm_layout.addWidget(bpm_box)

        # ----- Garis Tegak Pemisah -----
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFrameShadow(QFrame.Sunken)  # atau Plain
        vline.setStyleSheet("color: #ffffff;")  # warna garis
        bpm_layout.addWidget(vline)
        vline.setFixedHeight(155)

        # ----- Status Box -----
        status_box = QFrame()
        status_box.setStyleSheet("""
            QFrame {
                background-color: #fdfdfd;
                border-radius: 8px;
                border: none;          /* Hapus border internal */
                padding: 8px;
            }
        """)
        status_box_layout = QVBoxLayout(status_box)
        status_box_layout.setSpacing(4)

        status_label_small = QLabel("Status")
        status_label_small.setStyleSheet("font-size: 10pt; color: #1976D2; font-weight: bold; border: none;")
        status_box_layout.addWidget(status_label_small, alignment=Qt.AlignLeft)

        self.status_label = QLabel("-")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 36px; font-weight: bold; color: gray; border: none;")
        status_box_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        bpm_layout.addWidget(status_box)

        # Tambahkan ke right_panel
        right_panel.addWidget(bpm_frame)

        # variables
        self.signal = None
        self.peaks = None
        self.streaming = False
        self.timer = QTimer()

        self.stream_index = 0
        self.fs = 128

        # --- Inisialisasi streaming ---
        self.ecg_buffer = []      # simpan nilai ECG
        self.time_buffer = []     # simpan waktu (detik)
        self.stream_index = 0
        self.streaming = True

        self.display_seconds = 15
        self.buffer_size = int(self.fs * self.display_seconds)
        self.buffer = collections.deque([0]*self.buffer_size, maxlen=self.buffer_size)

        self.ptr = 0

        # Signals for clean close
        self._is_running = False

        # Timer to update plot (GUI thread)
        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(33)  # ~30 FPS
        self.update_timer.timeout.connect(self.update_plot)

        self.serial_conn = None
        self.timer.timeout.connect(self.update_plot)

        self.buffer_size = int(self.fs * self.display_seconds)

        self.time_axis = np.linspace(-self.display_seconds, 0, self.buffer_size)
        self.signal_buffer = []
        self.ptr = 0
        self.frame_count = 0
        self.reader = None

        
        self.start_time = None                          # waktu mulai streaming
        self.max_duration = 60  # detik

        # gunakan timer untuk update plot ~30 FPS
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_plot)

        # pantompkins
        self.latest_bpm = 0
        self.last_detect_time = time.time()

        self.first_exec_time = 0
        self.first_memory_used = 0

    # ==========================================================
    def start_stream(self):
        self.start_time_system = time.time()  # waktu sistem dimulai
        self.log("--------------------------------------------------")
        self.log("Memulai streaming pada: " + time.strftime("%H:%M:%S", time.localtime(self.start_time_system)))
        self.info_label.setText("Status Koneksi: Streaming")
        self.start_time = None
        port = self.com_dropdown.currentText().strip()
        mode = self.mode_dropdown.currentText().strip()
        mode = "Shimmer"

        if "Tidak ada" in port:
            QMessageBox.warning(self, "Error", "Tidak ada port yang bisa digunakan.")
            return
        try:
            if mode == "Shimmer" and SHIMMER_AVAILABLE:
                # Gunakan ShimmerBluetooth
                self.reader = ShimmerReader(port, baudrate=DEFAULT_BAUDRATE)
                # try:
                #     # muat data konfigurasi shimmer
                #     if hasattr(self.reader, "sample_rate"):
                #         shimmer_fs = self.reader.sample_rate
                #     else:
                #         shimmer_fs = None

                #     # Jika sample rate terbaca, update GUI dan variabel
                #     if shimmer_fs:
                #         self.fs = shimmer_fs
                #         self.detail_fs.setText(f"Frekuensi Sampling: {self.fs} Hz")
                #         print(f"[INFO] Shimmer Sampling Rate: {self.fs} Hz")
                #     else:
                #         print("⚠️ Gagal membaca sampling rate dari Shimmer, gunakan default 128 Hz")
                # except Exception as e:
                #     print(f"⚠️ Tidak bisa membaca sample_rate Shimmer: {e}")
            else:
                self.reader = SerialReader(port, baudrate=115200)
                
            self.reader.new_data.connect(self.on_new_data)
            self.reader.start()
            self.timer.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Gagal membuka port {port}\n\n{str(e)}")

    def stop_stream(self):
        self.info_label.setText("Status Koneksi: Berhenti Streaming")
        if self.reader:
            self.reader.stop()
            self.reader.wait()
            self.reader = None
        
        self.log("waktu eksekusi algoritma pertama: {:.7f} s".format(self.first_exec_time))
        self.log("memory digunakan algoritma pertama: {:.3f} kb".format(self.first_memory_used))
        self.log("Streaming dihentikan pada: " + time.strftime("%H:%M:%S", time.localtime(time.time())))
        self.log("--------------------------------------------------")

        # Hentikan timer update plot
        self.timer.stop()
        self.signal_buffer.clear()
        self.time_buffer.clear()

        # Reset semua buffer dan tampilan agar mulai dari awal

        self.ptr = 0
        self.frame_count = 0
        self.start_time = None
        self.start_time_system = None  # reset waktu sistem

        # Re-enable tombol
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.first_exec_time = 0
        self.first_memory_used = 0

    # ==========================================================
    def on_new_data(self, value: float):
        now = time.time()
        if self.start_time is None:
            self.start_time = now

        t = now - self.start_time  # waktu sejak mulai (detik)
        self.signal_buffer.append(value)
        self.time_buffer.append(t)

        # jika lebih dari 60 detik data, buang yang paling awal
        max_samples = int(self.fs * self.max_duration)
        if len(self.signal_buffer) > max_samples:
            excess = len(self.signal_buffer) - max_samples
            self.signal_buffer = self.signal_buffer[excess:]
            self.time_buffer = self.time_buffer[excess:]


    @pyqtSlot()
    def update_plot(self):
        if len(self.signal_buffer) == 0:
            return

        data = np.array(self.signal_buffer)
        tdata = np.array(self.time_buffer)

        # tampilkan hanya 20 detik terakhir
        t_max = tdata[-1]
        t_min = max(0, t_max - self.display_seconds)

        # ambil data hanya di range ini
        mask = (tdata >= t_min) & (tdata <= t_max)
        data_window = data[mask]
        t_window = tdata[mask]

        self.ecg_curve.setData(t_window, data_window)
        self.pg_plot.setXRange(t_min, t_max)

        # auto scale Y sesuai data window
        if len(data_window) > 0:
            ymin, ymax = np.min(data_window), np.max(data_window)
            if ymin == ymax:
                ymin, ymax = ymin - 0.1, ymax + 0.1
            # kasih sedikit margin biar tidak mepet
            margin = 0.1 * (ymax - ymin)
            self.pg_plot.setYRange(ymin - margin, ymax + margin)

        # Update per frame lainnya tetap sama
        if self.start_time_system:
            elapsed = time.time() - self.start_time_system
            minutes, seconds = divmod(int(elapsed), 60)
            self.system_exec_time.setText(f"Waktu Eksekusi Sistem: {minutes:02d}:{seconds:02d}")


        # hitung BPM tiap 0.5 detik
        now = time.time()
        if (now - self.last_detect_time) >= 0.5:
            fs, corrected_peaks, exec_time, bpm, algo_mem = self.compute_heart_rate(data, self.fs)
            if (self.first_exec_time == 0):
                self.first_exec_time = exec_time
            
            if (self.first_memory_used == 0):
                self.first_memory_used = algo_mem

            if bpm:
                if bpm < 60:
                    color = "#E53935"  
                    status = "Bradikardia"
                elif bpm > 100:
                    color = "#FB8C00"  
                    status = "Takikardia"
                else:
                    color = "#43A047" 
                    status = "Normal"
                self.bpm_label.setStyleSheet(f"font-size:32pt; font-weight:bold; color:{color};")
                self.latest_bpm = bpm
                self.bpm_label.setText(f"{int(bpm)}")
                self.status_label.setText(status)
                self.status_label.setStyleSheet(f"font-size:32pt; font-weight:bold; color:{color};")
                
            self.last_detect_time = now
            
            self.detail_fs.setText(f"Frekuensi Sampling: {fs} Hz")
            self.detail_peak.setText(f"Jumlah R-Peak: {len(corrected_peaks)}")
            if corrected_peaks is not None and len(corrected_peaks) > 1:
                rr_intervals = np.diff(corrected_peaks) / fs
                avg_interval = np.mean(rr_intervals) * 1000
                self.detail_interval.setText(f"Rata-rata Interval: {avg_interval:.2f} ms")
            else:
                self.detail_interval.setText("Rata-rata Interval: -")
            self.detail_exec_time.setText(f"Waktu Eksekusi Algoritma: {exec_time:.7f} s")
            # mem_usage = self.get_memory_usage()
            self.system_memory_usage.setText(f"Penggunaan Memory: {algo_mem:.5f} KB")

            # === Tampilkan titik R-peak merah ===
            if corrected_peaks is not None and len(corrected_peaks) > 0:
                total_samples = len(data)
                total_time = tdata[-1] if len(tdata) > 0 else 0
                buffer_duration = total_samples / fs
                peak_times = (corrected_peaks / fs) + (total_time - buffer_duration)

                # 💡 ambil langsung nilai sinyal di index peak, bukan np.interp
                corrected_peaks = corrected_peaks[corrected_peaks < len(data)]
                peak_values = data[corrected_peaks]

                mask_peak = (peak_times >= t_min) & (peak_times <= t_max)
                self.peak_scatter.setData(peak_times[mask_peak], peak_values[mask_peak])
            else:
                self.peak_scatter.setData([], [])



            
        
            

    # ==========================================================
    def closeEvent(self, event):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.timer.stop()
        event.accept()

    def compute_heart_rate(self, signal, fs):
        """
        Hitung heart rate (BPM) menggunakan Pan-Tompkins++
        """
        try:
            start_time = time.time()
            r_peaks, algo_mem, elapsed_algo = self.measure_memory(self.ptpp.rpeak_detection, ecg=signal, fs=fs)

            
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

            # Refinement (local max di sekitar R-peak)
            refined_r_peaks = []
            window = int(0.05 * fs)
            for r in corrected_peaks:
                start = max(0, int(r - window))
                end = min(len(signal), int(r + window))
                local_max = np.argmax(signal[start:end]) + start
                refined_r_peaks.append(local_max)

            corrected_peaks = np.array(refined_r_peaks)
 

            exec_time = time.time() - start_time

            if len(corrected_peaks) > 1:
                rr_intervals = np.diff(corrected_peaks) / fs  # detik
                bpm = 60.0 / rr_intervals
                avg_bpm = np.mean(bpm)
            else:
                avg_bpm = 0.0  # fallback kalau puncak belum cukup

            # selalu return 4 nilai biar aman
            return fs, corrected_peaks, exec_time, avg_bpm, algo_mem

        except Exception as e:
            print("Pan-Tompkins++ error:", e)
            # fallback aman (return kosong tapi bisa di-unpack)
            return fs, np.array([]), 0, 0.0

    def get_memory_usage(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # hasil MB
    
    def measure_memory(self, func, *args, **kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024)

        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        mem_after = process.memory_info().rss / (1024)
        mem_used = mem_after - mem_before

        if (self.first_memory_used == 0):
            self.first_memory_used = mem_used
        print(f"[ALGO] Memory digunakan: {mem_used:.5f} kb | Waktu: {elapsed:.4f} s")

        return result, mem_used, elapsed
