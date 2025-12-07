import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from algorithm.peak_detection import process_record

class RecordMixin:
    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih File Rekaman ECG", "", "ECG Files (*.dat *.txt *.csv);;All Files (*)", options=options)
        if file_path:
            self.record_input.setText(file_path)

    def load_record(self):
        MITBIH_PATH = "Dataset/mit-bih-arrhythmia-database"
        ECGID_PATH = "Dataset/ECG-ID_Database"
        dataset = self.dataset_combo.currentText()
        record_input_text = self.record_input.text().strip()
        if not record_input_text:
            QMessageBox.warning(self, "Input Kosong", "Masukkan nama record atau pilih file terlebih dahulu.")
            return

        # determine WFDB-compatible path
        if os.path.isfile(record_input_text) and record_input_text.endswith('.dat'):
            folder = os.path.dirname(record_input_text)
            base_name = os.path.basename(record_input_text).replace('.dat', '')
            record_path_for_wfdb = os.path.join(folder, base_name)
            display_name = os.path.basename(record_input_text)
        else:
            folder = MITBIH_PATH if dataset == 'MIT-BIH' else ECGID_PATH
            base_name = record_input_text
            record_path_for_wfdb = os.path.join(folder, base_name)
            display_name = base_name

        dat_file = record_path_for_wfdb + '.dat'
        hea_file = record_path_for_wfdb + '.hea'
        if not os.path.exists(dat_file) or not os.path.exists(hea_file):
            QMessageBox.critical(self, "Error", f"File record tidak lengkap:\n{dat_file}\natau\n{hea_file} tidak ditemukan.")
            return

        try:
            fs, signal, peaks, avg_bpm, status, exec_time, algo_mem_used = process_record(record_path_for_wfdb)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.signal = np.asarray(signal)
        self.peaks = np.asarray(peaks).astype(int) if peaks is not None else None
        self.fs = fs

        self.plot_signal(self.signal, self.peaks, display_name)
        self.update_bpm_and_status(avg_bpm, status)

        self.info_label.setText("Status Koneksi: Dataset dimuat ✅")
    
        self.detail_fs.setText(f"Frekuensi Sampling: {fs} Hz")
        self.detail_peak.setText(f"Jumlah R-Peak: {len(self.peaks) if self.peaks is not None else 0}")
        if self.peaks is not None and len(self.peaks) > 1:
            rr_intervals = np.diff(self.peaks) / fs
            avg_interval = np.mean(rr_intervals) * 1000
            self.detail_interval.setText(f"Rata-rata Interval: {avg_interval:.2f} ms")
        else:
            self.detail_interval.setText("Rata-rata Interval: -")
        self.detail_exec_time.setText(f"Waktu Eksekusi: {exec_time:.7f} s")
        self.system_memory_usage.setText(f"Memori Algoritma: {algo_mem_used:.2f} kb")

        if hasattr(self, 'log'):
            self.log("============================")
            self.log(f"Dataset {display_name}")
            self.log(f"Frekuensi Sampling: {fs} Hz")