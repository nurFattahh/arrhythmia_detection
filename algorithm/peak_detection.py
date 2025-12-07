import psutil
import os
import time
import numpy as np
import wfdb
from .pan_tompkins_plus_plus_v2 import Pan_Tompkins_Plus_Plus

def process_record(record_path, max_duration=20):
    record = wfdb.rdrecord(record_path)
    fs = record.fs
    signal = record.p_signal[:, 0]  # channel pertama

    # Ambil hanya max_duration detik
    max_samples = fs * max_duration
    if len(signal) > max_samples:
        p = 1

    ptpp = Pan_Tompkins_Plus_Plus()

    # 💡 --- MULAI PENGUKURAN MEMORI ---
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024  # dalam KB

    start_time = time.time()
    r_peaks = ptpp.rpeak_detection(ecg=signal, fs=fs)
    exec_time = time.time() - start_time

    mem_after = process.memory_info().rss / 1024  # dalam KB
    algo_mem_used = mem_after - mem_before  # penggunaan memori algoritma
    # 💡 --- SELESAI PENGUKURAN MEMORI ---

    # Koreksi duplikasi (<200 ms)
    corrected_peaks = []
    new_thresh = int(0.200 * fs)
    flag = 0
    for i in range(len(r_peaks)):
        if i > 0:
            if (r_peaks[i] - r_peaks[i-1]) < new_thresh:
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

    refined_r_peaks = np.array(refined_r_peaks)

    # Hitung BPM
    rr_intervals = np.diff(refined_r_peaks) / fs
    bpm = 60 / rr_intervals
    avg_bpm = np.mean(bpm) if len(bpm) > 0 else 0

    # Klasifikasi
    if avg_bpm > 100:
        status = "Takikardia"
    elif avg_bpm < 60:
        status = "Bradikardia"
    else:
        status = "Normal"

    # 💬 tambahkan algo_mem_used di return
    return fs, signal, refined_r_peaks, avg_bpm, status, exec_time, algo_mem_used


# ptpp_global = Pan_Tompkins_Plus_Plus()  # bisa reuse agar tidak init tiap kali

# def process_record_streamable(signal_buffer, fs):
#     """
#     Proses buffer ECG untuk streaming real-time.
#     Mengembalikan:
#     - refined_r_peaks: array index R-peak relatif buffer
#     - avg_bpm: rata-rata BPM
#     - status: Normal / Takikardia / Bradikardia
#     """
#     signal = np.array(signal_buffer)
    
#     if len(signal) < fs * 2:  # minimal 2 detik data
#         return [], 0, "-"
    
#     if np.std(signal[:fs]) < 0.05:  # ambil 1 detik pertama
#         return [], 0, "-"

#     ptpp = Pan_Tompkins_Plus_Plus()
#     r_peaks = ptpp.rpeak_detection(ecg=signal, fs=fs)

#     # Koreksi duplikasi (<200 ms)
#     corrected_peaks = []
#     new_thresh = int(0.200 * fs)
#     flag = 0
#     for i in range(len(r_peaks)):
#         if i > 0:
#             if (r_peaks[i] - r_peaks[i-1]) < new_thresh:
#                 if flag == 0:
#                     flag = 1
#                     continue
#         corrected_peaks.append(r_peaks[i])
#         flag = 0
#     corrected_peaks = np.array(corrected_peaks)

#     # Hitung BPM
#     if len(corrected_peaks) > 1:
#         rr_intervals = np.diff(corrected_peaks) / fs
#         bpm = 60 / rr_intervals
#         avg_bpm = np.mean(bpm)
#     else:
#         avg_bpm = 0

#     # Refinement
#     refined_r_peaks = []
#     window = int(0.05 * fs)
#     for r in corrected_peaks:
#         start = max(0, int(r - window))
#         end = min(len(signal), int(r + window))
#         local_max = np.argmax(signal[start:end]) + start
#         refined_r_peaks.append(local_max)

#     refined_r_peaks = np.array(refined_r_peaks)

#     # Status
#     if avg_bpm > 100:
#         status = "Takikardia"
#     elif avg_bpm < 60 and avg_bpm > 0:
#         status = "Bradikardia"
#     elif avg_bpm == 0:
#         status = "-"
#     else:
#         status = "Normal"

#     return refined_r_peaks, avg_bpm, status

