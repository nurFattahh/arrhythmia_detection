import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

from peak_detection import process_record


# ============================================================
# 🚀 Fungsi: proses CSV menggunakan Pan-Tompkins++
# ============================================================
def process_csv(csv_path, fs=128, max_duration=20):
    """
    Membaca CSV hasil streaming (time,value)
    lalu memanggil process_record() yang berbasis WFDB.
    CSV harus memiliki kolom: time,value
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File tidak ditemukan: {csv_path}")

    # === baca CSV ===
    df = pd.read_csv(csv_path)

    if not {"time", "value"}.issubset(df.columns):
        raise ValueError("CSV harus memiliki kolom: time,value")

    signal = df["value"].values.astype(float)

    # === buat file WFDB sementara agar cocok dengan process_record ===
    temp_dir = tempfile.mkdtemp()
    base = os.path.join(temp_dir, "temp_ecg")

    wfdb.wrsamp(
        record_name="temp_ecg",
        fs=fs,
        sig_name=["ecg"],
        units=["mV"],
        p_signal=signal.reshape(-1, 1),
        write_dir=temp_dir
    )

    # === panggil fungsi Pan-Tompkins++ ===
    fs, signal_out, peaks, bpm, status, exec_time, mem_used = process_record(base)

    return fs, signal_out, peaks, bpm, status, exec_time, mem_used


# ============================================================
# 🚀 Fungsi: plot ECG + R-peaks
# ============================================================
def plot_ecg(signal, peaks, bpm, status, fs=128, time_array=None):
    """
    signal      : array sinyal
    peaks       : index R-peak
    fs          : sampling rate
    time_array  : dari CSV (kalau ada)
    """

    # Buat time axis (detik)
    if time_array is None:
        t = np.arange(len(signal)) / fs
    else:
        t = time_array

    # Konversi peak index → detik
    peak_times = np.array(peaks) / fs

    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="ECG")

    if len(peaks) > 0:
        plt.scatter(peak_times, signal[peaks], color="red", label="R-peak")

    plt.title(f"ECG Plot | BPM={bpm:.1f} | Status={status}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 🚀 MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    # === ubah sesuai file CSV kamu ===
    CSV_PATH = r"D:\coding\Skripsi\monitoringApp\algorithm\recordtes1.csv"
    FS = 128   # ganti sesuai FS Shimmer kamu (128 atau 256)

    print("🔍 Membaca & memproses CSV...")
    fs, signal, peaks, bpm, status, exec_time, mem_used = process_csv(CSV_PATH, fs=FS)

    print("\n=== HASIL DETEKSI ===")
    print(f"Frekuensi Sampling  : {fs} Hz")
    print(f"Jumlah R-peak       : {len(peaks)}")
    print(f"BPM                 : {bpm:.2f}")
    print(f"Status              : {status}")
    print(f"Waktu Eksekusi      : {exec_time:.6f} s")
    print(f"Memori Algoritma    : {mem_used:.4f} KB")

    print("\n📈 Menampilkan plot...")
    plot_ecg(signal, peaks, bpm, status)
