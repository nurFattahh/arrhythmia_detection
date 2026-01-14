import os
import numpy as np
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from pan_tompkins_plus_plus_v2 import Pan_Tompkins_Plus_Plus


# ============================================================
# 🔹 FUNGSI DASAR: DETEKSI & EVALUASI
# ============================================================
def detect_rpeaks(signal, fs):
    ptpp = Pan_Tompkins_Plus_Plus()
    r_peaks = ptpp.rpeak_detection(ecg=signal, fs=fs)

    # Koreksi duplikasi <200 ms
    corrected = []
    thresh = int(0.200 * fs)
    flag = 0

    for i in range(len(r_peaks)):
        if i > 0 and (r_peaks[i] - r_peaks[i - 1]) < thresh:
            if flag == 0:
                flag = 1
                continue
        corrected.append(r_peaks[i])
        flag = 0

    return np.array(corrected, dtype=int)


def evaluate_rpeak_detection(pred_peaks, true_peaks, fs, tol=0.1):
    tolerance = int(tol * fs)
    tp, fp, fn = 0, 0, 0
    matched = []

    # Cocokan prediksi → ground truth
    for p in pred_peaks:
        diffs = np.abs(true_peaks - p)
        if np.any(diffs <= tolerance):
            tp += 1
            t = true_peaks[np.argmin(diffs)]
            matched.append((p, t))
        else:
            fp += 1

    fn = len(true_peaks) - tp

    # Hitung metrik akurasi
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) > 0 else 0

    # 🔥 Error timing (ms)
    if len(matched) > 0:
        diffs = [(p - t) / fs * 1000 for p, t in matched]  # ms
        mae = np.mean(np.abs(diffs))
        std = np.std(diffs)
    else:
        mae, std = None, None

    return {
        'TP': tp, 'FP': fp, 'FN': fn,
        'Sensitivity': sens,
        'Precision': prec,
        'F1': f1,
        'MAE_ms': mae,
        'STD_ms': std
    }


# ============================================================
# 🔹 PLOT (Annotation vs Algorithm)
# ============================================================
def plot_comparison(signal, fs, true_peaks, pred_peaks, record_name, dataset_name):
    t = np.arange(len(signal)) / fs
    true_peaks = np.array(true_peaks, dtype=int)
    pred_peaks = np.array(pred_peaks, dtype=int)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --- Plot Annotasi ---
    axs[0].plot(t, signal, color='black', linewidth=1)
    axs[0].scatter(true_peaks / fs, signal[true_peaks], color='red', s=40, label='Annotation')
    axs[0].set_title(f"{dataset_name} - {record_name} (Annotation)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # --- Plot Algoritma ---
    axs[1].plot(t, signal, color='black', linewidth=1)
    axs[1].scatter(pred_peaks / fs, signal[pred_peaks], color='blue', s=25, label='Algorithm')
    axs[1].set_title(f"{dataset_name} - {record_name} (Algorithm Output)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 🔹 EVALUASI MIT-BIH
# ============================================================
def evaluate_mitbih(dataset_path, plot_records=None, max_plots=5):
    results = []
    plotted = 0

    record_files = [f for f in os.listdir(dataset_path) if f.endswith('.dat')]

    for dat_file in record_files:
        record_name = os.path.splitext(dat_file)[0]
        record_path = os.path.join(dataset_path, record_name)

        try:
            rec = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')

            fs = rec.fs
            signal = rec.p_signal[:, 0]

            pred = detect_rpeaks(signal, fs)
            metrics = evaluate_rpeak_detection(pred, np.array(ann.sample), fs)
            metrics['record'] = record_name
            results.append(metrics)

            print(f"[OK] MIT-BIH {record_name} | F1={metrics['F1']:.3f} | MAE={metrics['MAE_ms']}ms")

            if (plot_records and record_name in plot_records) or (not plot_records and plotted < max_plots):
                plot_comparison(signal, fs, ann.sample, pred, record_name, "MIT-BIH")
                plotted += 1

        except Exception as e:
            print(f"[ERROR] MIT-BIH {record_name}: {e}")

    return pd.DataFrame(results)


# ============================================================
# 🔹 EVALUASI ECG-ID
# ============================================================
def evaluate_ecgid(dataset_path, plot_records=None, max_plots=5):
    results = []
    plotted = 0

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if not f.endswith('.dat'):
                continue

            record_path = os.path.join(root, os.path.splitext(f)[0])
            ecg_id = os.path.relpath(record_path, dataset_path).replace("\\", "/")

            ann_file = None
            for ext in ['qrs', 'atr', 'man']:
                if os.path.exists(f"{record_path}.{ext}"):
                    ann_file = ext
                    break
            if not ann_file:
                continue

            try:
                rec = wfdb.rdrecord(record_path)
                ann = wfdb.rdann(record_path, ann_file)

                fs = rec.fs
                signal = rec.p_signal[:, 0]

                # Ambil hanya anotasi 'N'
                if hasattr(ann, 'symbol'):
                    mask = np.isin(ann.symbol, ['N'])
                    ann_samples = np.array(ann.sample)[mask]
                else:
                    ann_samples = np.array(ann.sample)

                if len(ann_samples) == 0:
                    continue

                # 10 N terakhir
                selected_r = ann_samples[-10:]
                last_idx = selected_r[-1]

                # Potong sinyal
                signal = signal[:last_idx + int(0.5 * fs)]

                pred = detect_rpeaks(signal, fs)
                metrics = evaluate_rpeak_detection(pred, selected_r, fs)
                metrics['record'] = ecg_id
                results.append(metrics)

                print(f"[OK] ECG-ID {ecg_id} | F1={metrics['F1']:.3f} | MAE={metrics['MAE_ms']}ms")

                if (plot_records and ecg_id in plot_records) or (not plot_records and plotted < max_plots):
                    plot_comparison(signal, fs, selected_r, pred, ecg_id, "ECG-ID")
                    plotted += 1

            except Exception as e:
                print(f"[ERROR] ECG-ID {ecg_id}: {e}")

    return pd.DataFrame(results)


# ============================================================
# 🔹 MAIN PROGRAM
# ============================================================
mitbih_path = r"D:\coding\Skripsi\Dataset\mit-bih-arrhythmia-database"
ecgid_path = r"D:\coding\Skripsi\Dataset\ECG-ID_Database"
output_path = r"D:\coding\Skripsi\monitoringApp\algorithm\evaluasi_pantompkins2.xlsx"

records_to_plot_ecgid = [
    "Person_02/rec_16", 
    "Person_02/rec_18", 

]

records_to_plot_mitbih = [ "207", "233"]


print("\n=== Evaluasi ECG-ID ===")
df_ecgid = evaluate_ecgid(ecgid_path, plot_records=records_to_plot_ecgid)
print(df_ecgid.head())
print(df_ecgid[['F1', 'MAE_ms', 'STD_ms']].mean())


print("\n=== Evaluasi MIT-BIH ===")
df_mitbih = evaluate_mitbih(mitbih_path, plot_records=records_to_plot_mitbih)
print(df_mitbih.head())
print(df_mitbih[['F1', 'MAE_ms', 'STD_ms']].mean())


# Simpan Excel
with pd.ExcelWriter(output_path) as writer:
    df_ecgid.to_excel(writer, sheet_name="ECG-ID", index=False)
    df_mitbih.to_excel(writer, sheet_name="MIT-BIH", index=False)

print("\n✅ Selesai — hasil disimpan ke Excel")
