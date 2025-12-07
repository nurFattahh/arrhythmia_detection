import numpy as np
from scipy.interpolate import interp1d
import peakutils
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import get_window
import scipy.signal as sig
import os

def smoother(signal=None, kernel='boxzen', size=10, mirror=True, **kwargs):
    """Smoothing function"""
    if signal is None:
        raise TypeError("Please specify a signal to smooth.")

    signal_data = signal
    length = len(signal_data)

    if isinstance(kernel, str):
        if size > length:
            size = length - 1
        if size < 1:
            size = 1

        if kernel == 'boxzen':
            aux = smoother(signal_data, kernel='boxcar', size=size, mirror=mirror)
            smoothed = smoother(aux, kernel='parzen', size=size, mirror=mirror)
            return smoothed

        elif kernel == 'median':
            if size % 2 == 0:
                raise ValueError("When the kernel is 'median', size must be odd.")
            smoothed = sig.medfilt(signal_data, kernel_size=size)
            return smoothed

        else:
            win = get_window(kernel, size, **kwargs)

    elif isinstance(kernel, np.ndarray):
        win = kernel
        size = len(win)
        if size > length:
            raise ValueError("Kernel size is bigger than signal length.")
        if size < 1:
            raise ValueError("Kernel size is smaller than 1.")
    else:
        raise TypeError("Unknown kernel type.")

    w = win / win.sum()
    if mirror:
        aux = np.concatenate(
            (signal_data[0] * np.ones(size), signal_data, signal_data[-1] * np.ones(size)))
        smoothed = np.convolve(w, aux, mode='same')
        smoothed = smoothed[size:-size]
    else:
        smoothed = np.convolve(w, signal_data, mode='same')

    return smoothed


def generate_synthetic_ecg(duration=10, fs=200):
    """Generate synthetic ECG signal"""
    samples = int(fs * duration)
    t = np.arange(samples) / fs
    
    ecg = np.zeros(samples)
    heart_rate = 75  # BPM
    period = 60 / heart_rate
    
    for i, time in enumerate(t):
        phase = (time % period) / period
        
        # P wave
        if 0.1 < phase < 0.2:
            ecg[i] += 0.15 * np.sin(np.pi * (phase - 0.1) / 0.1)
        
        # QRS complex
        if 0.3 < phase < 0.38:
            qrs_phase = (phase - 0.3) / 0.08
            if qrs_phase < 0.3:
                ecg[i] -= 0.3 * np.sin(np.pi * qrs_phase / 0.3)  # Q
            elif qrs_phase < 0.6:
                ecg[i] += 1.2 * np.sin(np.pi * (qrs_phase - 0.3) / 0.3)  # R
            else:
                ecg[i] -= 0.4 * np.sin(np.pi * (qrs_phase - 0.6) / 0.4)  # S
        
        # T wave
        if 0.5 < phase < 0.7:
            ecg[i] += 0.25 * np.sin(np.pi * (phase - 0.5) / 0.2)
    
    # Add baseline wander
    ecg += 0.15 * np.sin(2 * np.pi * 0.5 * t)
    
    # Add noise
    ecg += 0.05 * np.random.randn(samples)
    
    return ecg, t


def load_mitbih_record(mitbih_path, record_name=None, duration=10):
    """Load MIT-BIH record"""
    try:
        import wfdb
        
        # List available records
        records = [f.split('.')[0] for f in os.listdir(mitbih_path) if f.endswith('.dat')]
        records = sorted(list(set(records)))
        
        if not records:
            raise FileNotFoundError("No .dat files found in the specified directory")
        
        print(f"Available records: {', '.join(records[:10])}...")
        
        # Use first record if not specified
        if record_name is None:
            record_name = records[9]
        
        print(f"Reading record: {record_name}")
        
        record_path = os.path.join(mitbih_path, record_name)
        record = wfdb.rdrecord(record_path)
        
        # Get ECG signal (usually channel 0)
        ecg = record.p_signal[:, 0]
        fs = record.fs
        
        # Use specified duration
        if duration is not None:
            samples = int(duration * fs)
            ecg = ecg[:samples]
        
        t = np.arange(len(ecg)) / fs
        
        print(f"Record info:")
        print(f"  - Record name: {record_name}")
        print(f"  - Sampling frequency: {fs} Hz")
        print(f"  - Signal length: {len(ecg)} samples ({len(ecg)/fs:.2f} seconds)")
        
        return ecg, t, fs
        
    except ImportError:
        raise ImportError("wfdb library not found! Install it using: pip install wfdb")
    except Exception as e:
        raise Exception(f"Error reading MIT-BIH data: {e}")


def pan_tompkins_visualization(ecg, fs):
    """
    Pan-Tompkins++ dengan visualisasi semua tahapan
    """
    
    # Store all stages
    stages = {}
    
    # ========== STAGE 0: Raw Signal ==========
    stages['raw'] = ecg.copy()
    print("Stage 0: Raw ECG Signal")
    print(f"  - Length: {len(ecg)} samples")
    print(f"  - Max: {np.max(ecg):.4f}, Min: {np.min(ecg):.4f}")
    
    # ========== STAGE 1: Bandpass Filter (5-18 Hz) ==========
    print("\nStage 1: Bandpass Filtering (5-18 Hz)")
    
    if fs == 200:
        ecg = ecg - np.mean(ecg)
        
        # Lowpass filter
        Wn = 12*2/fs
        N = 3
        a, b = signal.butter(N, Wn, btype='lowpass')
        ecg_l = signal.filtfilt(a, b, ecg)
        ecg_l = ecg_l/np.max(np.abs(ecg_l))
        
        # Highpass filter
        Wn = 5*2/fs
        N = 3
        a, b = signal.butter(N, Wn, btype='highpass')
        ecg_h = signal.filtfilt(a, b, ecg_l, padlen=3*(max(len(a), len(b))-1))
        ecg_h = ecg_h/np.max(np.abs(ecg_h))
    else:
        f1 = 5
        f2 = 18
        Wn = [f1*2/fs, f2*2/fs]
        N = 3
        a, b = signal.butter(N=N, Wn=Wn, btype='bandpass')
        ecg_h = signal.filtfilt(a, b, ecg, padlen=3*(max(len(a), len(b)) - 1))
        ecg_h = ecg_h/np.max(np.abs(ecg_h))
    
    stages['bandpass'] = ecg_h.copy()
    print(f"  - Filtered signal normalized")
    print(f"  - Max: {np.max(ecg_h):.4f}, Min: {np.min(ecg_h):.4f}")
    
    # ========== STAGE 2: Derivative Filter ==========
    print("\nStage 2: Derivative Filter")
    
    vector = [1, 2, 0, -2, -1]
    if fs != 200:
        int_c = 160/fs
        b = interp1d(range(1, 6), [i*fs/8 for i in vector])(np.arange(1, 5.1, int_c))
    else:
        b = [i*fs/8 for i in vector]
    
    ecg_d = signal.filtfilt(b, 1, ecg_h, padlen=3*(max(len(a), len(b)) - 1))
    ecg_d = ecg_d/np.max(ecg_d)
    
    stages['derivative'] = ecg_d.copy()
    print(f"  - Derivative kernel: {vector}")
    print(f"  - Max: {np.max(ecg_d):.4f}, Min: {np.min(ecg_d):.4f}")
    
    # ========== STAGE 3: Squaring ==========
    print("\nStage 3: Squaring")
    
    ecg_s = ecg_d**2
    
    stages['squared'] = ecg_s.copy()
    print(f"  - Signal squared (all positive)")
    print(f"  - Max: {np.max(ecg_s):.4f}, Min: {np.min(ecg_s):.4f}")
    
    # ========== STAGE 4: Smoothing ==========
    print("\nStage 4: Smoothing")
    
    sm_size = int(0.06 * fs)
    ecg_s = smoother(signal=ecg_s, kernel='flattop', size=sm_size, mirror=True)
    
    stages['smoothed'] = ecg_s.copy()
    print(f"  - Smoothing window size: {sm_size} samples ({sm_size/fs*1000:.1f} ms)")
    print(f"  - Max: {np.max(ecg_s):.4f}, Min: {np.min(ecg_s):.4f}")
    
    # ========== STAGE 5: Moving Window Integration ==========
    print("\nStage 5: Moving Window Integration (MWI)")
    
    temp_vector = np.ones((1, round(0.150*fs)))/round(0.150*fs)
    temp_vector = temp_vector.flatten()
    ecg_m = np.convolve(ecg_s, temp_vector)
    
    delay = round(0.150*fs)/2
    
    stages['mwi'] = ecg_m.copy()
    print(f"  - Window size: 150 ms ({round(0.150*fs)} samples)")
    print(f"  - Delay: {delay} samples ({delay/fs*1000:.1f} ms)")
    print(f"  - Max: {np.max(ecg_m):.4f}, Min: {np.min(ecg_m):.4f}")
    
    # ========== STAGE 6: Peak Detection ==========
    print("\nStage 6: Peak Detection")
    
    locs = peakutils.indexes(y=ecg_m, thres=0, min_dist=round(0.231*fs))
    pks = np.array([ecg_m[val] for val in locs])
    
    print(f"  - Minimum distance: {round(0.231*fs)} samples ({231} ms)")
    print(f"  - Total peaks found: {len(locs)}")
    
    # Filter low amplitude peaks
    amp_threshold = 0.001 * np.max(ecg_m)
    valid_mask = pks > amp_threshold
    locs = locs[valid_mask]
    pks = pks[valid_mask]
    
    print(f"  - Amplitude threshold: {amp_threshold:.6f}")
    print(f"  - Valid peaks after filtering: {len(locs)}")
    
    # ========== STAGE 7: Initialize Thresholds ==========
    print("\nStage 7: Threshold Initialization (2 sec training)")
    
    THR_SIG = np.max(ecg_m[:2*fs+1]) * 1/3
    THR_NOISE = np.mean(ecg_m[:2*fs+1]) * 1/2
    
    print(f"  - THR_SIG (Signal Threshold): {THR_SIG:.4f}")
    print(f"  - THR_NOISE (Noise Threshold): {THR_NOISE:.4f}")
    
    THR_SIG1 = np.max(ecg_h[:2*fs+1]) * 1/3
    THR_NOISE1 = np.mean(ecg_h[:2*fs+1]) * 1/2
    
    print(f"  - THR_SIG1 (Bandpass Signal Threshold): {THR_SIG1:.4f}")
    print(f"  - THR_NOISE1 (Bandpass Noise Threshold): {THR_NOISE1:.4f}")
    
    # Classify peaks
    qrs_peaks = []
    noise_peaks = []
    
    for i, loc in enumerate(locs):
        if pks[i] >= THR_SIG:
            qrs_peaks.append(loc)
        else:
            noise_peaks.append(loc)
    
    print(f"\nClassification Results:")
    print(f"  - QRS peaks (signal): {len(qrs_peaks)}")
    print(f"  - Noise peaks: {len(noise_peaks)}")
    
    return stages, locs, pks, THR_SIG, THR_NOISE, qrs_peaks, noise_peaks


def plot_all_stages(stages, t, locs, pks, THR_SIG, THR_NOISE, qrs_peaks, noise_peaks, fs):
    """
    Plot semua tahapan preprocessing di window terpisah
    """
    
    colors = {
        'signal': '#2563eb',
        'qrs': '#10b981',
        'noise': '#ef4444',
        'threshold_sig': '#f59e0b',
        'threshold_noise': '#8b5cf6'
    }
    
    # Adjust time for MWI (has extra samples due to convolution)
    t_mwi = np.arange(len(stages['mwi'])) / fs
    
    figures = []
    
    # Stage 0: Raw ECG
    fig0 = plt.figure(figsize=(12, 4))
    plt.plot(t, stages['raw'], color=colors['signal'], linewidth=1.5)
    plt.title('Stage 0: Raw ECG Signal', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig0)
    
    # Stage 1: Bandpass Filter
    fig1 = plt.figure(figsize=(12, 4))
    plt.plot(t, stages['bandpass'], color=colors['signal'], linewidth=1.5)
    plt.title('Stage 1: Bandpass Filter (5-18 Hz)', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig1)
    
    # Stage 2: Derivative
    fig2 = plt.figure(figsize=(12, 4))
    plt.plot(t, stages['derivative'], color=colors['signal'], linewidth=1.5)
    plt.title('Stage 2: Derivative Filter', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig2)
    
    # Stage 3: Squared
    fig3 = plt.figure(figsize=(12, 4))
    plt.plot(t, stages['squared'], color=colors['signal'], linewidth=1.5)
    plt.title('Stage 3: Squared Signal', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig3)
    
    # Stage 4: Smoothed
    fig4 = plt.figure(figsize=(12, 4))
    plt.plot(t, stages['smoothed'], color=colors['signal'], linewidth=1.5)
    plt.title('Stage 4: Smoothed Signal', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig4)
    
    # Stage 5: MWI
    fig5 = plt.figure(figsize=(12, 4))
    plt.plot(t_mwi, stages['mwi'], color=colors['signal'], linewidth=1.5)
    plt.title('Stage 5: Moving Window Integration (150ms)', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig5)
    
    # Stage 6: Peak Detection with Classification
    fig6 = plt.figure(figsize=(12, 4))
    plt.plot(t_mwi, stages['mwi'], color=colors['signal'], linewidth=1.5, label='MWI Signal')
    
    # Plot thresholds
    plt.axhline(y=THR_SIG, color=colors['threshold_sig'], linestyle='--', 
                linewidth=2, label=f'THR_SIG = {THR_SIG:.4f}')
    plt.axhline(y=THR_NOISE, color=colors['threshold_noise'], linestyle='--', 
                linewidth=2, label=f'THR_NOISE = {THR_NOISE:.4f}')
    
    # Plot QRS peaks
    if qrs_peaks:
        qrs_times = np.array(qrs_peaks) / fs
        qrs_amps = stages['mwi'][qrs_peaks]
        plt.scatter(qrs_times, qrs_amps, color=colors['qrs'], s=100, 
                   marker='o', zorder=5, label=f'QRS Peaks ({len(qrs_peaks)})', 
                   edgecolors='black', linewidths=1.5)
    
    # Plot Noise peaks
    if noise_peaks:
        noise_times = np.array(noise_peaks) / fs
        noise_amps = stages['mwi'][noise_peaks]
        plt.scatter(noise_times, noise_amps, color=colors['noise'], s=100, 
                   marker='x', zorder=5, label=f'Noise Peaks ({len(noise_peaks)})',
                   linewidths=2)
    
    plt.title('Stage 6: Peak Detection & Classification', fontweight='bold', fontsize=13)
    plt.ylabel('Amplitude', fontsize=11)
    plt.xlabel('Time (seconds)', fontsize=11)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([t[0], t[-1]])
    plt.tight_layout()
    figures.append(fig6)
    
    return figures


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # MIT-BIH Database path
    mitbih_path = r"D:\coding\Skripsi\Dataset\mit-bih-arrhythmia-database"
    
    print("="*60)
    print("PAN-TOMPKINS++ ALGORITHM VISUALIZATION")
    print("="*60)
    
    # Try to load MIT-BIH data
    try:
        print("\nReading MIT-BIH Arrhythmia Database...")
        ecg, t, fs = load_mitbih_record(mitbih_path, record_name=None, duration=10)
        
    except Exception as e:
        print(f"\n{e}")
        print("\nFalling back to synthetic ECG...")
        fs = 360  # MIT-BIH default sampling frequency
        duration = 10
        ecg, t = generate_synthetic_ecg(duration=duration, fs=fs)
        print(f"Generated {len(ecg)} samples ({duration} seconds at {fs} Hz)")
    
    # Process with Pan-Tompkins++
    print("\n" + "="*60)
    print("PROCESSING STAGES")
    print("="*60)
    
    stages, locs, pks, THR_SIG, THR_NOISE, qrs_peaks, noise_peaks = \
        pan_tompkins_visualization(ecg, fs)
    
    # Plot all stages
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION")
    print("="*60)
    
    figures = plot_all_stages(stages, t, locs, pks, THR_SIG, THR_NOISE, 
                              qrs_peaks, noise_peaks, fs)
    
    # Save all figures
    stage_names = ['raw', 'bandpass', 'derivative', 'squared', 'smoothed', 'mwi', 'peak_detection']
    for i, fig in enumerate(figures):
        filename = f'stage_{i}_{stage_names[i]}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    print("\nAll visualizations saved!")
    print("Close each window to continue...")
    
    plt.show()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)