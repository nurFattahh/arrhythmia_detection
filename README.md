# Real-Time ECG Monitoring using Pan-Tompkins++ and Shimmer ECG Sensor

This project implements the **Pan-Tompkins++ algorithm** for **real-time R-peak detection**, **heart rate (BPM) computation**, and **basic arrhythmia identification** using ECG data from a **Shimmer ECG Sensor**.  
The system provides a full processing pipeline, real-time visualization GUI, and dataset examples for offline testing.

---

## 📌 Features

- Real-time ECG acquisition (Shimmer ECG)
- Robust QRS & R-peak detection using **Pan-Tompkins++**
- BPM calculation using adaptive RR interval
- Automatic classification:
  - **Bradycardia (< 60 BPM)**
  - **Normal (60–100 BPM)**
  - **Tachycardia (> 100 BPM)**
- GUI built with **PyQt5 + PyQtGraph**
- Offline analysis using sample datasets
- Clean and modular code structure


---

## 📦 Requirements

These are the required Python dependencies:
numpy
scipy
matplotlib
pyqt5
pyqtgraph
pandas
psutil
wfdb
peakutils
pyshimmer

All dependencies can be installed easily as shown below.

---

# 🔧 Installation Guide

## 1️⃣ Clone the Repository
git clone https://github.com/username/repo-name.git

cd repo-name

---

## 2️⃣ Install Requirements

### (use virtual environment)

#### Create virtual environment
python -m venv venv

#### Activate environment  
venv\Scripts\activate
#### Install packages
pip install -r requirements.txt


---

# ▶️ Running the Program

Run the main application using:
python main.py


The GUI will open and start real-time streaming (if Shimmer connected) or allow browsing offline dataset mode (if implemented).

---

# 🔌 Connecting Shimmer ECG (Check COM Port)

Before starting real-time streaming, **ensure the Shimmer ECG is paired via Bluetooth**.

### ✔ Check COM Port via Bluetooth Settings
1. Open **Windows Settings → Bluetooth & Devices**
2. Make sure your **Shimmer ECG** is paired  
   (example name: `Shimmer3-ECG`)
3. After pairing, scroll down to **More Bluetooth Settings**
4. Check the **assigned COM Port**  
   Example: `COM5`, `COM7`, etc.

### ✔ Select COM Port Through the GUI
You **do not need to edit config.py manually**.  
After opening the program:

- Go to the **Port Selection dropdown** in the GUI  
- Choose the detected COM port (e.g., **COM5**)  
- Click **Start** to begin real-time ECG monitoring

If the wrong port is chosen, simply stop, restart, and select another one.

---

# 🧠 Pan-Tompkins++ Algorithm Overview

The project implements the enhanced **Pan-Tompkins++** algorithm with improvements over the classic version.

### Processing Pipeline:
1. **Bandpass filter (5–18 Hz)**
2. **Differentiation**  
   Enhances slope of QRS
3. **Squaring**  
   Amplifies large peaks
4. **Flattop moving average smoothing**
5. **Moving Window Integration (150 ms)**
6. **Adaptive 3-threshold detection**
7. **Search-back method for missed R-peaks**
8. **RR interval averaging**
9. **BPM calculation**
10. **Arrhythmia classification**

This implementation is optimized for **real-time processing** at ~0.5s update intervals.

---

# 📊 Dataset

Sample ECG files are provided under:
Dataset_example/

pls use folder Dataset (not Dataset_example) for running program without error
You can add more datasets (`.csv`, `.txt`, MIT-BIH format) for testing or benchmarking.

---

# 🖥 GUI Preview (Features)

- Real-time ECG waveform display  
- Detected R-peaks marked on plot  
- BPM updated in real-time  
- Heart rhythm classification  
- System resource usage (optional)  
- Start/Stop streaming  
- Load offline dataset

---

# 👨‍💻 Author

**Muhammad Nur Fattah**  
Computer Engineering  
Universitas Brawijaya  

---

# 📜 License

This project is created for academic and research purposes (Skripsi).  
Commercial use is not permitted unless approved by the author.

