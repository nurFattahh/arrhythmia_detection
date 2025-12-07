import numpy as np
from PyQt5.QtWidgets import QToolTip
from PyQt5.QtGui import QCursor

class PlotMixin:
    def plot_signal(self, signal, peaks, record_name=""):
        # ensure signal is numpy array
        signal = np.asarray(signal)
        self.ax.clear()
        time_axis = np.arange(len(signal)) / max(1, self.fs)
        self.ax.plot(time_axis, signal, label="ECG", linewidth=1)

        if peaks is not None and len(peaks) > 0:
            peaks = np.asarray(peaks).astype(int)
            # guard indices
            peaks = peaks[(peaks >= 0) & (peaks < len(signal))]
            self.ax.plot(peaks / max(1, self.fs), signal[peaks], "ro", label="R-Peaks")

        self.ax.set_xlabel("Waktu (s)")
        self.ax.set_ylabel("Amplitudo")
        self.ax.set_title(f"Rekaman: {record_name}")
        self.ax.grid(True)
        self.canvas.draw()

        # annotation for hover
        annot = self.ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                                 bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind, line):
            xdata, ydata = line.get_data()
            idx = ind["ind"][0]
            annot.xy = (xdata[idx], ydata[idx])
            annot.set_text(f"t={xdata[idx]:.2f}s\nAmp={ydata[idx]:.3f}")

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == self.ax:
                for line in self.ax.get_lines():
                    cont, ind = line.contains(event)
                    if cont and line.get_label() == "R-Peaks":
                        update_annot(ind, line)
                        annot.set_visible(True)
                        self.canvas.draw_idle()
                        return
            if vis:
                annot.set_visible(False)
                self.canvas.draw_idle()

        # reconnect hover handler (avoid duplicate handlers)
        try:
            self._mpl_hover_cid
        except AttributeError:
            self._mpl_hover_cid = self.canvas.mpl_connect("motion_notify_event", hover)

    def clear_plot(self):
        self.ax.clear()
        self.ecg_curve.setData([], [])
        self.ax.set_title("")
        self.canvas.draw()
        
        self.bpm_label.setText("0")
        self.bpm_label.setStyleSheet("font-size: 32pt; font-weight: bold; color: #1976D2;")

        self.status_label.setText("-")
        self.status_label.setStyleSheet("font-size: 32pt; font-weight: bold; color: #1976D2;")

        self.detail_fs.setText("Frekuensi Sampling: - Hz")
        self.detail_peak.setText("Jumlah Peak: -")
        self.detail_interval.setText("Rata-rata Interval: - s")
        self.detail_exec_time.setText("Waktu Eksekusi: - s")
        self.system_memory_usage.setText("Memori Algoritma: - kb")
        if hasattr(self, 'log'):
            self.log("Plot berhasil dikosongkan.")

        

    def on_hover_peak(self, event):
        # simple tooltip near cursor using QToolTip
        if getattr(self, 'peaks', None) is None or getattr(self, 'signal', None) is None or event.xdata is None:
            return
        x = event.xdata
        peaks = np.asarray(self.peaks).astype(int)
        # find nearest
        times = peaks / max(1, self.fs)
        idx = (np.abs(times - x)).argmin()
        nearest_time = times[idx]
        nearest_amp = float(self.signal[peaks[idx]])
        # threshold 20ms
        if abs(nearest_time - x) < 0.02:
            QToolTip.showText(QCursor.pos(), f"t={nearest_time:.2f}s, Amp={nearest_amp:.3f}")
        else:
            QToolTip.hideText()