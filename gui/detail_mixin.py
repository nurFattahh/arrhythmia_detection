from PyQt5.QtWidgets import QMessageBox
import serial.tools.list_ports

class DetailMixin:
    def update_bpm_and_status(self, avg_bpm, status):
        try:
            self.bpm_label.setText(f"{int(avg_bpm)}")
        except Exception:
            self.bpm_label.setText("0")
        if avg_bpm > 100 or avg_bpm < 60:
            self.bpm_label.setStyleSheet("font-size: 50px; font-weight: bold; color: red;")
        else:
            self.bpm_label.setStyleSheet("font-size: 50px; font-weight: bold; color: green;")

        if isinstance(status, str) and status.lower() == "normal":
            self.status_label.setText(status)
            self.status_label.setStyleSheet("font-size: 50px; font-weight: bold; color: green;")
        else:
            self.status_label.setText(status)
            self.status_label.setStyleSheet("font-size: 50px; font-weight: bold; color: red;")

    def refresh_com_ports(self):
        try:
            ports = serial.tools.list_ports.comports()
            self.com_dropdown.clear()
            if ports:
                for port in ports:
                    self.com_dropdown.addItem(port.device)
            else:
                self.com_dropdown.addItem("Tidak ada COM port terdeteksi")

            QMessageBox.information(self, "Refresh COM Ports", "Daftar COM port telah diperbarui.")
        except Exception as e:
            self.com_dropdown.clear()
            self.com_dropdown.addItem(f"Error: {str(e)}")

    def toggle_input_mode(self, mode):
        is_dataset = mode == "Dataset"
        self.input_label.setText("Pilih Dataset:" if is_dataset else "Pilih COM Port:")
        # self.mode_label.setText("" if is_dataset else "Mode Input:")
        self.dataset_combo.setVisible(is_dataset)
        self.record_input.setVisible(is_dataset)
        self.load_btn.setVisible(is_dataset)
        self.browse_btn.setVisible(is_dataset)
        self.com_dropdown.setVisible(not is_dataset)
        self.refresh_com_btn.setVisible(not is_dataset)
        self.start_btn.setVisible(not is_dataset)
        self.stop_btn.setVisible(not is_dataset)
        self.mode_label.setVisible(not is_dataset)
        self.system_exec_time.setVisible(not is_dataset)
        
        if not is_dataset:
            self.refresh_com_ports()
            self.canvas.hide()
            self.pg_plot.show()
            self.browse_btn.hide()
        else:
            self.canvas.show()
            self.pg_plot.hide()
            