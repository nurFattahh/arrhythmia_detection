import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import ECGApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ECGApp()
    win.showMaximized()
    sys.exit(app.exec_())