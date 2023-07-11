from PyQt6 import QtWidgets
from src.UI.control_UI import MainWindow_controller

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller()
    window.show()
    sys.exit(app.exec())