import cv2
import time
import numpy as np

from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import *
import sys

from pathlib import Path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.UI.lib_UI import *
from src.UI.UI import Ui_MainWindow

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self, source):
        super().__init__()
        self.source = source
        # QThread.
    def run(self):
        cap = cv2.VideoCapture(self.source)
        while(True):
            ret, frame = cap.read()
            if type(self.source) != int:
                cv2.waitKey(20)
            if ret:
                self.change_pixmap_signal.emit(frame)

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Fire and smoke demo")
        self.setup_control()
        
    def setup_control(self):
        self.ui.video.clicked.connect(self.open_video)

    def open_video(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./", filter = "*.avi *.mp4 *.wmv *.mov")# start path
        self.ui.show_path.setText(filename)
        self.check_video_thread()
        self.video_thread = VideoThread(filename)
        self.video_thread.change_pixmap_signal.connect(self.update_frame)
        self.video_thread.change_pixmap_signal.connect(self.detect)
        self.video_thread.start()
        self.ui.show_video.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        qt_img = convert_cv_qt(cv_img)
        self.ui.show_video.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def detect(self, src_img):
        pass
        #predict result
        # print(time.time())

    def check_video_thread(self):
        try: 
            if self.video_thread.isRunning():
                self.video_thread.terminate()
            elif self.video_thread.isFinished():
                self.ui.show_video.clear()
                self.ui.show_video.setText('Wait for new videos...')
        except:pass



