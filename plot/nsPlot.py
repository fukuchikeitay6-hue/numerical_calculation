import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import pyqtgraph as pg
import numpy as np
import sys
import os

class MainWindow(qtw.QMainWindow):
    def __init__(self, data: tuple[np.ndarray, np.ndarray, np.ndarray], Lx, Ly, num_frames: int, spf: float, parent=None):
        super().__init__(parent=parent)
        self.p = data[0]
        self.u = data[1]
        self.v = data[2]
        self.Lx = Lx
        self.Ly = Ly
        self.num_frames = num_frames
        self.spf = spf

        self.currentFrame = 0

        self.cmap = pg.colormap.get(pg.colormap.listMaps("matplotlib")[63], source="matplotlib")

        self.timer = qtc.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.onTimer)

        self.setWindowTitle("Navier-Stokes equation viewer")
        self.resize(800, 800)
        self.createWidget()

        self.updateDisplay()

    def createWidget(self):
        cw = qtw.QWidget(self)
        self.setCentralWidget(cw)
        mainLayout = qtw.QVBoxLayout(cw)

        self.label = qtw.QLabel(f"{self.currentFrame} / {self.num_frames}: {(self.spf * self.currentFrame):.4f} s")
        mainLayout.addWidget(self.label)

        # グラフ
        graph = pg.PlotWidget()
        graph.setAspectLocked(True)
        mainLayout.addWidget(graph)

        self.img = pg.ImageItem()
        self.img.setRect(qtc.QRectF(0, 0, self.Lx, self.Ly))
        graph.addItem(self.img)

        self.colorBar = pg.ColorBarItem(colorMap=self.cmap, interactive=True)
        self.colorBar.setImageItem(self.img, insert_in=graph.getPlotItem())

        ui_layout = qtw.QHBoxLayout()
        mainLayout.addLayout(ui_layout)

        play_btn = qtw.QPushButton("play / pause")
        play_btn.clicked.connect(self.togglePlay)
        ui_layout.addWidget(play_btn)

        self.slider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_frames - 1)
        self.slider.valueChanged.connect(self.onSliderChange)
        ui_layout.addWidget(self.slider)

    def setColorBarRange(self, min_val, max_val):
        pass

    def updateDisplay(self):
        self.img.setImage(self.p[self.currentFrame], autoLevels=False)
        self.slider.blockSignals(True)
        self.slider.setValue(self.currentFrame)
        self.slider.blockSignals(False)
        self.label.setText(f"{self.currentFrame} / {self.num_frames} : {(self.spf * self.currentFrame):.3f} s")

    def onSliderChange(self, val):
        self.currentFrame = val
        self.updateDisplay()

    def onTimer(self):
        self.currentFrame = (self.currentFrame + 1) % self.num_frames
        self.updateDisplay()

    def togglePlay(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()

data_dir = "/Users/fukuchikeita/Documents/programing/python/science/数値解析/data"
loader = np.load(os.path.join(data_dir, "simulation_results.npz"), allow_pickle=True)
p = loader["p_ani"]
params = loader["params"].item()

Lx, Ly = params["Lx"], params["Ly"]
T = params["t"]
dx, dy = params["dx"], params["dy"]
spf = params["spf"]

num_frames = len(p)

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow((p, np.array([]), np.array([])), Lx, Ly, num_frames, spf)
    window.show()
    sys.exit(app.exec())