import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import pyqtgraph as pg
import numpy as np

class MainWindow(qtw.QMainWindow):
    def __init__(self, data: np.ndarray, Lx: float, Ly: float, num_frames: int, spf: float,  parent=None):
        super().__init__(parent=parent)
        self.data = data
        self.Lx = Lx
        self.Ly = Ly
        self.num_frames = num_frames
        self.spf = spf

        self.currentFrame = 0

        self.cmap = pg.colormap.get(pg.colormap.listMaps("matplotlib")[63], source='matplotlib')

        self.timer = qtc.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.onTimer)

        self.setWindowTitle("Simulation viewer")
        self.resize(800, 800)
        self.createWidget()

        self.updateDisplay()

    def createWidget(self):
        cw = qtw.QWidget(self)
        self.setCentralWidget(cw)
        mainLayout = qtw.QVBoxLayout(cw)

        self.label = qtw.QLabel(f"{self.currentFrame} / {self.num_frames} : {(self.spf * self.currentFrame):.3f} s")
        mainLayout.addWidget(self.label)

        # グラフ
        graph = pg.PlotWidget()
        graph.setAspectLocked(True)
        mainLayout.addWidget(graph)
        
        self.img = pg.ImageItem()
        self.img.setLevels((-5, -1))
        self.img.setRect(qtc.QRectF(0, 0, self.Lx, self.Ly))
        graph.addItem(self.img)

        # カラーバーの設定
        self.colorBar = pg.ColorBarItem(values=(-5, -1), colorMap=self.cmap, interactive=False)
        self.colorBar.setImageItem(self.img, insert_in=graph.getPlotItem())

        # コントロールUI
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

    def setColorbarRange(self, log_min_val, log_max_val):
        self.img.setLevels((log_min_val, log_max_val))
        self.colorBar.setLevels((log_min_val, log_max_val))

    def updateDisplay(self):
        self.img.setImage(self.data[self.currentFrame], autoLevels=False)
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
