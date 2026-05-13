import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
import pyqtgraph as pg
import numpy as np
import sys
import os

def velocityOnPressureCells(u: np.ndarray, v: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    速度格子を圧力格子の中央に変換
    """
    nx, ny = p.shape
    u_center = 0.5 * (u[1:nx-1, 1:ny-1] + u[2:nx, 1:ny-1])
    v_center = 0.5 * (v[1:nx-1, 1:ny-1] + v[1:nx-1, 2:ny])
    return u_center, v_center

def createSturucturedArray(X:np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    速度行列から矢印をプロットするための構造化配列を返す
    """

    u_centered, v_centered = velocityOnPressureCells(u, v, p)
    shape = p.shape
    dt = [("x", float), ("y", float), ("vx", float), ("vy", float)]
    data = np.zeros(shape, dtype=dt)
    data["x"] = x
    data["y"] = Y
    data["vx"] = u_centered
    data["vy"] = v_centered
    return data

class MainWindow(qtw.QMainWindow):
    def __init__(self, p, u, v, X, Y, Lx, Ly, num_frames, spf, interval_ms: int, vector_stride:int=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Navier-Stokes equation viewer")
        self.resize(800, 800)

        self.p = p
        self.u = u
        self.v = v
        self.X_grid = X
        self.Y_grid = Y
        self.Lx = Lx
        self.Ly = Ly
        self.num_frames = num_frames
        self.spf = spf
        self.interval_ms = interval_ms
        self.vector_stride = max(1, vector_stride)

        self.currentFrame = 0

        self.cmap = pg.colormap.get(pg.colormap.listMaps("matplotlib")[63], source="matplotlib")

        self.timer = qtc.QTimer(self)
        self.timer.setInterval(self.interval_ms)
        self.timer.timeout.connect(self.next_frame)

        self.createWidget()
        self.createArrowItems()
        self.updateDisplay()
    
    def createWidget(self):
        cw = qtw.QWidget(self)
        self.setCentralWidget(cw)
        mainLayout = qtw.QVBoxLayout(cw)

        self.status_label = qtw.QLabel()
        mainLayout.addWidget(self.status_label)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        mainLayout.addWidget(self.plot)

        self.img = pg.ImageItem()
        self.img.setRect(qtc.QRectF(0, 0, self.Lx, self.Ly))
        self.img.setZValue(-10)
        self.plot.addItem(self.img)

        p_inner = self.p[:, 1:-1, 1:-1]
        self.levels = (float(np.nanmin(p_inner)), float(np.nanmax(p_inner)))
        self.img.setLevels(self.levels)

        self.colorBar = pg.ColorBarItem(values=self.levels, colorMap=self.cmap, interactive=False)
        self.colorBar.setImageItem(self.img, insert_in=self.plot.getPlotItem())

        # --- ベクトル描画 ---

        controls = qtw.QHBoxLayout()
        mainLayout.addLayout(controls)

        playBtn = qtw.QPushButton("play/stop")
        playBtn.clicked.connect(self.togglePlay)
        controls.addWidget(playBtn)

        self.slider = qtw.QSlider(qtc.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.num_frames -1)
        self.slider.valueChanged.connect(self.setFrame)
        controls.addWidget(self.slider)

    def togglePlay(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start()

    def updateDisplay(self):
        p = self.p[self.currentFrame]
        u = self.u[self.currentFrame]
        v = self.v[self.currentFrame]
        u_centerd, v_centered = velocityOnPressureCells(u, v, p)

        self.img.setImage(p[1:-1, 1:-1], autoLevels=False)
        self.updateArrow(u_centerd, v_centered)
        self.slider.blockSignals(True)
        self.slider.setValue(self.currentFrame)
        self.slider.blockSignals(False)
        self.status_label.setText(
            f"frame {self.currentFrame} / {self.num_frames - 1}, "
            f"t = {self.currentFrame * self.spf:.6g} s, "
        )

    def setFrame(self, val):
        self.currentFrame = int(val)
        self.updateDisplay()

    def next_frame(self):
        self.currentFrame = (self.currentFrame + 1) % self.num_frames
        self.updateDisplay()

    def createArrowItems(self):
        # 矢印をプロットする座標配列
        self.arrow_x:np.ndarray = self.X_grid[::self.vector_stride, ::self.vector_stride]
        self.arrow_y:np.ndarray = self.Y_grid[::self.vector_stride, ::self.vector_stride]
        self.arrows: list[pg.ArrowItem] = []  # ArrowItemを格納するリスト

        for x, y in zip(self.arrow_x.ravel(), self.arrow_y.ravel()):  # ravel(): 配列を平坦化
            # 全ての矢印を初期化
            arrow = pg.ArrowItem(
                pos = (float(x), float(y)),
                angle = 0,
                headLen = 12,
                tipAngle = 25,
                tailLen = 8,
                tailWidth = 2,
                brush = "w",
                pen = "w"
            )
            self.plot.addItem(arrow)
            self.arrows.append(arrow)

    def updateArrow(self, u_centered, v_centered):
        u_sample = u_centered[::self.vector_stride, ::self.vector_stride]
        v_sample = v_centered[::self.vector_stride, ::self.vector_stride]
        speed = np.hypot(u_sample, v_sample)  # hypot() = sqrt(u**2 + v**2)
        max_speed = float(np.nanmax(speed)) if speed.size else 0.0

        for arrow, x, y, u_val, v_val, speed_val in zip(
            self.arrows,
            self.arrow_x.ravel(), 
            self.arrow_y.ravel(), 
            u_sample.ravel(), 
            v_sample.ravel(), 
            speed.ravel()
        ):
            if not np.isfinite(speed_val) or max_speed == 0.0:
                arrow.setVisible(False)
                continue

            angle = np.degrees(np.arctan2(v_val, u_val))
            scale = 0.35 + 0.65 * speed_val / max_speed
            arrow.setVisible(True)
            arrow.setStyle(angle=float(angle), headLen=12*scale, tailLen=8*scale)
            arrow.setPos(float(x), float(y))

data_dir = "/Users/fukuchikeita/Documents/programing/python/science/数値解析/data"
loader = np.load(os.path.join(data_dir, "simulation_results.npz"), allow_pickle=True)
u = loader["u_ani"]
v = loader["v_ani"]
p = loader["p_ani"]
params = loader["params"].item()

Lx, Ly = params["Lx"], params["Ly"]
T = params["t"]
dx, dy = params["dx"], params["dy"]
spf = params["spf"]

x, y = np.arange(0, Lx, dx), np.arange(0, Ly, dy)
X, Y = np.meshgrid(x, y, indexing='ij')



num_frames = len(p)

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow(p, u, v, X, Y, Lx, Ly, num_frames, spf, 10, 10)
    window.show()
    sys.exit(app.exec())