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
    data["x"] = X
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

        self.createWidgets()
        self.updateDisplay()
    
    def createWidgets(self):
        # central widget
        cw = qtw.QWidget(self)
        self.setCentralWidget(cw)
        mainLayout = qtw.QVBoxLayout(cw)

        # status label: 現在のフレーム表示など
        self.status_label = qtw.QLabel()
        mainLayout.addWidget(self.status_label)

        # グラフ
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        mainLayout.addWidget(self.plot)

        self.createGraphItems()

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

    def createGraphItems(self):
        # --- 圧力 ---
        self.img = pg.ImageItem()
        self.img.setRect(qtc.QRectF(0, 0, self.Lx*0.01, self.Ly*0.01))  # 空間サイズ
        self.img.setZValue(-10)  # 圧力プロットを最背面に
        self.plot.addItem(self.img)

        p_inner = self.p[:, 1:-1, 1:-1]  # ゴーストセルを除く圧力配列
        self.levels = (float(np.nanmin(p_inner)), float(np.nanmax(p_inner)))  # 圧力の最大値と最小値を取得 -> カラーバーの範囲設定
        self.img.setLevels(self.levels)

        self.colorBar = pg.ColorBarItem(values=self.levels, colorMap=self.cmap, interactive=False)
        self.colorBar.setImageItem(self.img, insert_in=self.plot.getPlotItem())

        # --- 速度ベクトル ---
        self.vectorItem = pg.PlotDataItem(pen=pg.mkPen("w", width=1))
        self.plot.addItem(self.vectorItem)

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

    def updateArrow(self, u_centered, v_centered):
        # 矢印をプロットする座標
        x = self.X_grid[1:-1, 1:-1][::self.vector_stride, ::self.vector_stride]
        y = self.Y_grid[1:-1, 1:-1][::self.vector_stride, ::self.vector_stride]
        # プロットする速度成分
        u = u_centered[::self.vector_stride, ::self.vector_stride]
        v = v_centered[::self.vector_stride, ::self.vector_stride]

        speed = np.hypot(u, v)  # hypot(u, v) = sqrt(u**2 + v**2)
        maxSpeed = np.max(speed)
        if not np.isfinite(maxSpeed) or maxSpeed == 0.0:
            self.vectorItem.setData([], [])
            return
        
        scale = 0.05 * max(self.Lx, self.Ly) / maxSpeed  # 最大速度が短辺の0.5倍の長さとなるように変換

        # 矢印の始点
        x0 = x
        y0 = y
        # 矢印の終点
        x1 = x + u*scale
        y1 = y + v*scale

        # 単位ベクトル
        ux = u / speed
        uy = v / speed

        valid = np.isfinite(speed) & (speed > 0)
        ux = np.where(valid, ux, 0)
        uy = np.where(valid, uy, 0)

        # 矢尻サイズ
        arrow_len = 0.25 * scale * maxSpeed
        arrow_width = 0.12 * scale * maxSpeed

        # 進行方向と逆向きに戻り、法線方向へ左右に開く
        left_x = x1 - arrow_len * ux - arrow_width * (-uy)
        left_y = y1 - arrow_len * uy - arrow_width * ux

        right_x = x1 - arrow_len * ux + arrow_width * (-uy)
        right_y = y1 - arrow_len * uy + arrow_width * ux

        nan = np.full(x.size, np.nan)

        xs = np.column_stack([
            x0.ravel(), x1.ravel(), nan,          # 胴体
            x1.ravel(), left_x.ravel(), nan,      # 左矢尻
            x1.ravel(), right_x.ravel(), nan,     # 右矢尻
        ]).ravel()

        ys = np.column_stack([
            y0.ravel(), y1.ravel(), nan,
            y1.ravel(), left_y.ravel(), nan,
            y1.ravel(), right_y.ravel(), nan,
        ]).ravel()

        self.vectorItem.setData(xs, ys)

data_dir = "/Users/fukuchikeita/Documents/programing/python/science/数値解析/data"
loader = np.load(os.path.join(data_dir, "simulation_results.npz"), allow_pickle=True)
u = loader["u_ani"]
v = loader["v_ani"]
p = loader["p_ani"]
params = loader["params"].item()

Lx, Ly = params["Lx"], params["Ly"]
print(Lx, Ly)
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