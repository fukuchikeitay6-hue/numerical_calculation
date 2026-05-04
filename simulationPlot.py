import sys
import numpy as np
import PyQt6.QtWidgets as qtw
import PyQt6.QtCore as qtc
from mainwindow import MainWindow

# 1. データの読み込み
loader = np.load("data.npz", allow_pickle=True)
data = loader["data"]
params = loader["params"].item()
for k in params.keys():
        print(f"{k} : {params[k]}")

# パラメータ
L_x, L_y = params["Lx"], params["Ly"]
T = params["t"]
dx, dy = params["dx"], params["dy"]
spf = params["spf"]

num_frames = len(data)

# --- 対数表示(LogNorm)の事前準備 ---
safe_data = np.clip(data, 1e-5, 1e-1)
log_data = np.log10(safe_data)



if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = MainWindow(log_data, L_x, L_y, num_frames, spf)
    window.img.setRect(qtc.QRectF(0, 0, L_x, L_y))
    window.show()
    sys.exit(app.exec())