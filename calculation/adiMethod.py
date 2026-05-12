import mlx.core as mx
import numpy as np

# パラメータ
L_x, L_y = 0.01, 3e-4
# L_x, L_y = 0.05, 0.05
T = 0.5
# T = 50
D = 1e-8
vx, vy = 0.022, 0.0
# vx, vy = 0.0, 0.0
dx, dy = L_x*1e-3, L_y*1e-2
# dx, dy = L_x * 0.2e-3, L_y*1e-1
dt = T * 1e-4
# dt = 1e-2
sps = 100        # steps per save
spf = dt * sps  # seconds per frame

rx = D * dt / (2 * dx**2)
ry = D * dt / (2 * dy**2)

x = mx.arange(0, L_x, dx, dtype=mx.float32)
y = mx.arange(0, L_y, dy, dtype=mx.float32)
t = mx.arange(0, T, dt, dtype=mx.float32)
nx = len(x); ny = len(y); nt = len(t)

r = mx.zeros((nx, ny), dtype=mx.float32)
r[10:200, 1] = 1e3
# r[1, :] = 1e1

Ax = np.diag(np.full(nx-2, 1+2*rx), k=0) + np.diag(np.full(nx-3, -rx), k=1) + np.diag(np.full(nx-3, -rx), k=-1)
Ax_inv = mx.array(np.linalg.inv(Ax), dtype=mx.float32)

Ay = np.diag(np.full(ny-2, 1+2*ry), k=0) + np.diag(np.full(ny-3, -ry), k=1) + np.diag(np.full(ny-3, -ry), k=-1)
Ay_inv = mx.array(np.linalg.inv(Ay), dtype=mx.float32)

# 初期条件
c_old = mx.full((nx, ny), 1e-5, dtype=mx.float32)

# 保存用リスト
c_save = []
c_save.append(c_old)

@mx.compile
def step(c_old: mx.array, r: mx.array, dt: float, dx: float, dy: float, D: float, vx: float, vy: float, t: mx.array) -> mx.array:
    # --- 移流項・反応項 ---
    c_adv = mx.zeros_like(c_old)
    c_adv_x = vx * (c_old[1:-1, 1:-1] - c_old[0:-2, 1:-1]) / dx
    c_adv_y = vy * (c_old[1:-1, 1:-1] - c_old[1:-1, 0:-2]) / dy
    c_adv[1:-1, 1:-1] = c_old[1:-1, 1:-1] - dt * (c_adv_x + c_adv_y) + r[1:-1, 1:-1] * dt

    # 境界条件
    c_adv[0, :] = c_adv[1, :]
    c_adv[-1, :] = c_adv[-2, :]
    c_adv[:, 0] = c_adv[:, 1]
    c_adv[:, -1] = c_adv[:, -2]

    # --- 拡散項(x方向) ---
    c_half = mx.zeros_like(c_adv)
    rhs_x = c_adv[1:-1, 1:-1] + ry * (c_adv[1:-1, 2:] - 2 * c_adv[1:-1, 1:-1] + c_adv[1:-1, :-2])
    # 三重対角行列を一気に解く
    c_half[1:-1, 1:-1] = Ax_inv @ rhs_x

    c_half[0, :] = c_half[1, :]
    c_half[-1, :] = c_half[-2, :]
    c_half[:, 0] = c_half[:, 1]
    c_half[:, -1] = c_half[:, -2]

    # --- 拡散項(y方向) ---
    c_new = mx.zeros_like(c_half)
    rhs_y = c_half[1:-1, 1:-1] + rx * (c_half[2:, 1:-1] - 2 * c_half[1:-1, 1:-1] + c_half[:-2, 1:-1])
    # 三重対角行列を一気に解く(右辺の向きに注意)
    c_new[1:-1, 1:-1] = rhs_y @ Ay_inv.T

    c_new[0, :] = c_new[1, :]
    c_new[-1, :] = c_new[-2, :]
    c_new[:, 0] = c_new[:, 1]
    c_new[:, -1] = c_new[:, -2]

    return c_new

for n in range(1, nt):
    t_array = mx.array(n*dt)
    c_old = step(c_old, r, dt, dx, dy, D, vx, vy, t_array)

    if n % sps == 0:
        mx.eval(c_old)
        c_save.append(mx.array(c_old))
        print("\r",f"{(n / nt * 100):.2f}%", end="")

c_save = mx.stack(c_save, axis=0)
mx.eval(c_save)
c_save_arr = np.array(c_save)
params = {
    "D": D,
    "Lx": L_x,
    "Ly": L_y,
    "vx": vx,
    "vy": vy,
    "t": T,
    "dx": dx,
    "dy": dy,
    "dt": dt,
    "spf": spf
}
print("ファイルに書き込み中...")
mx.save("data.npy", c_save)
np.savez("data.npz", data=c_save_arr, params=params)
print("書き込み終了")