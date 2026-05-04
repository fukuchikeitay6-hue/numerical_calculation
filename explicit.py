import mlx.core as mx
import numpy as np

# ----- パラメータ -----
L_x, L_y = 0.05, 0.003
T = 1.0
D = 1e-6
vx, vy = 0.1, 0.0
dx, dy = 1.0e-4, 1.0e-5
dt = 1.0e-6
sps = 5000  # steps per save
spf = dt * sps  # seconds per frame
x = mx.arange(0, L_x, dx, dtype=mx.float32)
y = mx.arange(0, L_y, dy, dtype=mx.float32)
t = mx.arange(0, T, dt, dtype=mx.float32)
nx = len(x); ny = len(y); nt = len(t)

# 安定性判定
con = (2*D/dx**2 + 2*D/dy**2 + abs(vx)/dx + abs(vy)/dy)**-1
if dt > con:
    raise ValueError(f"CFL条件を満たしません; dt: {dt:.3e}, con: {con:.3e}")

# 電極反応
r = mx.zeros((nx, ny), dtype=mx.float32)
r[10:20, 1] = 1e2

# 初期条件
c_old = mx.full((nx, ny), 1e-5, dtype=mx.float32)

# 保存用リスト
c_save = []
c_save.append(c_old)

@mx.compile
def step(c_old: mx.array, r: mx.array, dt: float, dx: float, dy: float, D: float, vx: float, vy: float, t: mx.array) -> mx.array:
    adv_x = vx * (c_old[1:-1, 1:-1] - c_old[0:-2, 1:-1]) / dx
    adv_y = vy * (c_old[1:-1, 1:-1] - c_old[1:-1, 0:-2]) / dy

    diff_x = D * (c_old[2:, 1:-1] - 2*c_old[1:-1, 1:-1] + c_old[0:-2, 1:-1]) / dx**2
    diff_y = D * (c_old[1:-1, 2:] - 2*c_old[1:-1, 1:-1] + c_old[1:-1, 0:-2]) / dy**2

    res = c_old[1:-1, 1:-1] + dt * ((diff_x + diff_y) - (adv_x + adv_y))

    c_new = mx.array(c_old)
    c_new[1:-1, 1:-1] = res


    # 境界条件
    c_new[0, :] = c_new[1, :]
    c_new[-1, :] = c_new[-2, :]
    c_new[:, 0] = c_new[:, 1]
    c_new[:, -1] = c_new[:, -2]

    # 電極反応（ソース項）を加算
    is_active = (t >= 0.1) & (t < 0.2)
    c_new = mx.where(is_active, c_new + r * dt, c_new)
    # c_new += r * dt
    return c_new

for n in range(1, nt):
    t_array = mx.array(n*dt)
    c_old = step(c_old, r, dt, dx, dy, D, vx, vy, t_array)

    if n % sps == 0:
        mx.eval(c_old)
        c_save.append(mx.array(c_old))
        print(f"{(n / nt * 100):.2f}%")

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
# mx.save("data.npy", c_save)
np.savez("data.npz", data=c_save_arr, params=params)
print("書き込み終了")