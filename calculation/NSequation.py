import numpy as np
import mlx.core as mx

@mx.compile
def calc_intermediate_velocity(u_0: mx.array, v_0: mx.array, dx: float, dy:float, dt:float, nu:float) -> tuple[mx.array, mx.array]:
    """
    圧力項を無視して速度ベクトルを計算
    """
    # ===== x方向 =====
    u_star = mx.array(u_0)
    
    nx, ny = u_0.shape

    # u_0の各方向へのずれを取り出す
    u_c = u_0[1:-1, 1:-1]  # u_0[i, j]
    u_l = u_0[0:-2, 1:-1]  # u_0[i-1, j]
    u_r = u_0[2:, 1:-1]    # u_0[i+1, j]
    u_b = u_0[1:-1, 0:-2]  # u_0[i, j-1]
    u_u = u_0[1:-1, 2:]    # u_0[i, j+1]

    # v_0の周囲4点を取り出す
    v_00 = v_0[0:nx-2, 1:ny-1]  # u_0[i, j]の左下
    v_01 = v_0[1:nx-1, 1:ny-1]  # u_0[i, j]の右下
    v_10 = v_0[1:nx-1, 2:ny]  # u_0[i, j]の右上
    v_11 = v_0[0:nx-2, 2:ny]  # u_0[i, j]の左上

    # 4点平均を計算
    v_avg = 0.25 * (v_00 + v_01 + v_10 + v_11)

    # 移流項・拡散項
    adv_u = u_c * (dt / dx) * (u_c - u_l) + v_avg * (dt / dy) * (u_c - u_b)
    diff_u = nu * ((dt / dx**2) * (u_r - 2*u_c + u_l) + (dt / dy**2) * (u_u - 2*u_c + u_b))
    u_star[1:-1, 1:-1] = u_c - adv_u + diff_u


    # ===== y方向 =====
    v_star = mx.array(v_0)
    nx_v, ny_v = v_0.shape

    # v_0 のスライス
    v_c = v_0[1:-1, 1:-1]  # v_0[i, j]
    v_l = v_0[:-2, 1:-1]   # v_0[i-1, j]
    v_r = v_0[2:, 1:-1]    # v_0[i+1, j]
    v_d = v_0[1:-1, :-2]   # v_0[i, j-1]
    v_u = v_0[1:-1, 2:]    # v_0[i, j+1]

    # u_0 のスライス (v_0 のループインデックスに合わせる)
    u_00 = u_0[1:nx_v-1, 0:ny_v-2]  # u_0[i, j-1]
    u_10 = u_0[2:nx_v,   0:ny_v-2]  # u_0[i+1, j-1]
    u_01 = u_0[1:nx_v-1, 1:ny_v-1]  # u_0[i, j]
    u_11 = u_0[2:nx_v,   1:ny_v-1]  # u_0[i+1, j]

    # 4点平均
    u_avg = 0.25 * (u_00 + u_10 + u_01 + u_11)

    # 移流項と拡散項
    adv_v = u_avg * (dt / dx) * (v_c - v_l) + v_c * (dt / dy) * (v_c - v_d)
    diff_v = nu * ((dt / dx**2) * (v_r - 2 * v_c + v_l) + (dt / dy**2) * (v_u - 2 * v_c + v_d))

    # 更新
    v_star[1:-1, 1:-1] = v_c - adv_v + diff_v

    # ===== 境界条件 =====
    # --- x方向 ---
    # 境界条件: 滑りなし壁
    # 左右の壁(垂直方向): 勾配0
    u_star[0, :] = u_star[1, :]
    u_star[-1, :] = u_star[-2, :]
    # 上下の壁(平行方向): 滑りなし
    u_star[:, 0] = -u_star[:, 1]
    u_star[:, -1] = -u_star[:, -2]

    # --- y方向 ---
    # 左右の壁(平行方向): 勾配0
    v_star[0, :] = v_star[1, :]
    v_star[-1, :] = v_star[-2, :]
    # 上下の壁(垂直方向): 流入流出0
    v_star[:, 0] = 0.0
    v_star[:, -1] = 0.0

    # 流入速度
    u_star[0, :] = 1.0
    v_star[0, :] = 0.0

    return u_star, v_star

def update_pressure(u_star: mx.array, v_star: mx.array, p_0: mx.array, dx: float, dy: float, dt: float, rho: float, max_iter:int=100, max_tor:float = 1e-5, omega: float = 1.5, check_interval: int = 10) -> mx.array:
    """
    ポアソン方程式をSOR法で解く

    Parameters
    ----------
    omega : float
        SOR緩和係数 (1 < omega < 2)。大きいほど収束が速いが不安定になる場合がある。
        デフォルト 1.5。
    check_interval : int
        収束判定を行う反復間隔。ホスト同期の頻度を下げて計算効率を向上させる。
        デフォルト 10。
    """
    S = mx.zeros_like(p_0)

    nx, ny = p_0.shape
    S[1:-1, 1:-1] = rho / dt * ((u_star[2:nx, 1:ny-1] - u_star[1:nx-1, 1:ny-1]) / dx + (v_star[1:nx-1, 2:ny] - v_star[1:nx-1, 1:ny-1]) / dy)

    p = mx.array(p_0)
    denom = 2 * (dx**2 + dy**2)
    for i in range(max_iter):
        p_old = mx.array(p)

        # Jacobi更新量をSORで混合
        p_jacobi = (dx**2 * (p[1:-1, 2:] + p[1:-1, 0:-2]) + dy**2 * (p[2:, 1:-1] + p[0:-2, 1:-1]) - (dx*dy)**2 * S[1:-1, 1:-1]) / denom
        p[1:-1, 1:-1] = (1.0 - omega) * p[1:-1, 1:-1] + omega * p_jacobi

        # 境界条件の適用(壁での圧力勾配がゼロ)
        # 左右の壁
        p[0, :]  = p[1, :]   # 左壁
        p[-1, :] = 0          # 右壁: 流出 -> 外部に開放 -> p = 0(基準圧力)
        # 上下の壁
        p[:, 0]  = p[:, 1]   # 下壁
        p[:, -1] = p[:, -2]  # 上壁

        # check_interval ごとに収束判定 (ホスト同期の頻度を低減)
        if (i + 1) % check_interval == 0 or i == max_iter - 1:
            tor = mx.max(mx.abs(p_old - p)).item()  # ホスト同期
            if tor <= max_tor:
                return p
            if i == max_iter - 1:
                print("収束しませんでした")

    return p

@mx.compile
def calc_final_velocity(u_star: mx.array, v_star: mx.array, p_new: mx.array, dx: float, dy: float, dt: float, rho: float) -> tuple[mx.array, mx.array]:
    """
    速度ベクトルを補正
    """
    u_new = mx.zeros_like(u_star)
    v_new = mx.zeros_like(v_star)

    # ---------- x方向 -----------
    nx, ny = u_star.shape
    u_new[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt / rho * (p_new[1:nx-1, 1:ny-1] - p_new[0:nx-2, 1:ny-1]) / dx

    # ---------- y方向 ----------
    nx, ny = v_star.shape
    v_new[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt / rho * (p_new[1:nx-1, 1:ny-1] - p_new[1:nx-1, 0:ny-2]) / dy

    # ----- 境界条件 -----
    # --- 左右の壁 ---
    u_new[0, :]  = u_new[1, :]   # 勾配0
    u_new[-1, :] = u_new[-2, :]  # 勾配0
    v_new[0, :]  = v_new[1, :]   # 勾配0
    v_new[-1, :] = v_new[-2, :]  # 勾配0

    # --- 上下の壁 ---
    v_new[:, 0]  = 0  # 流入流出0
    v_new[:, -1] = 0  # 流入流出0
    u_new[:, 0]  = -u_new[:, 1]   # 滑りなし
    u_new[:, -1] = -u_new[:, -2]  # 滑りなし

    # 流入速度
    u_new[0, :] = 1.0
    v_new[0, :] = 0.0

    return u_new, v_new

def step_mlx(
        u_0: mx.array, 
        v_0: mx.array, 
        p_0: mx.array, 
        dx: float, 
        dy: float, 
        dt: float, 
        nu: float, 
        rho: float, 
        max_iter: int=100, 
        max_tor: float=1e-5,
        omega: float=1.5,
    ) -> tuple[mx.array, mx.array, mx.array]:
    # ====================
    #      ステップ1
    # ====================
    # ---------- x方向 ----------
    u_star, v_star = calc_intermediate_velocity(u_0, v_0, dx, dy, dt, nu)

    # ====================
    #      ステップ2
    # ====================
    p_new = update_pressure(u_star, v_star, p_0, dx, dy, dt, rho, max_iter, max_tor, omega)

    # ====================
    #      ステップ3
    # ====================
    u_new, v_new = calc_final_velocity(u_star, v_star, p_new, dx, dy, dt, rho)

    return u_new, v_new, p_new

def main() -> None:
    # --- 初期設定 ---
    Lx,Ly = 1e-2, 3e-4
    dx, dy = Lx*1e-2, Ly*1e-2
    x, y = mx.arange(0, Lx, dx), mx.arange(0, Ly, dy)
    Nx, Ny = len(x), len(y)
    T = 0.001
    dt = 0.000001
    t = mx.arange(0, T, dt)
    nt = len(t)
    nu = 1e-6
    rho = 1.0
    sps = 5  # steps per save
    spf = dt * sps # seconds per frame

    # 配列の初期化 (前回のA案：数学的座標系に合わせて i=x, j=y)
    u = mx.zeros((Nx+3, Ny+2))
    v = mx.zeros((Nx+2, Ny+3))
    p = mx.zeros((Nx+2, Ny+2))

    u[0, 25] = 1.0 

    p_ani = []
    u_ani = []
    v_ani = []
    # --- 時間発展ループ ---
    for n in range(nt):
        u, v, p = step_mlx(u, v, p, dx, dy, dt, nu, rho, max_iter=10000, max_tor=1e-4)
        mx.eval(u, v, p)
        if n % sps == 0:
            print(f"Step {n} completed.")
            p_ani.append(p)
            u_ani.append(u)
            v_ani.append(v)

    params = {
    "Lx": Lx,
    "Ly": Ly,
    "t": T,
    "dx": dx,
    "dy": dy,
    "dt": dt,
    "spf": spf,
    "nu": nu,
    "rho": rho
    }

    # --- 結果を保存 ---
    np.savez("simulation_results.npz", p_ani=p_ani, u_ani=u_ani, v_ani=v_ani, params=params)
    print("シミュレーション結果をsimulation_results.npzに保存しました。")

if __name__ == "__main__":
    main()
