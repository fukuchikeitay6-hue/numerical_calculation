# Numerical Calculation

このリポジトリでは、2 次元の移流拡散方程式を数値計算するために、陽解法と ADI 法を実装している。

対象にしている代表的な方程式は次の形である。

$$
\frac{\partial c}{\partial t}
= D \left(
\frac{\partial^2 c}{\partial x^2}
+ \frac{\partial^2 c}{\partial y^2}
\right)
- v_x \frac{\partial c}{\partial x}
- v_y \frac{\partial c}{\partial y}
+ r
$$

ここで、`c` は濃度、`D` は拡散係数、`vx`, `vy` は流速、`r` は反応または生成項を表す。

## ファイル構成

- `explicit.py`: 陽解法による時間発展
- `adiMethod.py`: ADI 法による時間発展
- `simulationPlot.py`: 保存した計算結果の可視化用スクリプト
- `data.npz`, `data.npy`: 計算結果の保存先

## 陽解法

陽解法は、現在時刻の値 `c_old` だけを使って次の時刻の値 `c_new` を直接計算する方法である。

この実装では、時間方向に前進オイラー法を使い、空間方向には差分近似を使っている。

$$
c^{n+1}_{i,j}
= c^n_{i,j}
+ \Delta t \left(
\mathrm{diff}_x + \mathrm{diff}_y
- \mathrm{adv}_x - \mathrm{adv}_y
\right)
$$

拡散項は 2 階中心差分で近似している。

$$
\mathrm{diff}_x
= D \frac{
c^n_{i+1,j} - 2c^n_{i,j} + c^n_{i-1,j}
}{\Delta x^2}
$$

$$
\mathrm{diff}_y
= D \frac{
c^n_{i,j+1} - 2c^n_{i,j} + c^n_{i,j-1}
}{\Delta y^2}
$$

移流項は上流差分に近い形で近似している。

$$
\mathrm{adv}_x
= v_x \frac{c^n_{i,j} - c^n_{i-1,j}}{\Delta x}
$$

$$
\mathrm{adv}_y
= v_y \frac{c^n_{i,j} - c^n_{i,j-1}}{\Delta y}
$$

### 特徴

- 実装が単純で、1 ステップの計算が軽い。
- 各格子点の更新を直接計算できるため、処理の流れが分かりやすい。
- 時間刻み `dt` を大きくすると不安定になりやすい。

`explicit.py` では、計算開始前に次の条件で安定性を確認している。

$$
\Delta t
\le
\frac{1}{
\frac{2D}{\Delta x^2}
+ \frac{2D}{\Delta y^2}
+ \frac{|v_x|}{\Delta x}
+ \frac{|v_y|}{\Delta y}
}
$$

この条件を満たさない場合は `ValueError` を出して停止する。

### 境界条件

境界では隣の内側の値をコピーしている。

$$
c_{0,j} = c_{1,j}, \quad
c_{N_x-1,j} = c_{N_x-2,j}
$$

$$
c_{i,0} = c_{i,1}, \quad
c_{i,N_y-1} = c_{i,N_y-2}
$$

これは濃度勾配を 0 とみなす Neumann 境界条件に相当する。

### 反応項

`explicit.py` では、電極反応を表す生成項 `r` を設定している。

$$
r_{i,1} = 10^2 \quad (10 \le i < 20)
$$

また、反応は `0.1 <= t < 0.2` の範囲だけ有効になる。

## ADI 法

ADI 法は Alternating Direction Implicit method の略で、2 次元の拡散計算を x 方向と y 方向に分けて陰的に解く方法である。

`adiMethod.py` では、1 ステップを大きく次の 3 段階に分けている。

1. 移流項と反応項を陽的に更新する。
2. x 方向の拡散を陰的に解き、中間値 `c_half` を求める。
3. y 方向の拡散を陰的に解き、次時刻の値 `c_new` を求める。

ADI 法では、2 次元問題を一度に巨大な連立方程式として解く代わりに、各方向の 1 次元的な連立方程式に分けて解く。

この実装では、x 方向と y 方向の三重対角行列を作成し、その逆行列をあらかじめ計算している。

$$
r_x = \frac{D \Delta t}{2\Delta x^2},
\quad
r_y = \frac{D \Delta t}{2\Delta y^2}
$$

x 方向の更新では `Ax_inv @ rhs_x` を使い、y 方向の更新では `rhs_y @ Ay_inv.T` を使っている。

### 特徴

- 拡散項を陰的に扱うため、陽解法より大きな `dt` を使いやすい。
- 2 次元拡散を x 方向と y 方向に分割するため、完全な 2 次元陰解法より扱いやすい。
- 行列を解く処理が入るため、陽解法より実装は複雑になる。
- 現在の実装では逆行列を明示的に作っているため、格子数が大きい場合は計算量とメモリ使用量が増える。

### 境界条件

ADI 法でも陽解法と同じく、境界では隣の内側の値をコピーしている。

$$
c_{0,j} = c_{1,j}, \quad
c_{N_x-1,j} = c_{N_x-2,j}
$$

$$
c_{i,0} = c_{i,1}, \quad
c_{i,N_y-1} = c_{i,N_y-2}
$$

### 反応項

`adiMethod.py` では、左側付近の格子に生成項を与えている。

$$
r_{1,j} = 10^1
$$

この反応項は、移流項を陽的に更新する段階で加算される。

## 陽解法と ADI 法の比較

| 項目 | 陽解法 | ADI 法 |
| --- | --- | --- |
| 実装 | 単純 | やや複雑 |
| 1 ステップの計算 | 軽い | 行列計算が必要 |
| 安定性 | `dt` の制限が強い | 拡散項に対して比較的安定 |
| 大きな時間刻み | 使いにくい | 使いやすい |
| 向いている用途 | 小規模・確認用 | 拡散支配の長時間計算 |

## 実行方法

仮想環境を有効化したあと、次のように実行する。

```bash
python explicit.py
```

または

```bash
python adiMethod.py
```

計算結果は `data.npz` に保存される。`adiMethod.py` では `data.npy` にも保存している。

`data.npz` には、濃度データ `data` と、計算条件 `params` が含まれる。

## 注意点

- `explicit.py` は CFL 条件を満たすように `dt`, `dx`, `dy`, `D`, `vx`, `vy` を設定する必要がある。
- `adiMethod.py` は拡散項に対して安定だが、移流項と反応項は陽的に扱っているため、極端に大きな `dt` では精度や安定性に注意が必要である。
- `data.npz` と `data.npy` は大きくなりやすいため、Git では追跡しない設定にしている。
