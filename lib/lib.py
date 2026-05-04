import numpy as np
import mlx.core as mx

import mlx.core as mx

def solveTM(a: mx.array, b: mx.array, c: mx.array, d: mx.array) -> mx.array:
    """
    a: 下対角ベクトル(len: N-1)
    b: 主対角ベクトル(len: N)
    c: 上対角ベクトル(len: N-1)
    d: 右辺ベクトル(len: N)
    """
    N = len(d)
    
    # 元の配列を破壊しないように、関数内でコピーを作成する
    b_tmp = mx.array(b)
    d_tmp = mx.array(d)

    # 前進消去 (Forward elimination)
    for i in range(1, N):
        m = a[i-1] / b_tmp[i-1]
        b_tmp[i] = b_tmp[i] - m * c[i-1]
        d_tmp[i] = d_tmp[i] - m * d_tmp[i-1]

    x = mx.zeros(N, dtype=mx.float32)
    x[-1] = d_tmp[-1] / b_tmp[-1]
    
    # 後退代入 (Backward substitution)
    for i in range(N-2, -1, -1):
        x[i] = (d_tmp[i] - c[i] * x[i+1]) / b_tmp[i]

    return x