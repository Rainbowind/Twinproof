import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def smooth_ore(
        df,
        ore_col='Ore',
        fs=50,
        max_step_deg=20.0,
        smooth_window_sec=1.5,
        polyorder=3
):
    """
    对航向角 Ore 进行“超强平滑”，适合作为 PDR 的航向输入（相对还原用）。

    参数：
    - df: 包含 Ore 列的 DataFrame
    - ore_col: 航向列名，默认 'Ore'
    - fs: 采样频率 Hz（影响平滑窗口长度）
    - max_step_deg: 相邻两采样点最大允许角度变化（剪掉离谱跳变）
    - smooth_window_sec: Savitzky-Golay 平滑窗口长度（秒），越大越平滑
    - polyorder: Savitzky-Golay 多项式阶数，一般 2 就够

    返回：
    - pandas.Series：和 df 同长度的平滑后航向角（单位：度，范围 [-180, 180)）
    """

    angles = df[ore_col].astype(float).to_numpy()

    # 1) 先把角度约到 [-180, 180)
    angles = (angles + 180.0) % 360.0 - 180.0

    n = len(angles)
    if n == 0:
        return pd.Series([], index=df.index, name=ore_col + '_smooth')

    # 2) 做一个“解跳变 + 最大步长限制”的一阶处理
    cleaned = np.zeros_like(angles)
    cleaned[0] = angles[0]

    for i in range(1, n):
        raw_diff = angles[i] - angles[i - 1]

        # 处理跨 ±180° 的跳变
        if raw_diff > 180.0:
            raw_diff -= 360.0
        elif raw_diff < -180.0:
            raw_diff += 360.0

        # 限制每一步最大变化量，剪掉离谱抖动
        if abs(raw_diff) > max_step_deg:
            raw_diff = np.sign(raw_diff) * max_step_deg

        cleaned[i] = cleaned[i - 1] + raw_diff

    # 3) 对 cleaned 做一次强平滑（Savitzky-Golay）
    #    窗口长度 = smooth_window_sec * fs，必须是奇数
    win = int(smooth_window_sec * fs)
    if win < (polyorder + 2):
        win = polyorder + 2
    if win % 2 == 0:
        win += 1

    if n >= win:
        smoothed = savgol_filter(cleaned, window_length=win, polyorder=polyorder)
    else:
        # 数据太短，退化为简单移动平均
        smoothed = pd.Series(cleaned).rolling(window=min(5, n), min_periods=1, center=True).mean().to_numpy()

    # 4) 把结果重新约到 [-180, 180)
    smoothed = (smoothed + 180.0) % 360.0 - 180.0

    return pd.Series(smoothed, index=df.index, name=ore_col + '_smooth')
