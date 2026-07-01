import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import find_peaks, savgol_filter
from typing import List, Tuple, Dict

# 1.文件预处理
def preprocessing(data):
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S:%f', errors='coerce')
    data['Time_sec'] = (data['Time'] - data['Time'].min()).dt.total_seconds()
    data['Acc_Z_corrected'] = data['Acc_Z'] - 9.8
    data['Acc_mag'] = np.sqrt(data['Acc_X']**2 + data['Acc_Y']**2 + data['Acc_Z_corrected']**2)
    data['Acc_mag'] -= data['Acc_mag'].mean()
    return data


# 2.步态检测 & 振幅计算
def detect_amplitudes(acc_data, fs=50, peak_threshold_factor=1.0, window_size=10):
    mean_val = np.mean(acc_data)
    std_val = np.std(acc_data)
    peak_threshold = mean_val + peak_threshold_factor * std_val
    min_distance = int(0.4 * fs)

    peaks, _ = find_peaks(acc_data, height=peak_threshold, distance=min_distance)

    amplitudes = []
    for peak in peaks:
        start = max(peak - window_size, 0)
        end = min(peak + window_size, len(acc_data))
        peak_val = acc_data[peak]
        valley_before = np.min(acc_data[start:peak]) if peak > 0 else peak_val
        valley_after = np.min(acc_data[peak:end]) if peak < len(acc_data) - 1 else peak_val
        A_i = peak_val - np.mean([valley_before, valley_after])
        amplitudes.append(A_i)
    return amplitudes, peaks


# 7.寻找转弯方向和对应的步伐下标（保持你的参数不变）
def get_turn(data, peaks):
    ore_raw = data['Ore'].copy().reset_index(drop=True)

    # 平滑
    window_length = 25 if 25 % 2 == 1 else 26
    ore_smoothed = pd.Series(savgol_filter(ore_raw, window_length=window_length, polyorder=2))

    angle_diffs = ore_smoothed.diff().fillna(0).to_numpy()

    # 处理跨越180度跳变
    angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360, angle_diffs)
    angle_diffs = np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs)

    jump_threshold = 180
    angle_diffs = np.where(np.abs(angle_diffs) > jump_threshold, 0, angle_diffs)

    window_size = 200
    cumulative_diffs = pd.Series(angle_diffs).rolling(window=window_size, min_periods=1).sum().to_numpy()

    turn_threshold = 50
    min_interval = 400

    turn_info = []
    prev_idx = -min_interval

    for idx, cum_diff in enumerate(cumulative_diffs):
        if cum_diff > turn_threshold or cum_diff < -turn_threshold:
            if idx - prev_idx >= min_interval:
                direction = -1 if cum_diff > turn_threshold else 1
                peak_idx = np.argmin(np.abs(peaks - idx))
                turn_info.append([direction, peak_idx])
                prev_idx = idx
    return turn_info


def analyze_segment_turns(df, start_idx, end_idx, best_k=0.47):
    df_segment = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    df_segment = preprocessing(df_segment)

    acc_mag = df_segment["Acc_mag"].values
    amplitudes, peaks = detect_amplitudes(acc_mag)

    if len(amplitudes) < 2:
        return [(0, 0.0)]

    # 计算步长和累计距离
    step_lengths = [best_k * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
    cumulative_distances = np.cumsum([0] + step_lengths)

    if len(peaks) < 2:
        return [(0, np.sum(step_lengths))]

    # 获取转弯信息
    turn_info = get_turn(df_segment, peaks)
    turn_peak_indices = sorted(set([info[1] for info in turn_info]))
    turn_cum_dists = [cumulative_distances[tp] for tp in turn_peak_indices]

    # 构造段落
    segments = []
    prev_dist = 0

    for i, turn_dist in enumerate(turn_cum_dists):
        # 第 i 个转弯前的直行距离
        straight = turn_dist - prev_dist
        if straight > 0.1:
            segments.append((0, straight))

        # 转弯后直行的距离
        if i + 1 < len(turn_cum_dists):
            next_straight = turn_cum_dists[i + 1] - turn_dist
        else:
            next_straight = cumulative_distances[-1] - turn_dist

        if next_straight > 0.1:
            segments.append((1, next_straight))

        prev_dist = turn_dist + next_straight

    # 若最后一段（终点后）没有被包含进来，再补充
    if cumulative_distances[-1] - prev_dist > 0.1:
        segments.append((0, cumulative_distances[-1] - prev_dist))

    return segments


# 调用示例
# df = pd.read_csv("../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230706_2104.csv")
# segments = analyze_segment_turns(df, start_idx=3080, end_idx=3680)


from typing import List, Tuple, Dict, Optional
import copy

# 全局变量用于保存上一次窗口的超出方向和误差
last_overrun: Optional[Tuple[int, float]] = None  # 形如 (方向, 误差)


def compare_user_path_v2(user_segments: List[Tuple[int, float]],
                         path_segments: Dict[Tuple[int, int], List[Tuple[int, float]]],
                         threshold: float = 10.0
                         ) -> Tuple[bool, Dict[Tuple[int, int], List[Tuple[int, float]]],
                                    Optional[Tuple[int, int]], Optional[Tuple[int, float]]]:
    """
    判断用户是否偏离路径（不区分左右转），更新路径段。
    返回：
        - 是否偏离 (bool)
        - 更新后的路径段字典
        - 当前完成的路径段 key 或 None
        - 本窗口遗留的超出误差 (方向, 距离) 或 None
    """
    global last_overrun
    updated_path_segments = copy.deepcopy(path_segments)
    segment_keys = list(updated_path_segments.keys())
    if not segment_keys:
        return True, updated_path_segments, None, None

    segment_key = segment_keys[0]
    expected_segments = updated_path_segments[segment_key]
    user_idx = 0
    expected_idx = 0
    error_accumulator = 0.0

    # 衔接上一窗口的超出距离
    if last_overrun and user_segments:
        last_dir, last_dist = last_overrun
        user_dir, user_dist = user_segments[0]
        if user_dir == 0 and last_dir != 0:
            # 方向逻辑合理，累积
            combined = abs(last_dist) + user_dist
            if combined > threshold:
                last_overrun = None
                return False, path_segments, None, None
            else:
                user_segments[0] = (user_dir, combined)
                last_overrun = None
        else:
            # 方向不接续，不合并
            last_overrun = None

    while user_idx < len(user_segments) and expected_idx < len(expected_segments):
        user_type, user_dist = user_segments[user_idx]
        exp_type, exp_dist = expected_segments[expected_idx]

        # 不区分转向，只区分是否是转弯（action ≠ 0）
        if (user_type == 0) != (exp_type == 0):  # 一个转弯一个直行
            error_accumulator += user_dist
            if error_accumulator > threshold:
                return False, path_segments, None, None
            user_idx += 1
            continue

        # 匹配：直行对直行 or 转弯后直行对转弯后直行
        delta = exp_dist - user_dist
        expected_segments[expected_idx] = (exp_type, delta)
        if delta <= 0:
            # 本段完成，记录多走的部分
            last_overrun = (exp_type, -delta)
            expected_idx += 1
            error_accumulator = 0.0
        else:
            error_accumulator = 0.0

        user_idx += 1

    updated_path_segments[segment_key] = expected_segments[expected_idx:]
    is_finished = all(dist <= 0 for _, dist in updated_path_segments[segment_key])

    if error_accumulator > threshold:
        return False, path_segments, None, None

    if is_finished:
        last_overrun = None
        return True, updated_path_segments, segment_key, None
    else:
        return True, updated_path_segments, None, last_overrun


def update_path_and_nodes(user_segments: List[Tuple[int, float]],
                             path_segments: Dict[Tuple[int, int], List[Tuple[int, float]]],
                             path_nodes: List[int],
                             threshold: float = 10.0
                             ) -> Tuple[bool, Dict[Tuple[int, int], List[Tuple[int, float]]], List[int]]:
    """
    根据用户行为更新路径段和剩余节点。
    """
    success, updated_segments, finished_key, _ = compare_user_path_v2(user_segments, path_segments, threshold)

    if not success:
        return False, path_segments, path_nodes

    if finished_key:
        del updated_segments[finished_key]
        if path_nodes and path_nodes[0] == finished_key[0]:
            path_nodes = path_nodes[1:]

    return True, updated_segments, path_nodes
