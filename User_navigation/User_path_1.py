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


# 3.寻找转弯方向和对应的步伐下标（
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


# 4.构造路径段落
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
        # 转弯前的直行
        straight = turn_dist - prev_dist
        if straight > 0.1:
            segments.append((0, straight))

        # 转弯后的直行
        if i + 1 < len(turn_cum_dists):
            next_straight = turn_cum_dists[i + 1] - turn_dist
        else:
            next_straight = cumulative_distances[-1] - turn_dist

        if next_straight > 0.1:
            # 这里仍标记为 1，表示这是转弯之后的一段
            segments.append((1, next_straight))

        prev_dist = turn_dist + next_straight

    # 最后一段补充
    if cumulative_distances[-1] - prev_dist > 0.1:
        segments.append((0, cumulative_distances[-1] - prev_dist))

    # ✅ 保证第一段是直行
    if segments and segments[0][0] != 0:
        segments.insert(0, (0, 0.0))

    return segments


# 5.检测用户是否偏离路径
def compare_path_with_trajectory(path_dict, user_segments, current_path, error_threshold=15):
    """
    比较路径字典与用户轨迹；支持“末段+下一段”合并，并在合并后立刻与用户最后一段比较。
    规则：
    1) 方向一致 -> 路径段与用户段方向都视作已确认(将路径段方向置0)。
    2) 非最后一段用户段：计算误差=|路径段-用户段|；若>阈值(默认10m) -> 偏离；否则删除该段。
    3) 最后一段用户段：不计算误差，仅把路径段距离更新为差值(max(0, 路径-用户))。
    4) 若比较时路径键的“最后一段”需要与用户段比较，则与下一键的“第一段”合并：
       合并方向=保留前段方向；距离=两段距离相加；合并段放入“下一键”的开头；当前键删除。
    5) 键(锚点对)的所有段走完则删除该键，并在current_path中移除起点。
    """
    path_dict = {k: v.copy() for k, v in path_dict.items()}  # 深拷贝
    deviated = False
    current_path = current_path.copy()  # 避免修改原始列表

    # 用户段逐个处理
    for u_idx, (u_dir, u_dist) in enumerate(user_segments):
        keys = list(path_dict.keys())
        while keys and (not path_dict[keys[0]]):
            # 当前段已经走完，更新current_path
            finished_key = keys[0]
            if finished_key[0] in current_path:
                try:
                    current_path.remove(finished_key[0])
                except ValueError:
                    pass
            path_dict.pop(keys[0])
            keys = list(path_dict.keys())
        if not keys:
            break

        curr_key = keys[0]
        segs = path_dict[curr_key]

        # 若当前键只有最后一段，并且需要与当前用户段比较 -> 合并到下一键
        if len(segs) == 1 and len(keys) >= 2:
            next_key = keys[1]
            if path_dict[next_key]:
                last_dir, last_dist = segs[-1]
                next_dir, next_dist = path_dict[next_key][0]
                merged_dir = last_dir
                merged_dist = last_dist + next_dist
                path_dict[next_key][0] = (merged_dir, merged_dist)
                path_dict.pop(curr_key)
                # 更新current_path，删除起点
                if curr_key[0] in current_path:
                    try:
                        current_path.remove(curr_key[0])
                    except ValueError:
                        pass
                curr_key = next_key
                segs = path_dict[curr_key]

        if not segs:
            continue

        p_dir, p_dist = segs[0]

        # Step1: 方向一致性
        if abs(p_dir) != abs(u_dir):
            deviated = True
            return path_dict, current_path, deviated
        else:
            segs[0] = (0, p_dist)

        # Step2: 距离逻辑
        is_last_user = (u_idx == len(user_segments) - 1)
        if is_last_user:
            new_dist = p_dist - u_dist
            if new_dist < 0:
                new_dist = 0
            segs[0] = (0, new_dist)
        else:
            err = abs(p_dist - u_dist)
            if err > error_threshold:
                deviated = True
                return path_dict, current_path, deviated
            segs.pop(0)
            if not segs:
                # 段走完，删除key并更新current_path
                path_dict.pop(curr_key)
                if curr_key[0] in current_path:
                    try:
                        current_path.remove(curr_key[0])
                    except ValueError:
                        pass

    return path_dict, current_path, deviated



# 6.导航结束检测
def check_navigation_finished(path_dict, threshold=10):
    """
    检查用户是否到达终点

    参数:
        path_dict: dict {(start,end): [(dir, dist), ...]}
        threshold: float，允许的终点误差（默认10米）

    返回:
        finished: bool，是否完成导航
        error: float，终点误差（米）
    """
    # 如果字典为空，说明路径走完了，误差=0
    if not path_dict:
        return True, 0.0

    # 字典中剩下的锚点对
    if len(path_dict) == 1:
        last_key = list(path_dict.keys())[0]
        segments = path_dict[last_key]

        # 仅剩下最后一个元组
        if len(segments) == 1:
            _, dist = segments[0]
            if dist < threshold:
                return True, dist  # 导航完成，误差=剩余距离

    return False, None


