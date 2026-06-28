import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from sklearn.decomposition import PCA
import Data_processing

matplotlib.use('TkAgg')

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 最优 K 值
BEST_K2 = 0.4706652832031252
BEST_K=0.47

# 读取相关文件
# file_path_2 = "../data/collectionData/局部_小米MAX3/sensor_20230705_1306.csv"
# df_2 = pd.read_csv(file_path_2)
# Anchor_data_2 = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_局部_小米MAX3_refined.csv')
# file_path = "../data/collectionData_new/荟聚_华为_1/sensor_20250412_1434.csv"
# df = pd.read_csv(file_path)
# Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_荟聚_华为_1.csv')
# file_path = "../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230706_2104.csv"
# df = pd.read_csv(file_path)
# Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_交叉_小米MAX3_移动卡_refined.csv')

file_path = "../data/collectionData_new/mi_max_3_path6/sensor_20251125_1635.csv"
df = pd.read_csv(file_path)
Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_mi_max_3_path6_refined.csv')

file_path_2 = "../data/collectionData_new/vivo_path8/sensor_20251125_1640.csv"
df_2 = pd.read_csv(file_path_2)
Anchor_data_2 = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_vivo_path8_refined.csv')

# 1.文件预处理
def preprocessing(data):
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S:%f', errors='coerce')
    data['Time_sec'] = (data['Time'] - data['Time'].min()).dt.total_seconds()
    # 去除Z轴重力加速度(约9.8 m/s²)
    data['Acc_Z_corrected'] = data['Acc_Z'] - 9.8
    # 计算合成加速度并去均值
    data['Acc_mag'] = np.sqrt(data['Acc_X'] ** 2 + data['Acc_Y'] ** 2 + data['Acc_Z_corrected'] ** 2)
    data['Acc_mag'] -= data['Acc_mag'].mean()
    return data


# 2.步态检测 & 振幅计算
def detect_amplitudes(acc_data, fs=50, peak_threshold_factor=1.0, window_size=10):
    mean_val = np.mean(acc_data)
    std_val = np.std(acc_data)
    peak_threshold = mean_val + peak_threshold_factor * std_val
    min_distance = int(0.4 * fs)  # 约0.4秒的最小间隔

    peaks, _ = find_peaks(acc_data, height=peak_threshold, distance=min_distance)

    amplitudes = []
    for peak in peaks:
        start = max(peak - window_size, 0)
        end = min(peak + window_size, len(acc_data))
        peak_val = acc_data[peak]
        valley_before = np.min(acc_data[start:peak]) if peak > 0 else peak_val
        valley_after  = np.min(acc_data[peak:end])   if peak < len(acc_data)-1 else peak_val
        A_i = peak_val - np.mean([valley_before, valley_after])
        amplitudes.append(A_i)
    return amplitudes, peaks


# 5.路径构建
def reconstruct_path(step_lengths, step_angles):
    positions = [(0, 0)]
    for L, angle in zip(step_lengths, step_angles):
        x_prev, y_prev = positions[-1]
        dx = L * np.cos(np.deg2rad(angle))
        dy = L * np.sin(np.deg2rad(angle))
        positions.append((x_prev + dx, y_prev + dy))
    return np.array(positions)

# 6.寻找锚点对应的步伐下标
def find_anchor(data, peaks):
    # 提取 Anchor_Info 并转换为数值
    data[['Start', 'End']] = data['Anchor_Info'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
    # 找到Start在peaks中的最近步数
    start_times = data['Start'].to_numpy()
    closest_peaks = np.array([peaks[np.abs(peaks - t).argmin()] for t in start_times])
    peak_indices = np.array([np.where(peaks == p)[0][0] for p in closest_peaks])
    # 替换Start为步数索引
    data['Start'] = peak_indices
    # 计算每个类别的开始步数的中值
    median_start_colum = data.groupby('Cluster_Label')['Start'].median().round().astype(int)

    # 使用层次聚类合并相差小于10的类别
    Z = linkage(median_start_colum.values.reshape(-1, 1), method='complete')  # 完全连接聚类
    threshold = 10  # 注意：这里的阈值是索引的差值，不是时间
    cluster_assignments = fcluster(Z, threshold, criterion='distance')  # 生成新的类别

    # 重新映射类别索引
    data['New_Cluster_Label'] = data['Cluster_Label'].map(dict(zip(median_start_colum.index, cluster_assignments)))

    # 计算合并后的中值，并按中值大小重新编号
    grouped_medians = data.groupby('New_Cluster_Label')['Start'].median().round().astype(int)
    sorted_labels = {old: new for new, old in
                     enumerate(sorted(grouped_medians.index, key=lambda x: grouped_medians[x]))}

    data['Sorted_Cluster_Label'] = data['New_Cluster_Label'].map(sorted_labels)
    median_start_colum = grouped_medians.sort_values().to_numpy()

    # 手动调整部分中值时间
    # median_start_colum[3] -= 10
    return median_start_colum


# 7.寻找转弯方向和对应的步伐下标
def get_turn(data, peaks):
    ore_raw = data['Ore'].copy().reset_index(drop=True)

    # 对 Ore 数据进行平滑处理
    window_length = 25 if 25 % 2 == 1 else 26  # 确保是奇数
    ore_smoothed = pd.Series(savgol_filter(ore_raw, window_length=window_length, polyorder=2))

    # 计算相邻差分（角度变化量）
    angle_diffs = ore_smoothed.diff().fillna(0).to_numpy()

    # 处理角度跳变问题（跨越180°和-180°）
    angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360, angle_diffs)
    angle_diffs = np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs)

    # 设置跳跃阈值（例如如果相邻角度变化超过180°，则认为是跳跃，跳过）
    jump_threshold = 180  # 角度差异大于180度的变化被视为跳跃
    angle_diffs = np.where(np.abs(angle_diffs) > jump_threshold, 0, angle_diffs)

    # 滑动窗口累计角度变化
    window_size = 200  # 4秒的窗口
    cumulative_diffs = pd.Series(angle_diffs).rolling(window=window_size, min_periods=1).sum().to_numpy()

    # 转向检测参数
    turn_threshold = 50  # 角度变化阈值
    min_interval = 400  # 6秒最小间隔

    # 搜索转向点
    turn_info = []  # 存 [方向, 在 peaks 中的下标]
    prev_idx = -min_interval  # 初始值

    for idx, cum_diff in enumerate(cumulative_diffs):
        if cum_diff > turn_threshold or cum_diff < -turn_threshold:
            if idx - prev_idx >= min_interval:
                # 确定转向方向
                direction = -1 if cum_diff > turn_threshold else 1

                # 找到peaks中最接近idx的索引
                peak_idx = np.argmin(np.abs(peaks - idx))

                # 记录转向信息
                turn_info.append([direction, peak_idx])
                prev_idx = idx

    return turn_info

# 手动修改转向种存在的误差
# 这个在更换文件时是需要手动修改的，目的是避免一些错误识别的转向
def correction_turn(turn_info):
    for turn in turn_info:
        if turn[1] == 471:
              turn[0] = 1
        if turn[1] == 360:
            turn[0] = -1
        if turn[1] == 705:
            turn[0] = -1
    remove_rows = {597,607 , 628}
    turn_info = [turn for turn in turn_info if turn[1] not in remove_rows]
    return turn_info


# 8.寻找锚点间的路径
def find_path_between_anchors(median_start_colum, turn_info, step_lengths):
    paths = []

    # 遍历每一对相邻的锚点
    for i in range(len(median_start_colum) - 1):
        A = median_start_colum[i]
        B = median_start_colum[i + 1]

        # 初始化路径，先存入起点和终点
        path = [(i, i+1)]
        last_position = A  # 记录当前路径的起始行位置
        if_turn = 0  # 初始没有转向

        for turn in turn_info:
            if A < turn[1] < B:  # 只考虑在两个锚点之间的转弯
                # 计算直行距离
                distance = np.sum(step_lengths[last_position:turn[1]])  # 计算当前位置到转向点的行走距离
                path.append((if_turn, int(round(distance))))  # 二元组（方向, 距离）
                if_turn = turn[0]  # 更新当前的转向
                last_position = turn[1]  # 更新转弯位置

        # 如果最后一个锚点之间没有转向，直接计算直行到B点的距离
        if last_position < B:
            distance = np.sum(step_lengths[last_position:B])  # 计算最后一个锚点到B点的行走距离
            path.append((if_turn, int(round(distance))))  # 二元组（方向, 距离）

        paths.append(path)

    return paths


# 让第二条路径向第一条对齐
def align_paths_by_start_and_direction(path_src, path_target, steps_for_direction=10):
    """
    将 path_src 平移+旋转，使其起点和初始方向对齐 path_target。
    """
    # 1. 对齐起点
    delta = path_target[0] - path_src[0]
    path_src_aligned = path_src + delta

    # 2. 计算初始方向向量
    if len(path_src_aligned) < steps_for_direction or len(path_target) < steps_for_direction:
        return path_src_aligned  # 数据太短无法对齐方向，直接返回已平移路径

    vec_src = path_src_aligned[steps_for_direction] - path_src_aligned[0]
    vec_tgt = path_target[steps_for_direction] - path_target[0]

    # 3. 计算旋转角
    angle_src = np.arctan2(vec_src[1], vec_src[0])
    angle_tgt = np.arctan2(vec_tgt[1], vec_tgt[0])
    angle_diff = angle_tgt - angle_src

    # 4. 构建旋转矩阵并应用
    rotation_matrix = np.array([
        [np.cos(angle_diff), -np.sin(angle_diff)],
        [np.sin(angle_diff),  np.cos(angle_diff)]
    ])
    rotated = (rotation_matrix @ (path_src_aligned - path_src_aligned[0]).T).T + path_src_aligned[0]

    return rotated



# 图示
def show_two_paths(path_ore_1, anchors_1, turns_1, df_1,
                   path_ore_2, anchors_2, turns_2, df_2):
    plt.figure(figsize=(12, 10))

    # 路径1
    plt.plot(path_ore_1[:, 0], path_ore_1[:, 1], '--', label='路径1（设备1）', color='blue')
    anchor_pos_1 = path_ore_1[anchors_1]
    plt.plot(anchor_pos_1[:, 0], anchor_pos_1[:, 1], 'ro', label='锚点1', markersize=8)

    turn_pos_1 = np.array([path_ore_1[t[1]] for t in turns_1])
    # if len(turn_pos_1) > 0:
    #     plt.plot(turn_pos_1[:, 0], turn_pos_1[:, 1], 'o', color='orange', label='转向点1', markersize=8)

    # 路径2
    plt.plot(path_ore_2[:, 0], path_ore_2[:, 1], '--', label='路径2（设备2）', color='green')
    anchor_pos_2 = path_ore_2[anchors_2]
    plt.plot(anchor_pos_2[:, 0], anchor_pos_2[:, 1], 'mo', label='锚点2', markersize=8)

    turn_pos_2 = np.array([path_ore_2[t[1]] for t in turns_2])
    # if len(turn_pos_2) > 0:
    #     plt.plot(turn_pos_2[:, 0], turn_pos_2[:, 1], 'o', color='orange', label='转向点2', markersize=8)

    # 起点（假设两者起点一致，取 path_ore_1[0]）
    start = path_ore_1[0]
    plt.scatter(start[0], start[1], color='black', s=100, label='起点', zorder=5)

    # 图形设置
    plt.xlabel('X 坐标 (米)', fontsize=16)
    plt.ylabel('Y 坐标 (米)', fontsize=16)
    plt.title('双路径对比（含锚点与转向点）', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # 航向角对比图
    plt.figure(figsize=(14, 6))
    plt.plot(df_1['Time_sec'], df_1['Ore'], label='设备1 航向角', linestyle='--', color='blue')
    plt.plot(df_2['Time_sec'], df_2['Ore'], label='设备2 航向角', linestyle='--', color='green')
    plt.xlabel('时间 (秒)', fontsize=16)
    plt.ylabel('航向角 (°)', fontsize=16)
    plt.title('航向角对比', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


# PDR算法复现路径

df = preprocessing(df)
df_2 = preprocessing(df_2)

# 2.步态检测 & 振幅计算
amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)  # peaks步数下标
amplitudes_2, peaks_2 = detect_amplitudes(df_2['Acc_mag'].values, fs=50)  # peaks步数下标

# 3.计算步长
step_lengths = [BEST_K * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
total_distance = np.sum(step_lengths)  # 计算总行走距离

step_lengths_2 = [BEST_K2 * (A_i ** 0.25) for A_i in amplitudes_2 if A_i > 0]
total_distance_2 = np.sum(step_lengths_2)  # 计算总行走距离

# 4.寻找航向
ore_angles = df['Ore'].values
step_angles_ore = ore_angles[peaks]  # 取步伐时刻的 Ore 数据  这个和peaks的下表是一致的

ore_angles_2 = df_2['Ore'].values
step_angles_ore_2 = ore_angles_2[peaks_2]  # 取步伐时刻的 Ore 数据  这个和peaks的下表是一致的


# 5.路径构建
path_ore = reconstruct_path(step_lengths, step_angles_ore)
path_ore_2 = reconstruct_path(step_lengths_2, step_angles_ore_2)

path_ore_2 = align_paths_by_start_and_direction(path_ore_2, path_ore, steps_for_direction=10)


# 调用这些函数，可以得到路径
median_start_colum = find_anchor(Anchor_data, peaks)
turn_info = get_turn(df, peaks)
paths = find_path_between_anchors(median_start_colum, turn_info, step_lengths)

median_start_colum_2 = find_anchor(Anchor_data_2, peaks_2)
turn_info_2 = get_turn(df_2, peaks_2)
paths_2 = find_path_between_anchors(median_start_colum_2, turn_info_2, step_lengths_2)

# 画图展示拐弯和锚点

show_two_paths(
    path_ore, median_start_colum, turn_info, df,
    path_ore_2, median_start_colum_2, turn_info_2, df_2
)
