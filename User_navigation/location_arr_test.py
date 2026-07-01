import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import Data_processing

matplotlib.use('TkAgg')

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 最优 K 值
BEST_K = 0.47

# 读取相关文件
# file_path = "../data/collectionData_new/荟聚_华为_1/sensor_20250412_1434.csv"
# df = pd.read_csv(file_path)
# Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_荟聚_华为_1.csv')
# 读取相关文件
# file_path = "../data/collectionData/局部_小米MAX3/sensor_20230513_1858.csv"
# df = pd.read_csv(file_path)
# Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_局部_小米MAX3_refined.csv')
# 读取相关文件
file_path = "../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230628_1443.csv"
df = pd.read_csv(file_path)
Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_交叉_小米MAX3_移动卡_refined.csv')


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
    median_start_colum[1] += 3
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
        if turn[1] == 210:
              turn[0] = 1
        if turn[1] == 144:
              turn[0] = -1
        if turn[1] == 108:
            turn[0] = 1
        if turn[1] == 314:
              turn[0] = -1
        if turn[1] == 360:
            turn[0] = -1
        if turn[1] == 705:
            turn[0] = -1
    remove_rows = {0,59,72, 310}
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


# 9.图示
def show(path_ore, median_start_colum, turn_info, df):
    # 绘制路径图
    plt.figure(figsize=(10, 8))
    # 路径
    plt.plot(path_ore[:, 0], path_ore[:, 1], '--', label='仅使用 Ore 重建路径', color='blue')
    # 锚点
    anchor_positions = path_ore[median_start_colum]  # 从路径中取出锚点位置
    plt.plot(anchor_positions[:, 0], anchor_positions[:, 1], 'ro', markersize=8, label='锚点')
    # 转向点
    turn_positions = [path_ore[turn[1], :] for turn in turn_info]
    turn_positions = np.array(turn_positions)  # 转向点位置
    plt.plot(turn_positions[:, 0], turn_positions[:, 1], 'o', color='orange', markersize=8, label='转向点')
    # 起点
    start_position = path_ore[0, :]  # 路径的起点
    plt.scatter(start_position[0], start_position[1], color='green', s=100, label='起点', zorder=5)

    # 图表设置
    plt.xlabel('X 坐标 (米)', fontsize=16)
    plt.ylabel('Y 坐标 (米)', fontsize=16)
    plt.title('路径重建对比（带锚点和转向点）', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # 绘制航向角对比图
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_sec'], df['Ore'], label='原始 Ore 航向', color='green', linestyle='--')
    plt.xlabel('时间 (秒)', fontsize=16)
    plt.ylabel('航向角 (°)', fontsize=16)
    plt.title('航向角随时间变化', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    # plt.show()


def get_position_from_row(row_index, peaks, step_lengths, path_ore):
    # 找到 row_index 所处的步数
    i = np.searchsorted(peaks, row_index, side='right') - 1
    if i < 0:
        print("当前行编号在路径开始之前")
        return 0.0, path_ore[0]
    elif i >= len(step_lengths):
        print("当前行编号超出路径范围")
        return np.sum(step_lengths), path_ore[-1]

    distance = np.sum(step_lengths[:i])
    position = path_ore[i]
    return distance, position


# PDR算法复现路径
def PDR(df,Anchor_data,If_show=True):
    # 1.文件预处理
    df = preprocessing(df)

    # 2.步态检测 & 振幅计算
    amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)  # peaks步数下标
    print(peaks)
    # 3.计算步长
    step_lengths = [BEST_K * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
    total_distance = np.sum(step_lengths)  # 计算总行走距离
    print(f"估计总行走距离: {total_distance:.3f} 米")

    # 4.寻找航向
    ore_angles = df['Ore'].values
    step_angles_ore = ore_angles[peaks]
    # ✨ 正确 reshape 成二维数组
    angles_for_clustering = step_angles_ore.reshape(-1, 1)
    # 进行聚类（这里选 8 类方向）
    kmeans = KMeans(n_clusters=8, random_state=0).fit(angles_for_clustering)
    # 将每个角度用对应聚类中心值替代（方向更平稳）
    step_angles_ore = kmeans.cluster_centers_[kmeans.labels_].flatten()

    # 5.路径构建
    path_ore = reconstruct_path(step_lengths, step_angles_ore)

    # 调用这些函数，可以得到路径
    median_start_colum = find_anchor(Anchor_data, peaks)

    turn_info = get_turn(df, peaks)
    turn_info = correction_turn(turn_info)

    paths = find_path_between_anchors(median_start_colum, turn_info, step_lengths)
    row_index =3040   # 你感兴趣的行编号
    distance, position = get_position_from_row(row_index, peaks, step_lengths, path_ore)
    print(f"在第 {row_index} 行时，已经走了约 {distance:.2f} 米，路径坐标为 {position}")


    print("\n每个锚点对应的步数位置（在路径中）及累计距离：")
    anchor_distances = []
    for i, step_idx in enumerate(median_start_colum):
        distance = np.sum(step_lengths[:step_idx])
        anchor_distances.append(distance)
        print(f"锚点 {i}：第 {step_idx} 步，对应累计距离 ≈ {distance:.2f} 米")

    if If_show==True:
        # 画图展示拐弯和锚点
        show(path_ore, median_start_colum, turn_info, df)

    return median_start_colum,turn_info,paths


# 调用PDR方法重建路径
median_start_colum,turn_info,paths=PDR(df,Anchor_data,True)