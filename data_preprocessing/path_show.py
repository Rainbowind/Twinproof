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

matplotlib.use('TkAgg')

# 设置 Matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 最优 K 值
BEST_K = 0.47

# 读取相关文件
file_path = "../data/collectionData_new/MIMAX3_Path11/sensor_20260111_1752.csv"
df = pd.read_csv(file_path)

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
    turn_threshold = 30  # 角度变化阈值
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

# 9.图示
def show(path_ore, turn_info):
    # 绘制路径图
    plt.figure(figsize=(10, 8))
    # 路径
    plt.plot(path_ore[:, 0], path_ore[:, 1], '--', label='仅使用 Ore 重建路径', color='blue')

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


# PDR算法复现路径
def PDR(df,If_show=True):
    # 1.文件预处理
    df = preprocessing(df)

    # 2.步态检测 & 振幅计算
    amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)  # peaks步数下标

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

    turn_info = get_turn(df, peaks)
    turn_info = correction_turn(turn_info)


    if If_show==True:
        # 画图展示拐弯和锚点
        show(path_ore, turn_info)



# 调用PDR方法重建路径
PDR(df,True)