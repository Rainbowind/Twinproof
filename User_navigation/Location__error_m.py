import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans

import Window_preprocessing
import Location

# ================== 基本设置 ==================
BEST_K = 0.47  # Weinberg-style 步长系数

merged_csv = "../data/collectionData_02/交叉_小米MAX3_移动卡/20230709_1315_merged.csv"
sensor_csv = "../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230709_1315.csv"

anchor_feat_detect = "../path_reconstruction/Anchor_features_DBA.csv"  # 用于候选+DTW匹配
anchor_cluster_truth = pd.read_csv("../Find_Anchor/anchor_cluster/anchor_cluster_交叉_小米MAX3_移动卡_refined.csv")  # 用于真实锚点步序

data_merged = pd.read_csv(merged_csv)
anchor_data_detect = pd.read_csv(anchor_feat_detect)
sensor_csv = pd.read_csv(sensor_csv)

window_size = 400
step_size = 20
num_rows = len(data_merged)

detected_anchors = []  # [(pred_label, row_index), ...]
start = 0
while start <= num_rows - window_size:
    end = start + window_size

    # 提取窗口内 MEG / LTE（来自 merged）
    meg = data_merged['Meg'].iloc[start:end].values
    lte = []
    for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']:
        lte_column = data_merged[col].iloc[start:end:100].values[:4]
        if len(lte_column) < 4:
            lte_column = list(lte_column) + [None] * (4 - len(lte_column))
        lte.extend(lte_column)

    # 预处理
    meg_df = pd.DataFrame([meg])
    lte_df = pd.DataFrame([lte])
    meg_processed = Window_preprocessing.Meg_Preprocessing(meg_df)
    lte_processed = Window_preprocessing.LTE_Preprocessing(lte_df)

    # LTE 候选
    result_labels, lte_distances = Location.find_similar_anchors(
        lte_processed, anchor_data_detect, threshold=0.05
    )
    if result_labels == -1:
        start += step_size
        continue

    # 窗口是否为锚点
    anchor_flag = Location.is_anchor_by_meg(meg)
    if anchor_flag == 1:
        # 最终用 MEG-DTW 精匹配
        best_label, _ = Location.match_meg_with_dtw(
            meg_processed, anchor_data_detect, result_labels, lte_distances, step=8, alpha=0.5
        )
        detected_anchors.append((int(best_label), int(start + window_size)))
        start += 20  # 命中锚点时快推进
    else:
        start += step_size

print("预测锚点 (label, row_index)：", detected_anchors)


# 计算误差

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
# def PDR(df,Anchor_data):
#     # 1.文件预处理
#     df = preprocessing(df)
#
#     # 2.步态检测 & 振幅计算
#     amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)  # peaks步数下标
#     print(peaks)
#     # 3.计算步长
#     step_lengths = [BEST_K * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
#     total_distance = np.sum(step_lengths)  # 计算总行走距离
#     print(f"估计总行走距离: {total_distance:.3f} 米")
#
#     # 4.寻找航向
#     ore_angles = df['Ore'].values
#     step_angles_ore = ore_angles[peaks]
#     # ✨ 正确 reshape 成二维数组
#     angles_for_clustering = step_angles_ore.reshape(-1, 1)
#     # 进行聚类（这里选 8 类方向）
#     kmeans = KMeans(n_clusters=8, random_state=0).fit(angles_for_clustering)
#     # 将每个角度用对应聚类中心值替代（方向更平稳）
#     step_angles_ore = kmeans.cluster_centers_[kmeans.labels_].flatten()
#
#     # 5.路径构建
#     path_ore = reconstruct_path(step_lengths, step_angles_ore)
#
#     # 调用这些函数，可以得到路径
#     median_start_colum = find_anchor(Anchor_data, peaks)
#
#     row_index =10900   # 你感兴趣的行编号
#     distance, position = get_position_from_row(row_index, peaks, step_lengths, path_ore)
#     print(f"在第 {row_index} 行时，已经走了约 {distance:.2f} 米，路径坐标为 {position}")
#
#     print("\n每个锚点对应的步数位置（在路径中）及累计距离：")
#     anchor_distances = []
#     for i, step_idx in enumerate(median_start_colum):
#         distance = np.sum(step_lengths[:step_idx])
#         anchor_distances.append(distance)
#         print(f"锚点 {i}：第 {step_idx} 步，对应累计距离 ≈ {distance:.2f} 米")
#
#     return median_start_colum
def PDR(df, Anchor_data, detected_anchors, true_labels=None):
    """
    df: sensor DataFrame
    Anchor_data: refined cluster DataFrame
    detected_anchors: [(pred_label, row_index), ...]
    true_labels: 自定义真实锚点编号，比如 [4,5,6,0,8]
    """
    # 1. 文件预处理
    df = preprocessing(df)

    # 2. 步态检测 & 振幅计算
    amplitudes, peaks = detect_amplitudes(df['Acc_mag'].values, fs=50)
    step_lengths = [BEST_K * (A_i ** 0.25) for A_i in amplitudes if A_i > 0]
    total_distance = np.sum(step_lengths)
    print(f"估计总行走距离: {total_distance:.3f} 米")

    # 3. 航向角离散化
    ore_angles = df['Ore'].values
    step_angles_ore = ore_angles[peaks]
    angles_for_clustering = step_angles_ore.reshape(-1, 1)
    kmeans = KMeans(n_clusters=8, random_state=0).fit(angles_for_clustering)
    step_angles_ore = kmeans.cluster_centers_[kmeans.labels_].flatten()

    # 4. 路径构建
    path_ore = reconstruct_path(step_lengths, step_angles_ore)

    # 5. 真实锚点
    median_start_colum = find_anchor(Anchor_data, peaks)
    true_anchor_distances = []
    for i, step_idx in enumerate(median_start_colum):
        dist = np.sum(step_lengths[:step_idx])
        true_anchor_distances.append(dist)

    # 如果提供了自定义编号，就替换
    if true_labels is None:
        true_labels = list(range(len(true_anchor_distances)))
    else:
        if len(true_labels) != len(true_anchor_distances):
            raise ValueError("true_labels 数量必须和真实锚点数量一致")

    print("\n真实锚点编号与距离：")
    for lbl, dist in zip(true_labels, true_anchor_distances):
        print(f"锚点 {lbl}: 距离 ≈ {dist:.2f} 米")

    # 6. 预测锚点与真实锚点的误差
    print("\n=== 定位误差计算 ===")
    errors = []
    for pred_label, row_index in detected_anchors:
        pred_dist, _ = get_position_from_row(row_index, peaks, step_lengths, path_ore)

        # 在 true_labels 中找相同编号
        if pred_label in true_labels:
            idx = true_labels.index(pred_label)
            true_dist = true_anchor_distances[idx]
            err = abs(pred_dist - true_dist)
            errors.append(err)
            print(f"锚点 {pred_label}: 真实={true_dist:.2f} m, 预测={pred_dist:.2f} m, 误差={err:.2f} m")
        else:
            print(f"[跳过] 预测锚点 {pred_label} 不在真实锚点编号列表 {true_labels}")

    if errors:
        print(f"\n平均误差: {np.mean(errors):.2f} m, 中位误差: {np.median(errors):.2f} m")

    return true_anchor_distances, errors


# 假设真实锚点编号应该是 [4,5,6,0,8]
true_labels = [4,5,6,0,8]


# 调用PDR方法重建路径
true_anchor_distances, errors=PDR(sensor_csv,anchor_cluster_truth,detected_anchors, true_labels=true_labels)
# print(errors)