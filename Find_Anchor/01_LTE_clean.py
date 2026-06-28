import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw

# ========== 配置区域 ==========
folder_path = r'../data/collectionData_new_02/vivox100_Path12'
labels = ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']
TARGET_LEN = 200  # 目标采样长度，控制速度
# ======================================

csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
csv_files.sort()

def extract_rssi_columns(file_path):
    """提取 RSSI 列并降采样到 TARGET_LEN 长度"""
    df = pd.read_csv(file_path)
    rssi_dict = {}
    for col in labels:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy()
            if len(vals) > 0:
                x = np.linspace(0, 1, len(vals))
                x_new = np.linspace(0, 1, TARGET_LEN)
                vals = np.interp(x_new, x, vals)
            rssi_dict[col] = vals
        else:
            rssi_dict[col] = np.array([])
    return rssi_dict

# 加载并降采样数据
rssi_data = {f: extract_rssi_columns(f) for f in csv_files if any(len(v) > 0 for v in extract_rssi_columns(f).values())}
file_list = list(rssi_data.keys())
print(f"共加载 {len(file_list)} 个有效文件用于DTW比较 (降采样到 {TARGET_LEN} 点)")

def avg_dtw_distance(file1, file2):
    """对RSSI1,2,3分别计算DTW距离并取平均"""
    dists = []
    for col in labels:
        seq1 = rssi_data[file1][col]
        seq2 = rssi_data[file2][col]
        if len(seq1) > 0 and len(seq2) > 0:
            dist, _ = fastdtw(seq1, seq2, dist=lambda a, b: abs(a - b))
            dists.append(dist)
    return np.mean(dists) if dists else np.nan

# 计算平均距离
distances = []
for i in range(len(file_list)):
    for j in range(i + 1, len(file_list)):
        avg_dist = avg_dtw_distance(file_list[i], file_list[j])
        if not np.isnan(avg_dist):
            distances.append((file_list[i], file_list[j], avg_dist))

# 计算每个文件的平均 DTW 距离
avg_distances = {f: [] for f in file_list}
for f1, f2, d in distances:
    avg_distances[f1].append(d)
    avg_distances[f2].append(d)

avg_distances = {f: np.mean(dlist) for f, dlist in avg_distances.items()}
mean_dist = np.mean(list(avg_distances.values()))
std_dist = np.std(list(avg_distances.values()))
threshold = mean_dist + std_dist

print(f"\nDTW平均距离: {mean_dist:.2f}, 阈值: {threshold:.2f}")
print("异常文件：")
for f, avg_d in avg_distances.items():
    if avg_d > threshold:
        print(f" - {os.path.basename(f)} (平均DTW: {avg_d:.2f})")
