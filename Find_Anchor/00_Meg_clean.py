import os
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 文件夹路径
folder_path = "../data/collectionData_new_02/vivox100_Path12"  # 修改为你的路径
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

meg_series_list = []
file_names = []

for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path, usecols=['Meg'])
        # 确保是一维浮点数数组
        meg_series = pd.to_numeric(df['Meg'], errors='coerce').dropna().to_numpy(dtype=float).ravel()

        if len(meg_series) == 0:
            print(f"⚠️ {file_name} 的 Meg 列为空或无效，跳过。")
            continue

        meg_series_list.append(meg_series)
        file_names.append(file_name)
    except Exception as e:
        print(f"❌ 读取 {file_name} 出错: {e}")

if len(meg_series_list) == 0:
    raise ValueError("没有有效的 Meg 序列！")

# 选择第一个文件作为参考序列
reference_series = meg_series_list[0]

# 计算每个序列与参考序列的 FastDTW 距离
distances = []
for i, series in enumerate(meg_series_list):
    try:
        dist, _ = fastdtw(reference_series, series, dist=lambda x, y: abs(x - y))
        distances.append(dist)
    except Exception as e:
        print(f"❌ 计算 {file_names[i]} 的 DTW 距离出错: {e}")
        distances.append(np.inf)

# 判定异常：距离大于均值 + 3*标准差
valid_distances = [d for d in distances if np.isfinite(d)]
mean_dist = np.mean(valid_distances)
std_dist = np.std(valid_distances)
threshold = mean_dist + std_dist

print(f"\nDTW 平均距离: {mean_dist:.2f}, 阈值: {threshold:.2f}")
print("==== 可能异常的文件 ====")
for file, dist in zip(file_names, distances):
    if dist > threshold:
        print(f"{file} -> DTW 距离: {dist:.2f}")
