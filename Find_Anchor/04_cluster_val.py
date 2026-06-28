import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import umap.umap_ as umap
from dtaidistance import dtw
from dtaidistance import dtw_ndim
import Data_processing

matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


# 使用 DTW 计算类内异常值
def detect_outliers_dtw(data_df, labels, threshold=2.0):
    """
    data_df: 特征数据 (降维后的 MEG 片段)
    labels: 类别标签
    threshold: 距离超过 (均值 + threshold*std) 视为异常
    """
    outlier_indices = []
    unique_labels = [l for l in np.unique(labels) if l != -1]

    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            continue

        avg_dists = np.zeros(len(idx))
        for i, id_i in enumerate(idx):
            s1 = data_df.iloc[id_i].values.astype(np.double)
            dists = [
                dtw.distance_fast(s1, data_df.iloc[id_j].values.astype(np.double))
                for id_j in idx if id_j != id_i
            ]
            avg_dists[i] = np.mean(dists)

        mean_dist, std_dist = np.mean(avg_dists), np.std(avg_dists)
        for i, sample_id in enumerate(idx):
            if avg_dists[i] > mean_dist + threshold * std_dist:
                outlier_indices.append(sample_id)
                print(f"类 {label} 的异常点: index={sample_id}, 平均距离={avg_dists[i]:.4f}")

    return outlier_indices


# ===================== Step 1: 读取和预处理数据 =====================
data = pd.read_csv('anchor_cluster/anchor_cluster_vivos9_path3.csv')
data = Data_processing.Data_Preprocessing(data)

# 提取地磁信号
columns_meg = data.iloc[:, :400]
columns_meg_front = data.iloc[:, 400:800]
columns_meg_back = data.iloc[:, 800:1200]

# Meg 预处理 (已包含平滑和归一化)
meg_pred, meg_front_pred, meg_back_pred = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front,
                                                                            columns_meg_back)

meg_front_pred = pd.DataFrame(meg_front_pred).interpolate(axis=1).fillna(0)
meg_back_pred = pd.DataFrame(meg_back_pred).interpolate(axis=1).fillna(0)

# ===================== Step 2: 前后 MEG 段分别降维 =====================
umap_model_front = umap.UMAP(n_components=20, random_state=42)
meg_front_umap = pd.DataFrame(umap_model_front.fit_transform(meg_front_pred), index=meg_front_pred.index)

umap_model_back = umap.UMAP(n_components=20, random_state=42)
meg_back_umap = pd.DataFrame(umap_model_back.fit_transform(meg_back_pred), index=meg_back_pred.index)

# ===================== Step 3: 类内异常点检测并删除 =====================
labels = data.iloc[:, -1].values  # 最后一列为 Cluster_Label

# 前段异常点
outliers_front = detect_outliers_dtw(meg_front_umap, labels, threshold=2.0)
# 后段异常点
outliers_back = detect_outliers_dtw(meg_back_umap, labels, threshold=2.0)

# 合并前后段的异常点
outliers_total = set(outliers_front).union(set(outliers_back))
print(f"总异常点数量: {len(outliers_total)}")

# 删除异常点
data_cleaned = data.drop(index=outliers_total).reset_index(drop=True)

# 保存清理后的数据
data_cleaned.to_csv('anchor_cluster/anchor_cluster_vivos9_path3_refined.csv', index=False)
print("清理后的数据已保存。")
