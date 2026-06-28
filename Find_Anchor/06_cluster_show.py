import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from dtaidistance import dtw
from tqdm import tqdm
import umap.umap_ as umap  # 确保已安装 umap-learn 包
import Data_processing

matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 使用 DTW 计算样本之间的距离矩阵
def compute_dtw_distance_matrix(data_df):
    data = data_df.values
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples), desc="计算 DTW 距离矩阵"):
        for j in range(i + 1, n_samples):
            dist = dtw.distance(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


# ===================== Step 1: 读取和预处理数据 =====================
data = pd.read_csv('anchor_cluster/anchor_cluster_Samsung_Path1_refined.csv')
labels = data.iloc[:, -1].values  # 最后一列是 Cluster_Label

# 地磁信号
columns_meg = data.iloc[:, :400]
columns_meg_front = data.iloc[:, 400:800]
columns_meg_back = data.iloc[:, 800:1200]

# LTE 主信号段（拼接）
columns_lte = pd.concat([
    data.iloc[:, 1204:1208],
    data.iloc[:, 1216:1220],
    data.iloc[:, 1228:1232]
], axis=1)

# LTE 前后片段
columns_lte_front = pd.concat([
    data.iloc[:, 1200:1204],
    data.iloc[:, 1212:1216],
    data.iloc[:, 1224:1228]
], axis=1)

columns_lte_back = pd.concat([
    data.iloc[:, 1208:1212],
    data.iloc[:, 1220:1224],
    data.iloc[:, 1232:1236]
], axis=1)

# 预处理
meg_pred, meg_front_pred, meg_back_pred = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
lte_pred = Data_processing.LTE_Preprocessing(columns_lte) * 4
lte_front_pred = Data_processing.LTE_Preprocessing(columns_lte_front)
lte_back_pred = Data_processing.LTE_Preprocessing(columns_lte_back)

meg_pred = pd.DataFrame(meg_pred)
meg_front_pred = pd.DataFrame(meg_front_pred)
meg_back_pred = pd.DataFrame(meg_back_pred)

# 合并特征
meg_combined = pd.concat([meg_pred, meg_front_pred, meg_back_pred], axis=1)
meg_combined = meg_combined.interpolate(axis=1).fillna(0)

lte_combined = pd.concat([lte_pred, lte_front_pred, lte_back_pred], axis=1)
lte_combined = lte_combined.fillna(0)

# meg数据降维
umap_model = umap.UMAP(n_components=10, random_state=42)
meg_umap = pd.DataFrame(umap_model.fit_transform(meg_pred), index=meg_pred.index)

# ===================== Step 2: 距离矩阵计算 =====================
meg_dist = compute_dtw_distance_matrix(meg_umap)
lte_dist = pairwise_distances(lte_combined, metric='euclidean')

# ===================== Step 3: 距离归一化与加权 =====================
scaler = MinMaxScaler()
meg_dist_norm = scaler.fit_transform(meg_dist)
lte_dist_norm = scaler.fit_transform(lte_dist)

alpha = 0.4
D_total = np.sqrt((alpha * meg_dist_norm) ** 2 + ((1 - alpha) * lte_dist_norm) ** 2)
np.fill_diagonal(D_total, 0)

# ===================== Step 4: 聚类评分 =====================
mask = labels != -1  # 去掉噪声点
if np.sum(mask) > 1:
    score = silhouette_score(D_total[mask][:, mask], labels[mask], metric='precomputed')
    print(f"Silhouette Score: {score:.4f}")
else:
    print("有效聚类点不足，无法计算 Silhouette Score。")

# ===================== Step 5: PCA降维保存及可视化 =====================
pca_2d = PCA(n_components=2)
D_total_2d = pca_2d.fit_transform(D_total)

# 保存二维数据和类别标签
df_2d = pd.DataFrame(D_total_2d, columns=['PCA1', 'PCA2'])
df_2d['Cluster_Label'] = labels
df_2d.to_csv('anchor_cluster/anchor_cluster_Samsung_Path1_show.csv', index=False)
print("二维数据及标签已保存到 anchor_cluster/D_total_2d_with_labels.csv")

# 绘图
plt.figure(figsize=(10, 7))
scatter = plt.scatter(D_total_2d[:, 0], D_total_2d[:, 1], c=labels, cmap='tab20', s=15)
plt.title('D_total 融合距离的2D可视化 (PCA)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()
