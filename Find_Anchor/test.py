import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import Data_processing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from dtaidistance import dtw
import numpy as np
from tqdm import tqdm
import umap.umap_ as umap  # 确保已安装 umap-learn 包
import pandas as pd
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号



# 使用 DTW 计算样本之间的距离矩阵
def compute_dtw_distance_matrix(data_df):
    data = data_df.values
    n_samples = data.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    # 避免重复计算对称项
    for i in tqdm(range(n_samples), desc="计算 DTW 距离矩阵"):
        for j in range(i + 1, n_samples):
            dist = dtw.distance(data[i], data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


# ===================== Step 1: 读取和预处理数据 =====================
data = pd.read_csv('anchor/anchor_combined_交叉_小米MAX3_移动卡.csv')
data = Data_processing.Data_Preprocessing(data)

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
lte_combined=lte_combined.fillna(0)

# meg数据降维
# pca = PCA(n_components=50)  # 自动确定主成分数
# meg_pca = pd.DataFrame(pca.fit_transform(meg_pred), index=meg_pred.index)
umap_model = umap.UMAP(n_components=20, random_state=42)
meg_umap = pd.DataFrame(umap_model.fit_transform(meg_pred), index=meg_pred.index)

# ===================== Step 2: 距离矩阵计算 =====================
# 计算meg的DTW距离和lte的欧氏距离
meg_dist = compute_dtw_distance_matrix(meg_umap)
lte_dist = pairwise_distances(lte_combined, metric='euclidean')
# lte_dist = compute_dtw_distance_matrix(lte_combined)

# ===================== Step 3: 距离归一化与加权 =====================
scaler = MinMaxScaler()
meg_dist_norm = scaler.fit_transform(meg_dist)
lte_dist_norm = scaler.fit_transform(lte_dist)

alpha = 0.4
D_total = np.sqrt((alpha * meg_dist_norm) ** 2 + ((1 - alpha) * lte_dist_norm) ** 2)
np.fill_diagonal(D_total, 0)
print(D_total.shape)
# ===================== Step 4: 聚类（HDBSCAN） =====================
clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5)
labels = clusterer.fit_predict(D_total)
data['Cluster_Label'] = labels

# 保存覆盖原始文件
# data.to_csv('anchor_cluster/anchor_cluster_荟聚_华为_1.csv', index=False)

# ===================== Step 5: 聚类评分 =====================
score = silhouette_score(D_total, labels, metric='precomputed')
# 去除噪声点的评分
# mask = labels != -1
# score = silhouette_score(meg_dist_norm[mask][:, mask], labels[mask], metric='precomputed')
print(f"Silhouette Score: {score:.4f}")

# ===================== Step 6: 可视化 =====================
umap_model = umap.UMAP(n_components=2, metric='precomputed', random_state=42)
embedding = umap_model.fit_transform(D_total)

embedding_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
embedding_df['Cluster_Label'] = labels
embedding_df.to_csv('embedding_signal_labels.csv', index=False, encoding='utf-8-sig')

print("降维后的二维坐标和聚类标签已保存到 embedding_signal_labels.csv")

plt.figure(figsize=(10, 7))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab20', s=15)
plt.title('融合距离聚类结果二维可视化 (UMAP)')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()