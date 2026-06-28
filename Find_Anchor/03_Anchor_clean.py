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
data = pd.read_csv('anchor/anchor_combined_vivos9_path3.csv')

# 地磁信号
columns_meg = data.iloc[:, :400]
columns_meg_front = data.iloc[:, 400:800]
columns_meg_back = data.iloc[:, 800:1200]

# 预处理
meg_pred, meg_front_pred, meg_back_pred = Data_processing.Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)

meg_pred = pd.DataFrame(meg_pred)

# meg数据降维
pca = PCA(n_components=20)
meg_pca = pd.DataFrame(pca.fit_transform(meg_pred), index=meg_pred.index)

# ===================== Step 2: 距离矩阵计算 =====================
# 计算欧氏距离
meg_dist = compute_dtw_distance_matrix(meg_pca)

min_distances = []
for i in range(len(meg_dist)):
    # 排除自身距离（设为inf），然后找最小值
    dists = meg_dist[i].copy()
    dists[i] = np.inf
    min_distances.append(np.min(dists))

min_distances = np.array(min_distances)

# 设置距离阈值：可使用分位数或均值+标准差方式
threshold = np.percentile(min_distances, 75)  # 或者 threshold = min_distances.mean() + min_distances.std()

# 标记为伪锚点的索引
pseudo_anchor_indices = np.where(min_distances > threshold)[0]

# 输出伪锚点数量及其索引
print(f"检测出伪锚点数量: {len(pseudo_anchor_indices)}")
print("伪锚点索引:", pseudo_anchor_indices)


# ===================== Step 3: 删除伪锚点并保存 =====================
# 从原始数据中删除这些伪锚点行
cleaned_data = data.drop(index=pseudo_anchor_indices).reset_index(drop=True)

# 保存覆盖原始文件
cleaned_data.to_csv('anchor/anchor_combined_vivos9_path3.csv', index=False)
print(f"已删除伪锚点并保存。剩余锚点数量: {len(cleaned_data)}")
