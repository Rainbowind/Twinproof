import pandas as pd
import pywt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import Window_preprocessing
from numpy.linalg import norm
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


#匹配lte信号
def find_similar_anchors(lte_processed, anchor_data, threshold=0.05):
    """
    比对 LTE 信号和锚点 LTE 信号，返回匹配到的锚点标签列表和对应距离，或 -1。

    参数:
        lte_processed: pd.DataFrame (shape = (1, 12))
            当前窗口的 LTE 信号，已预处理。
        anchor_file: str
            包含锚点特征的 CSV 文件路径。
        threshold: float
            相似度判定的归一化欧氏距离阈值。

    返回:
        matched_labels: list 或 int
            匹配的标签列表，如果没有匹配则返回 -1。
        matched_distances: list
            匹配的标签对应的归一化欧氏距离列表。
    """
    # 1. 读取锚点数据
    lte_cols = [f"lte_{i}" for i in range(1, 13)]
    anchor_lte = anchor_data[lte_cols]
    anchor_labels = anchor_data["label"].astype(int)

    # 2. 对锚点 LTE 信号进行预处理
    anchor_lte_processed = Window_preprocessing.LTE_Preprocessing(anchor_lte)

    # 3. 获取当前 LTE 和锚点 LTE 的 numpy 数组
    lte_current = lte_processed.iloc[0].values
    anchor_array = anchor_lte_processed.values

    # 4. 计算所有欧氏距离
    distances = [norm(lte_current - anchor) for anchor in anchor_array]

    # 5. 归一化欧氏距离（按最大值归一化）
    max_dist = max(distances) if max(distances) != 0 else 1.0
    distances_norm = [d / max_dist for d in distances]

    # 6. 找出低于阈值的匹配项
    similar_indices = [i for i, d in enumerate(distances_norm) if d < threshold]
    if similar_indices:
        matched_labels = anchor_labels.iloc[similar_indices].tolist()
        matched_distances = [distances_norm[i] for i in similar_indices]
    else:
        matched_labels = -1
        matched_distances = []

    return matched_labels, matched_distances


# ========== 6. 锚点筛选函数 ==========
def is_anchor_by_meg(meg_signal, mean_th=0.81, peak_th=2.5, valley_th=2.2):
    """
    通过分析原始 MEG 信号的波动特征，判断是否为锚点。
    参数:
        meg_signal: np.array, 原始 MEG 信号 (长度 400)
        mean_th, std_th, peak_th, valley_th: float, 阈值
    返回:
        1: 是锚点
       -1: 不是锚点
    """
    meg_df = pd.DataFrame({'Meg': meg_signal})

    # 计算一阶差分并平滑
    meg_df['Meg_diff1'] = meg_df['Meg'].diff().rolling(window=5, min_periods=1).mean()

    # 标准化
    scaler = StandardScaler()
    meg_df['Meg_diff1_scaler'] = scaler.fit_transform(meg_df[['Meg_diff1']])

    # 取绝对值
    meg_df['Meg_diff1_abs'] = np.abs(meg_df['Meg_diff1_scaler'])

    # 计算特征
    mean_val = meg_df['Meg_diff1_abs'].mean()
    peak_val = meg_df['Meg_diff1_abs'].max()
    valley_val = -meg_df['Meg_diff1_scaler'].min()  # 取负号便于和peak对比
    # 判断是否超过阈值
    if (mean_val > mean_th) and (peak_val > peak_th) and (valley_val > valley_th):
        return 1
    else:
        return -1


# ========== 8. MEG 匹配函数（基于 DTW） ==========
def match_meg_with_dtw(meg_processed, anchor_data, candidate_labels, lte_distances, step=8, alpha=0.5):
    """
    使用 DTW 距离匹配 MEG 信号与候选锚点 MEG 信号，并结合 LTE 距离进行综合判断。

    参数:
        meg_processed: pd.DataFrame (shape = (1, 400))
            当前窗口的 MEG 信号（已预处理）
        anchor_file: str
            包含锚点特征的 CSV 文件路径
        candidate_labels: list
            LTE 匹配返回的候选锚点标签
        lte_distances: list
            与 candidate_labels 一一对应的 LTE 归一化欧氏距离
        step: int
            降维采样间隔，例如 8 表示每 8 个点取一次
        alpha: float
            LTE 和 MEG 的权重（0~1），例如 0.5 表示两者平权

    返回:
        best_label: int
            最小综合距离对应的标签
        best_distance: float
            最小的综合距离
    """
    # 1. 读取锚点数据

    # 2. 筛选候选标签对应的行
    candidate_rows = anchor_data[anchor_data['label'].isin(candidate_labels)]
    if candidate_rows.empty:
        return -1, float("inf")

    # 3. 提取 MEG 数据并预处理
    meg_cols = [f"meg_{i}" for i in range(400)]
    anchor_meg = candidate_rows[meg_cols]
    anchor_meg_processed = Window_preprocessing.Meg_Preprocessing(anchor_meg)

    # ✅ 当前窗口的 MEG 同样也要预处理（防止尺度不一致）
    meg_processed = Window_preprocessing.Meg_Preprocessing(meg_processed)

    # 4. 降采样
    current_meg = np.asarray(meg_processed.iloc[0].values[::step], dtype=float)
    anchor_meg_sampled = anchor_meg_processed.iloc[:, ::step]

    # 5. 计算 DTW 距离
    meg_distances = []
    labels = candidate_rows['label'].values
    for i in range(anchor_meg_sampled.shape[0]):
        anchor_signal = np.asarray(anchor_meg_sampled.iloc[i].values, dtype=float)
        dist = dtw.distance(current_meg, anchor_signal)
        meg_distances.append(dist)

    # 6. 归一化 MEG 距离
    max_meg_dist = max(meg_distances) if max(meg_distances) > 0 else 1.0
    meg_distances_norm = [d / max_meg_dist for d in meg_distances]

    # 7. 融合 LTE 和 MEG 距离
    combined_distances = [
        alpha * lte_d + (1 - alpha) * meg_d
        for lte_d, meg_d in zip(lte_distances, meg_distances_norm)
    ]

    # 8. 选择最优标签
    best_idx = np.argmin(combined_distances)
    best_label = labels[best_idx]
    best_distance = combined_distances[best_idx]

    # print(f"LTE 归一化距离: {lte_distances}")
    # print(f"MEG 归一化 DTW 距离: {meg_distances_norm}")
    # print(f"最小综合距离: {best_distance}")
    return best_label, best_distance


# 锚点检测函数
def detect_stable_anchor(data, anchor_file, start, num_rows, threshold=0.5):
    """
    检测连续两次匹配同一个锚点且相似度距离低于阈值的情况

    参数:
        data: pd.DataFrame - 原始数据
        anchor_file: str - 锚点特征文件路径
        start: int - 起始窗口位置
        num_rows: int - 数据总行数
        threshold: float - 相似度阈值

    返回:
        (best_label, position): Tuple[int, int] - 匹配锚点编号和窗口终止位置
        若未检测到则返回 None
    """
    window_size = 400
    step_size = 20
    previous_label = None

    while start <= num_rows - window_size:
        end = start + window_size

        # 提取MEG和LTE信号
        meg = data['Meg'].iloc[start:end].values
        lte = []
        for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']:
            lte_column = data[col].iloc[start:end:100].values[:4]
            if len(lte_column) < 4:
                lte_column = list(lte_column) + [None] * (4 - len(lte_column))
            lte.extend(lte_column)

        meg_df = pd.DataFrame([meg])
        lte_df = pd.DataFrame([lte])
        meg_processed = Window_preprocessing.Meg_Preprocessing(meg_df)
        lte_processed = Window_preprocessing.LTE_Preprocessing(lte_df)

        result_labels, lte_distances = find_similar_anchors(lte_processed, anchor_file, threshold=0.05)

        if result_labels == -1:
            start += step_size
            continue

        anchor_flag = is_anchor_by_meg(meg)
        if anchor_flag == 1:
            best_label, distance = match_meg_with_dtw(
                meg_processed, anchor_file, result_labels, lte_distances, step=8, alpha=0.6
            )
            # print(f"检测到锚点 {best_label}, 相似度距离: {distance}")

            if distance < threshold:
                if previous_label == best_label:
                    # print(f"连续两次检测到锚点 {best_label}，位置：{start + 400}")
                    return best_label, start + 400
                previous_label = best_label
            else:
                previous_label = None  # 重置，需连续两次满足条件

            start += 20
        else:
            start += step_size

    return None

def detect_stable_anchor_middle(data, anchor_file, start, num_rows):
    """
    检测是否匹配到任意锚点（用于中段重新定位）
    一旦检测到锚点就立即返回，不需要连续检测、也不判断距离阈值
    """
    window_size = 400
    step_size = 20

    while start <= num_rows - window_size:
        end = start + window_size
        print(f"[调试] 当前窗口: {start} ~ {end}")

        # 提取 MEG 和 LTE 信号
        meg = data['Meg'].iloc[start:end].values
        lte = []
        for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']:
            lte_column = data[col].iloc[start:end:100].values[:4]
            if len(lte_column) < 4:
                lte_column = list(lte_column) + [None] * (4 - len(lte_column))
            lte.extend(lte_column)

        meg_df = pd.DataFrame([meg])
        lte_df = pd.DataFrame([lte])
        meg_processed = Window_preprocessing.Meg_Preprocessing(meg_df)
        lte_processed = Window_preprocessing.LTE_Preprocessing(lte_df)

        result_labels, lte_distances = find_similar_anchors(lte_processed, anchor_file, threshold=0.5)

        if result_labels == -1:
            print(f"[调试] 未找到 LTE 匹配锚点（start={start}）")
            start += step_size
            continue
        else:
            print(f"[调试] LTE 匹配候选锚点: {result_labels},{lte_distances}")

        # 检查 MEG 是否具有锚点特征
        anchor_flag = is_anchor_by_meg(meg, mean_th=0.8, peak_th=2, valley_th=2)
        if anchor_flag != 1:
            print(f"[调试] MEG 特征不满足锚点标准（start={start}）")
            start += step_size
            continue
        else:
            print(f"[调试] MEG 特征满足锚点标准，尝试匹配")

        # 执行 MEG-DTW 匹配
        best_label, _ = match_meg_with_dtw(
            meg_processed, anchor_file, result_labels, lte_distances, step=8, alpha=0.5
        )
        print(f"[成功] 检测到锚点 {best_label}，位置：{end}")
        return best_label, end

        start += step_size

    print("[失败] 所有窗口均未匹配到锚点")
    return None

file_path = "../data/collectionData_02/交叉_小米MAX3_移动卡/20230706_2059_merged.csv"  # 替换为实际路径
data = pd.read_csv(file_path)
anchor_file = "../path_reconstruction/Anchor_features_DBA.csv"
anchor_data = pd.read_csv(anchor_file)


# result = detect_stable_anchor(data, anchor_data, start=0, num_rows=len(data))
#
# if result:
#     best_label, position = result
#     print(f"成功检测到稳定锚点 {best_label}，位置 {position}")
# else:
#     print("未检测到稳定锚点")
