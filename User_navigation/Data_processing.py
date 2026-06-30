import pandas as pd
import pywt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 预处理LTE_ID
def Data_Preprocessing(data):
    # 找出所有 Cell_RSSI 和 Cell_ID 的列名
    rssi_cols = [col for col in data.columns if col.startswith('Cell_RSSI_')]
    id_cols = [col for col in data.columns if col.startswith('Cell_ID_')]

    # 按照 RSSI 和 ID 顺序配对（假设列名结构一致）
    for rssi_col in rssi_cols:
        # 提取编号后缀，例如 '1_3'
        suffix = rssi_col.replace('Cell_RSSI_', '')
        id_col = f'Cell_ID_{suffix}'

        # 如果对应 ID 列存在
        if id_col in data.columns:
            # 当 RSSI 是 NaN，则将对应 ID 设置为 0
            data.loc[data[rssi_col].isna(), id_col] = 0
    return data


# 小波去噪（增强高频）
def wavelet_denoising(signal, wavelet='db6', level=3, threshold_factor=0.5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = threshold_factor * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_denoised, wavelet)
def wavelet_denoising_df(df, wavelet='db6', level=3, threshold_factor=0.5):
    return df.apply(lambda row: wavelet_denoising(row.values, wavelet, level, threshold_factor), axis=1, result_type='expand')

# 标准化数据
def row_zscore_manual(df):
    # 计算每行均值和标准差
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    # 利用广播，逐行做标准化
    df_standardized = df.sub(mean, axis=0).div(std, axis=0)
    return df_standardized


# 对MEG信号进行预处理
def Meg_Preprocessing(meg, meg_front, meg_back,sigma=2.0):
    # 填充空缺值-平滑填充
    meg = meg.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)
    meg_front = meg_front.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)
    meg_back = meg_back.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)

    # 小波去噪
    meg = pd.DataFrame(wavelet_denoising_df(meg))
    meg_front = pd.DataFrame(wavelet_denoising_df(meg_front))
    meg_back = pd.DataFrame(wavelet_denoising_df(meg_back))

    # 高斯滤波处理信号
    meg = pd.DataFrame(gaussian_filter1d(meg, sigma=sigma, axis=1))
    meg_front = pd.DataFrame(gaussian_filter1d(meg_front, sigma, axis=1))
    meg_back = pd.DataFrame(gaussian_filter1d(meg_back, sigma, axis=1))

    meg=pd.DataFrame(meg)
    meg_front=pd.DataFrame(meg_front)
    meg_back=pd.DataFrame(meg_back)

    # 标准化
    meg = row_zscore_manual(meg)*4
    meg_front = row_zscore_manual(meg_front)
    meg_back = row_zscore_manual(meg_back)

    return meg,meg_front,meg_back


# LTE信号预处理
def LTE_Preprocessing(lte):
    # 填充空白值，用0
    lte = lte.apply(pd.to_numeric, errors='coerce').fillna(0)
    # 平滑
    lte = pd.DataFrame(gaussian_filter1d(lte.values, sigma=1.5, axis=1))

    return lte

#
# # 假设数据已经加载并预处理
# data = pd.read_csv('anchor/anchor_combined_局部_小米MAX3.csv')
# columns_meg = data.iloc[:, :400]  # 0到399列的MEG信号数据
# columns_meg_front = data.iloc[:, 400:800]  # 前磁场信号数据
# columns_meg_back = data.iloc[:, 800:1200]  # 后磁场信号数据
#
# meg_pred,_,_=Meg_Preprocessing(columns_meg,columns_meg_front,columns_meg_back)
# print(meg_pred.head())
#
#
# # 提取 LTE 主信号段（拼接）
# columns_lte = pd.concat([
#     data.iloc[:, 1204:1208],
#     data.iloc[:, 1216:1220],
#     data.iloc[:, 1228:1232]
# ], axis=1)
#
# # 提取前后各4列
# columns_lte_front = pd.concat([
#     data.iloc[:, 1200:1204],
#     data.iloc[:, 1212:1216],
#     data.iloc[:, 1224:1228]
# ], axis=1)
#
# columns_lte_back = pd.concat([
#     data.iloc[:, 1208:1212],
#     data.iloc[:, 1220:1224],
#     data.iloc[:, 1232:1236]
# ], axis=1)
#
#
# # # 预处理后数据（已标准化 & 加权）
# # meg_features, meg_front_features, meg_back_features = Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
# # lte_pred=LTE_Preprocessing(columns_lte)
# # lte_front_pred=LTE_Preprocessing(columns_lte_front)
# # lte_back_pred=LTE_Preprocessing(columns_lte_back)
# #
# # all_features = np.hstack([meg_features, meg_front_features, meg_back_features,lte_pred,lte_front_pred,lte_back_pred])
# # pd.DataFrame(all_features).to_csv('output.csv', index=False)
#
#
#
# # 保存原始第一行
# raw_meg_first_row = columns_meg.iloc[0].values
#
# # 预处理
# meg_processed, _, _ = Meg_Preprocessing(columns_meg, columns_meg_front, columns_meg_back)
#
# # 获取处理后第一行
# processed_meg_first_row = meg_processed.iloc[0].values
#
# # ------------------------
# # 绘图
# # ------------------------
# plt.figure(figsize=(12, 6))
#
# # 原始信号
# plt.subplot(2, 1, 1)
# plt.plot(raw_meg_first_row, label='原始 MEG（第1行）', color='blue')
# plt.title('原始 MEG 信号（第1行样本）')
# plt.xlabel('时间点 / 维度索引')
# plt.ylabel('信号强度')
# plt.grid(True)
# plt.legend()
#
# # 预处理后信号
# plt.subplot(2, 1, 2)
# plt.plot(processed_meg_first_row, label='预处理后 MEG（第1行）', color='orange')
# plt.title('预处理后 MEG 信号（第1行样本）')
# plt.xlabel('时间点 / 维度索引')
# plt.ylabel('信号强度（标准化）')
# plt.grid(True)
# plt.legend()
#
# plt.tight_layout()
# plt.show()
