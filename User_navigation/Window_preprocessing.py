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
def Meg_Preprocessing(meg,sigma=2.0):
    # 填充空缺值-平滑填充
    meg = meg.apply(pd.to_numeric, errors='coerce', axis=1).ffill(axis=1).bfill(axis=1)

    # 小波去噪
    meg = pd.DataFrame(wavelet_denoising_df(meg))

    # 高斯滤波处理信号
    meg = pd.DataFrame(gaussian_filter1d(meg, sigma=sigma, axis=1))
    meg=pd.DataFrame(meg)

    # 标准化
    meg = row_zscore_manual(meg)

    return meg


# LTE信号预处理
def LTE_Preprocessing(lte):
    # 填充空白值，用0
    lte = lte.apply(pd.to_numeric, errors='coerce').fillna(0)
    # 平滑
    lte = pd.DataFrame(gaussian_filter1d(lte.values, sigma=1.5, axis=1))
    return lte
