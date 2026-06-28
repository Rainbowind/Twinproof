import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict
import numpy as np
import os
import matplotlib
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import timedelta

folder_path = '../data/collectionData_02/局部_小米MAX3'
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

window_size = 400
step_size = 10

all_segment_data = []
anchor_points_info = []

def find_anchor(data, window_size=400, step_size=10, mode='or'):
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data['Time'] = (data['Time'] - data['Time'].min()).dt.total_seconds()

    for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']:
        data[col] = data[col].ffill().bfill()
        data[col] = data[col].rolling(window=5, min_periods=1).mean()

    for col in ['Cell_ID_1', 'Cell_ID_2', 'Cell_ID_3']:
        data[col] = data[col].ffill().bfill()

    data['Meg_diff1'] = data['Meg'].diff().rolling(window=5, min_periods=1).mean()
    scaler = StandardScaler()
    data['Meg_diff1_scaler'] = scaler.fit_transform(data[['Meg_diff1']])
    data['Meg_diff1_abs'] = np.abs(data['Meg_diff1_scaler'])

    mean_value = data['Meg_diff1_abs'].mean()
    std_value = data['Meg_diff1_abs'].std()
    upper_bound_meg = mean_value + 0.08 * std_value

    peaks, _ = find_peaks(data['Meg_diff1_scaler'], distance=50)
    valleys, _ = find_peaks(-data['Meg_diff1_scaler'], distance=50)
    upper_bound_peaks = data['Meg_diff1_scaler'].iloc[peaks].mean() + 0.6 * data['Meg_diff1_scaler'].iloc[peaks].std()
    under_bound_valley = data['Meg_diff1_scaler'].iloc[valleys].mean() - data['Meg_diff1_scaler'].iloc[valleys].std()

    anchor_points = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        window = data.iloc[start:end]

        mean_meg_diff1 = window['Meg_diff1_abs'].mean()
        peak_values = window['Meg_diff1_scaler'].iloc[find_peaks(window['Meg_diff1_scaler'], distance=50)[0]]
        valley_values = window['Meg_diff1_scaler'].iloc[find_peaks(-window['Meg_diff1_scaler'], distance=50)[0]]
        max_peak = peak_values.max() if len(peak_values) > 0 else 0
        min_valley = valley_values.min() if len(valley_values) > 0 else 0
        rolling_std = window['Meg_diff1_scaler'].rolling(window=20, min_periods=1).std().mean()
        # skewness_val = skew(window['Meg_diff1_scaler'].dropna())

        lte_rssi_std = np.mean([
            window['Cell_RSSI_1'].std(),
            window['Cell_RSSI_2'].std(),
            window['Cell_RSSI_3'].std()
        ])

        id_changes = 0
        for col in ['Cell_ID_1', 'Cell_ID_2', 'Cell_ID_3']:
            ids = window[col].fillna(-1).astype(int).values
            id_changes += np.count_nonzero(np.diff(ids) != 0)
        lte_id_change_rate = id_changes / (3 * (window_size - 1))

        meg_flag = (mean_meg_diff1 >= upper_bound_meg and max_peak >= upper_bound_peaks
                    and min_valley <= under_bound_valley and rolling_std >= 0.08)
        lte_flag = (lte_rssi_std > 2.0 or lte_id_change_rate > 0.02)

        if (mode == 'or' and (meg_flag or lte_flag)) or (mode == 'and' and (meg_flag and lte_flag)):
            anchor_points.append((start, end))

    merged_segments = []
    for seg in sorted(anchor_points):
        if not merged_segments or merged_segments[-1][1] < seg[0]:
            merged_segments.append(seg)
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], seg[1]))

    final_segments = []
    for seg in merged_segments:
        if seg[1] - seg[0] > window_size:
            max_index = data['Meg_diff1_abs'].iloc[seg[0]:seg[1]].idxmax()
            new_start = max(max_index - window_size // 2, seg[0])
            new_end = min(new_start + window_size, seg[1])
            final_segments.append((new_start, new_end))
        else:
            final_segments.append(seg)

    filtered_segments = []
    i = 0
    while i < len(final_segments) - 1:
        if final_segments[i + 1][0] - final_segments[i][1] <= 600:
            combined_start = final_segments[i][0]
            combined_end = final_segments[i + 1][1]
            max_index = data['Meg_diff1_abs'].iloc[combined_start:combined_end].idxmax()
            new_start = max(max_index - window_size // 2, combined_start)
            new_end = min(new_start + window_size, combined_end)
            filtered_segments.append((new_start, new_end))
            i += 2
        else:
            filtered_segments.append(final_segments[i])
            i += 1
    if i == len(final_segments) - 1:
        filtered_segments.append(final_segments[i])

    return filtered_segments

def plot_signal_with_anchor_points(data, filtered_segments, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data['Meg'], label='Meg Signal', color='blue')
    for start, end in filtered_segments:
        plt.axvspan(data['Time'].iloc[start], data['Time'].iloc[end], color='red', alpha=0.5)
    plt.title(f'Meg Signal and Anchor Points - {file_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Meg Signal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_lte_with_anchor_points(data, filtered_segments, file_name):
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue']
    lte_cols = ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']

    for idx, col in enumerate(lte_cols):
        if col in data.columns:
            plt.plot(data['Time'], data[col], label=col, color=colors[idx], alpha=0.7)

    for start, end in filtered_segments:
        plt.axvspan(data['Time'].iloc[start], data['Time'].iloc[end], color='yellow', alpha=0.3, label='Anchor Region')

    plt.title(f'LTE RSSI and Anchor Points - {file_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('RSSI (dBm)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    data = pd.read_csv(file_path)
    data_copy=data.copy(deep=True)   # 复制数据用于存储
    filtered_segments = find_anchor(data_copy)

    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    data['Time'] = (data['Time'] - data['Time'].min()).dt.total_seconds()

    # ✅ 输出每个文件的锚点段位置
    print(f"{file_name} 锚点段：{[(s, e) for s, e in filtered_segments]}")
    plot_signal_with_anchor_points(data, filtered_segments, file_name)
    # plot_lte_with_anchor_points(data, filtered_segments, file_name)

    for seg_start, seg_end in filtered_segments:
        segment = data['Meg'].iloc[seg_start:seg_end].values[:400]
        if len(segment) < 400:
            segment = list(segment) + [None] * (400 - len(segment))

        pre_anchor_segment = data['Meg'].iloc[max(0, seg_start - 400):seg_start].values[-400:]
        if len(pre_anchor_segment) < 400:
            pre_anchor_segment = [None] * (400 - len(pre_anchor_segment)) + list(pre_anchor_segment)

        post_anchor_segment = data['Meg'].iloc[seg_end:seg_end + 400].values[:400]
        if len(post_anchor_segment) < 400:
            post_anchor_segment = list(post_anchor_segment) + [None] * (400 - len(post_anchor_segment))

        lte_segments = []
        for col in ['Cell_RSSI_1', 'Cell_RSSI_2', 'Cell_RSSI_3']:
            lte_segment = data[col].iloc[max(0, seg_start - 4):min(len(data), seg_end + 4)].values[:12]
            if len(lte_segment) < 12:
                lte_segment = list(lte_segment) + [None] * (12 - len(lte_segment))
            lte_segments.extend(lte_segment)

        lte_id_segments = []
        for col in ['Cell_ID_1', 'Cell_ID_2', 'Cell_ID_3']:
            id_segment = data[col].iloc[max(0, seg_start - 4):min(len(data), seg_end + 4)].values[:12]
            if len(id_segment) < 12:
                id_segment = list(id_segment) + [None] * (12 - len(id_segment))
            lte_id_segments.extend(id_segment)

        start_time = data['Time'].iloc[seg_start]
        end_time = data['Time'].iloc[seg_end - 1]
        mid_time = (seg_start + seg_end) // 2
        mid_time_str = pd.to_datetime(data['Time'].iloc[mid_time], unit='s').strftime('%H:%M:%S')

        combined_segment = (
            list(segment)
            + list(pre_anchor_segment)
            + list(post_anchor_segment)
            + lte_segments
            + lte_id_segments
        )
        combined_segment.append(file_name)
        combined_segment.append(mid_time_str)
        all_segment_data.append(combined_segment)
        anchor_points_info.append((file_name, seg_start, seg_end))


columns = (
    [f'Column_{i + 1}' for i in range(400)]
    + [f'Pre_Anchor_{i + 1}' for i in range(400)]
    + [f'Post_Anchor_{i + 1}' for i in range(400)]
    + [f'Cell_RSSI_1_{i + 1}' for i in range(12)]
    + [f'Cell_RSSI_2_{i + 1}' for i in range(12)]
    + [f'Cell_RSSI_3_{i + 1}' for i in range(12)]
    + [f'Cell_ID_1_{i + 1}' for i in range(12)]
    + [f'Cell_ID_2_{i + 1}' for i in range(12)]
    + [f'Cell_ID_3_{i + 1}' for i in range(12)]
    + ['File_Name']
    + ['Mid_Time']
)

segment_df = pd.DataFrame(all_segment_data, columns=columns)
anchor_info_df = pd.DataFrame(anchor_points_info, columns=['File', 'Start_Index', 'End_Index'])
segment_df['Anchor_Info'] = anchor_info_df.apply(lambda x: f"({x['Start_Index']},{x['End_Index']})", axis=1)

# output_file_path = 'anchor/anchor_combined_vivox100_Path12.csv'
# segment_df.to_csv(output_file_path, index=False)
#
# print(f"\n锚点段保存完成！文件路径：{output_file_path}")
