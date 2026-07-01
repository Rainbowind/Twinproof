import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import Window_preprocessing
import Location
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ========== 1. 读取 CSV ==========
file_path = "../data/collectionData_02/局部_小米MAX3/20230705_1303_merged.csv"  # 替换为实际路径
data = pd.read_csv(file_path)
anchor_file = "../path_reconstruction/Anchor_features_DBA.csv"
anchor_data=pd.read_csv(anchor_file)

all_meg_processed = []
all_lte_processed = []

window_size = 400
step_size = 20
num_rows = len(data)



start = 0
while start <= num_rows - window_size:
    end = start + window_size

    # 提取MEG信号和lte信号，并预处理
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

    result_labels, lte_distances = Location.find_similar_anchors(lte_processed, anchor_data, threshold=0.05)

    if result_labels == -1:
        start += step_size
        continue
    else:
        anchor_flag = Location.is_anchor_by_meg(meg)

    if anchor_flag == 1:
        best_label,_ = Location.match_meg_with_dtw(
            meg_processed, anchor_data, result_labels, lte_distances, step=8, alpha=0.5
        )

        print(f"最相似的 MEG 锚点标签: {best_label}")
        print(start+400)
        start += 20  # 成功匹配，跳过600
    else:
        start += step_size  # 不是锚点，继续正常滑动
