import ast
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import PDR_Path
import Same

# 读取路径信息
# file_path = "../data/collectionData/局部_小米MAX3/sensor_20230705_1306.csv"
# df = pd.read_csv(file_path)
# Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_局部_小米MAX3_refined.csv')
# file_path = "../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230706_2104.csv"
# df = pd.read_csv(file_path)
# Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_交叉_小米MAX3_移动卡_refined.csv')
file_path = "../data/collectionData_new/vivos9_path3/sensor_20251127_1114.csv"
df = pd.read_csv(file_path)
Anchor_data = pd.read_csv('../Find_Anchor/anchor_cluster/anchor_cluster_vivos9_path3_refined.csv')



# 1.保存锚点间的连接关系
def Anchor_into_csv(data, csv_path='Anchor_connection.csv', feature_folder='Anchor_feature_parking'):
    """
    功能：
    1. 在 Anchor_connection 中保存锚点和与之关联锚点 (Cluster_Label, Connected_Classes)。
    2. 将 Anchor_data 另存为 csv 文件，放在 Anchor_feature_parking 文件夹中，用序号命名。
    3. 在 Anchor_connection 中保存每个锚点对应的从属文件路径 (feature_csv)。
    4. 自动修改 Anchor_data 中的 Cluster_Label，使其与 Anchor_connection 中的编号一致。
    """
    # --------- Step 0: 准备工作 ---------
    os.makedirs(feature_folder, exist_ok=True)  # 确保特征文件夹存在

    # 确定 Anchor_connection.csv 中现有最大 Cluster_Label
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        if 'Cluster_Label' in existing_data.columns:
            max_existing_label = existing_data['Cluster_Label'].max() + 1
        else:
            max_existing_label = 0
    else:
        max_existing_label = 0

    # --------- Step 1: 更新 Anchor_data 的 Cluster_Label ---------
    # 获取当前数据中的类别标签
    unique_labels = sorted(data['Sorted_Cluster_Label'].unique())
    label_mapping = {old: new for new, old in enumerate(unique_labels, start=max_existing_label)}
    data['Cluster_Label'] = data['Sorted_Cluster_Label'].map(label_mapping)

    # --------- Step 2: 保存 Anchor_data 特征文件 ---------
    feature_files = {}
    for label in sorted(data['Cluster_Label'].unique()):
        subset = data[data['Cluster_Label'] == label]
        feature_file = os.path.join(feature_folder, f"anchor_feature_{label}.csv")
        subset.to_csv(feature_file, index=False)
        feature_files[label] = feature_file

    # --------- Step 3: 构建 Anchor_connection 表 ---------
    cluster_labels = sorted(data['Cluster_Label'].unique())
    connected_classes = []
    feature_csvs = []

    for i, label in enumerate(cluster_labels):
        connected = []
        # 简单假设：当前锚点只和前后锚点相连
        if i > 0:
            connected.append(cluster_labels[i-1])
        if i < len(cluster_labels)-1:
            connected.append(cluster_labels[i+1])

        connected_classes.append(connected)
        feature_csvs.append(feature_files[label])

    connection_df = pd.DataFrame({
        'Cluster_Label': cluster_labels,
        'Connected_Classes': connected_classes,
        'feature_csv': feature_csvs
    })

    # --------- Step 4: 保存 Anchor_connection ---------
    if os.path.exists(csv_path):
        connection_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        connection_df.to_csv(csv_path, index=False)

    print(f"Anchor_connection 已更新：{csv_path}")
    print(f"特征文件已保存到：{feature_folder}")
    return max_existing_label, label_mapping


# 2.保存路径
def Path_into_csv(paths, csv_path='Paths.csv', index=0):
    # 准备保存数据
    path_data = []
    # 读取现有文件以获取当前最大 Path_ID
    if os.path.exists(csv_path):
        existing_data = pd.read_csv(csv_path)
        if 'Path_ID' in existing_data.columns:
            max_existing_id = existing_data['Path_ID'].max()  # 获取现有最大 Path_ID
        else:
            max_existing_id = -1  # 如果没有路径数据，ID从0开始
    else:
        max_existing_id = -1  # 如果文件不存在，ID从0开始

    # 遍历路径数据并生成新的 Path_ID
    for path in paths:
        start_end = path[0]  # 起点和终点
        start_anchor = start_end[0] + index  # 为锚点编号加上 index
        end_anchor = start_end[1] + index

        # 计算路径长度
        path_length = sum([segment[1] for segment in path[1:]])

        # 获取路径情况（去掉第一个元组）
        path_situation = path[1:]

        # 保存路径信息
        path_data.append([max_existing_id + 1, start_anchor, end_anchor, path_length, path_situation])
        max_existing_id += 1  # 更新 Path_ID

    # 将路径数据保存到 CSV 文件
    path_data_df = pd.DataFrame(path_data,
                                columns=['Path_ID', 'Start_Anchor', 'End_Anchor', 'Path_Length', 'Path_Situation'])

    # 如果文件已存在，则追加数据
    if os.path.exists(csv_path):
        path_data_df.to_csv(csv_path, mode='a', header=False, index=False)  # 追加模式，不写入表头
    else:
        path_data_df.to_csv(csv_path, index=False)  # 第一次写入时写表头

    print(f"路径数据已保存到 {csv_path}")


# 3.整合相同的锚点，继承不同锚点的连接关系，同时修改路径对应的锚点编号
def merge_anchor_features(feature_folder, old_idx, new_idx):
    """
    将 new_idx 对应的 anchor_feature 文件合并到 old_idx 文件中。
    """
    old_file = os.path.join(feature_folder, f"anchor_feature_{old_idx}.csv")
    new_file = os.path.join(feature_folder, f"anchor_feature_{new_idx}.csv")

    if not (os.path.exists(old_file) and os.path.exists(new_file)):
        print(f"❌ 锚点文件不存在: {old_file} 或 {new_file}")
        return

    df_old = pd.read_csv(old_file)
    df_new = pd.read_csv(new_file)
    df_merged = pd.concat([df_old, df_new], ignore_index=True)

    df_merged.to_csv(old_file, index=False)
    os.remove(new_file)  # 删除新文件
    print(f"✅ 合并 {new_file} 到 {old_file} 并删除 {new_file}")

def update_anchor_labels(anchor_connection, path, feature_folder, new_index, k=3.0, alpha=0.5):
    """
    根据 same() 匹配结果，合并锚点并更新：
    1. anchor_connection.csv
    2. path.csv
    3. feature_folder 中的 anchor_feature_xx.csv 文件
    """
    # ========== 1. 读取 anchor_connection.csv ==========
    data = pd.read_csv(anchor_connection)
    if isinstance(data['Connected_Classes'].iloc[0], str):
        data['Connected_Classes'] = data['Connected_Classes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # ========== 2. 获取匹配对 ==========
    pairs = Same.same(feature_folder, new_index, k=k, alpha=alpha)
    pairs=[[16,19],[17,21]]
    if not pairs:
        print("⚠ 没有检测到可合并的锚点。")
        return

    label_changes = {}

    # ========== 3. 按顺序合并匹配对 ==========
    for old_label, new_label in pairs:
        if new_label in label_changes:  # 避免重复合并
            continue
        label_changes[new_label] = old_label
        print(f"🔗 合并新锚点 {new_label} -> 旧锚点 {old_label}")

        # === (2) 合并 feature_csv 文件 ===
        old_path = os.path.join(feature_folder, f"anchor_feature_{old_label}.csv")
        new_path = os.path.join(feature_folder, f"anchor_feature_{new_label}.csv")
        if os.path.exists(new_path):
            df_old = pd.read_csv(old_path) if os.path.exists(old_path) else pd.DataFrame()
            df_new = pd.read_csv(new_path)
            pd.concat([df_old, df_new], ignore_index=True).to_csv(old_path, index=False)
            os.remove(new_path)
            print(f"  🗂 已合并 feature 文件 {new_label} -> {old_label}")

    # ========== 4. 更新 Cluster_Label ==========
    data['Cluster_Label'] = data['Cluster_Label'].replace(label_changes)

    # ========== 5. 更新 Connected_Classes ==========
    def update_connections(classes):
        updated = [label_changes.get(cls, cls) for cls in classes]
        return sorted(set(updated))  # 去重

    data['Connected_Classes'] = data['Connected_Classes'].apply(update_connections)

    # === (3) 合并重复 Cluster_Label 行 ===
    merged_rows = []
    for label in sorted(data['Cluster_Label'].unique()):
        same_rows = data[data['Cluster_Label'] == label]
        if same_rows.shape[0] > 1:
            all_connections = set()
            feature_csv = os.path.join(feature_folder, f"anchor_feature_{label}.csv")
            for _, row in same_rows.iterrows():
                all_connections.update(row['Connected_Classes'])
            if label in all_connections:
                all_connections.remove(label)
            merged_rows.append({
                "Cluster_Label": label,
                "Connected_Classes": sorted(all_connections),
                "feature_csv": feature_csv
            })
        else:
            merged_rows.append(same_rows.iloc[0].to_dict())
    data = pd.DataFrame(merged_rows)

    # ========== 6. 更新 path.csv ==========
    path_data = pd.read_csv(path)
    def update_anchor(anchor_id):
        return label_changes.get(anchor_id, anchor_id)
    path_data['Start_Anchor'] = path_data['Start_Anchor'].apply(update_anchor)
    path_data['End_Anchor'] = path_data['End_Anchor'].apply(update_anchor)

    # ========== 7. 保存 ==========
    data.to_csv(anchor_connection, index=False)
    path_data.to_csv(path, index=False)
    print(f"✅ 已更新 {anchor_connection} 和 {path} 中的锚点编号")
    print(f"最终合并映射表: {label_changes}")


# 调用PDR算法，展示路径
if __name__ == "__main__":
    median_start_colum,turn_info,paths=PDR_Path.PDR(df, Anchor_data, If_show=True)
    max_existing_label, label_mapping=Anchor_into_csv(Anchor_data,'Anchor_connection.csv','Anchor_feature_parking')
    Path_into_csv(paths,csv_path='Paths.csv', index=max_existing_label)

    # 相同位置锚点合并
    # update_anchor_labels('Anchor_connection.csv','Paths.csv',"Anchor_feature_parking",max_existing_label)