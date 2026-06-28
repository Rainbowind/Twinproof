import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('anchor_cluster/anchor_cluster_vivos9_path3_refined.csv')

# 解析 'Anchor_Info' 列，将其拆分成 start 和 end 两列
data[['start', 'end']] = data['Anchor_Info'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)

# 计算区间长度
data['length'] = data['end'] - data['start']

# 获取各类别数据量
category_counts = data['Cluster_Label'].value_counts()

# 只保留样本数 >= 1 的类别
valid_categories = category_counts[category_counts > 2].index.tolist()
data = data[data['Cluster_Label'].isin(valid_categories)]

# 存储处理后的数据
filtered_dfs = []

# 逐类别处理
for label in data['Cluster_Label'].unique():
    subset = data[data['Cluster_Label'] == label].copy()

    # 仅对数量 > 2 的类别进行异常值检测
    if category_counts[label] > 2:
        q1, q3 = subset['start'].quantile([0.2  , 0.8])
        iqr = q3 - q1
        lower_bound = q1 - 0.5 * iqr
        upper_bound = q3 + 0.5 * iqr

        subset = subset[(subset['start'] >= lower_bound) & (subset['start'] <= upper_bound)]

    filtered_dfs.append(subset)

# 合并数据  
filtered_data = pd.concat(filtered_dfs).reset_index(drop=True)

# **根据 'Anchor_Info' 排序并按照 'start' 的值为每个类别重新编号**
# 这里通过先按 `start` 对每个类别进行排序，并重新赋予一个连续的编号
category_order = filtered_data.groupby('Cluster_Label')['start'].median().sort_values().index.tolist()

# **生成新的连续编号，按照排序后的顺序**
new_labels = {old_label: new_label for new_label, old_label in enumerate(category_order)}

# **将 Cluster_Label 重新赋值为新编号**
filtered_data['Cluster_Label'] = filtered_data['Cluster_Label'].map(new_labels)

# 删除辅助列
filtered_data = filtered_data.drop(columns=['start', 'end', 'length'])

# **按新的 Cluster_Label 排序，确保保存时是按顺序的**
filtered_data = filtered_data.sort_values(by='Cluster_Label').reset_index(drop=True)

# 保存清理后的数据
filtered_data.to_csv('anchor_cluster/anchor_cluster_vivos9_path3_refined.csv', index=False)
print("处理完成，数据已保存。")
