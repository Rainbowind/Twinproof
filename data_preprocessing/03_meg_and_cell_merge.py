import os
import pandas as pd

# 定义原始数据文件夹路径和目标文件夹路径
input_data_folder = '../data/collectionData_new_01'
output_data_folder = '../data/collectionData_new_02'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(output_data_folder):
    os.makedirs(output_data_folder)

# 定义合并数据并填充的方法
def merge_and_fill_sensor_signal(sensor_file, signal_file, output_folder, filename_prefix):
    # 从文件名中提取日期部分，例如 '20230703' 并转换为 '2023-07-03'
    date_str = filename_prefix.split('_')[2]  # 假设日期在文件名的第三部分
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"  # 转换为 'YYYY-MM-DD'
    time_format = "%Y-%m-%d %H:%M:%S:%f"  # 完整的日期时间格式

    # 加载清洗后的 sensor 和 signal 数据
    sensor_data = pd.read_csv(sensor_file)
    signal_data = pd.read_csv(signal_file)

    # 将时间列转换为字符串，以便与格式化后的日期组合
    sensor_data['Time'] = formatted_date + ' ' + sensor_data['Time'].astype(str)
    signal_data['Time'] = formatted_date + ' ' + signal_data['Time'].astype(str)

    # 将合并后的时间列转换为日期时间格式
    sensor_data['Time'] = pd.to_datetime(sensor_data['Time'], format=time_format, errors='coerce')
    signal_data['Time'] = pd.to_datetime(signal_data['Time'], format=time_format, errors='coerce')

    # 去除包含无效时间的数据行
    sensor_data.dropna(subset=['Time'], inplace=True)
    signal_data.dropna(subset=['Time'], inplace=True)

    # 以 sensor 的时间为主进行左连接
    merged_data = pd.merge(sensor_data, signal_data, on='Time', how='left')

    # 使用 `Cell_ID_1`~`Cell_ID_5` 列的第一个非空值填充整个列
    for i in range(1, 6):
        id_col = f'Cell_ID_{i}'
        # 检查列中是否有非空值
        if merged_data[id_col].notna().any():
            first_value = merged_data[id_col].dropna().iloc[0]  # 获取第一个非空值
            merged_data.fillna({id_col: first_value}, inplace=True)  # 用第一个值填充整个列

    # 对每个 `Cell_RSSI_1`~`Cell_RSSI_5` 列进行填充（如有需要）
    for i in range(1, 6):
        rssi_col = f'Cell_RSSI_{i}'
        id_col = f'Cell_ID_{i}'

        # 获取非空的行的索引
        non_null_indices = merged_data[merged_data[rssi_col].notna()].index

        # 对每个非空的单元格向下填充 100 行
        for idx in non_null_indices:
            fill_range = range(idx, min(idx + 100, len(merged_data)))
            merged_data.loc[fill_range, rssi_col] = merged_data.loc[idx, rssi_col]
            merged_data.loc[fill_range, id_col] = merged_data.loc[idx, id_col]

    # 提取所需的日期和时间部分，用于输出文件命名
    date_time_part = "_".join(filename_prefix.split('_')[2:4])  # 提取 '20230703_1734_'

    # 构建输出文件路径
    output_merged_file = os.path.join(output_folder, f'{date_time_part}_merged.csv')

    # 保存合并后的数据到新的CSV文件
    merged_data.to_csv(output_merged_file, index=False)
    print(f'Merged data with filled values saved to {output_merged_file}')

# 递归遍历每个文件夹并处理
def process_and_fill_folder(input_folder, output_folder):
    contents = os.listdir(input_folder)

    for item in contents:
        item_path = os.path.join(input_folder, item)

        if os.path.isdir(item_path):  # 如果是子文件夹，递归调用处理
            output_subfolder = os.path.join(output_folder, item)
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)
            process_and_fill_folder(item_path, output_subfolder)

        elif item.startswith('sensor') and item.endswith('.csv'):  # 如果是 sensor 开头的 CSV 文件
            sensor_file_path = item_path
            signal_file_path = os.path.join(input_folder, item.replace('sensor', 'signal'))

            # 检查对应的 signal 文件是否存在
            if os.path.exists(signal_file_path):
                # 获取文件名前缀用于输出文件命名
                filename_prefix = os.path.splitext(item)[0]

                # 合并并填充 sensor 和 signal 数据
                merge_and_fill_sensor_signal(sensor_file_path, signal_file_path, output_folder, filename_prefix)

# 开始处理并填充文件夹中的数据
process_and_fill_folder(input_data_folder, output_data_folder)
