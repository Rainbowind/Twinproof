import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# 定义文件夹路径
folder_path = "../data/collectionData_01/局部_小米MAX3"  # 这里改成你的文件夹路径

# 绘制 sensor 数据的图表
def plot_sensor_data(sensor_data, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(sensor_data['Time'], sensor_data['Meg'], label='Meg', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Meg')
    plt.title(f'Sensor Data: Meg over Time\n{file_name}')

    # 设置较少的 xticks
    plt.xticks(sensor_data['Time'][::max(1, len(sensor_data)//10)], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 遍历文件夹中的所有 sensor 文件
for file in os.listdir(folder_path):
    if file.endswith('.csv') and file.startswith('sensor'):
        file_path = os.path.join(folder_path, file)
        print(f"处理文件: {file_path}")

        # 加载预处理后的数据
        sensor_data = pd.read_csv(file_path)

        # 时间格式处理
        sensor_data['Time'] = pd.to_datetime(sensor_data['Time'], format="%H:%M:%S:%f", errors='coerce')
        sensor_data.dropna(subset=['Time'], inplace=True)

        # 绘图
        plot_sensor_data(sensor_data, file)
