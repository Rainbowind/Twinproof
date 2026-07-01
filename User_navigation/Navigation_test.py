import matplotlib.pyplot as plt
import pandas as pd
import Window_preprocessing
import matplotlib
import Navigation
import Location
import User_path_1
import ast
import heapq
import time  # 加入 sleep 所需的模块

matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 路径设置
anchor_path = "../path_reconstruction/Anchor_features_DBA.csv"
connection_path = "../path_reconstruction/Anchor_connection.csv"
paths_path = "../path_reconstruction/Paths.csv"
user_path = "../data/collectionData_02/交叉_小米MAX3_移动卡/20230709_1315_merged.csv"
sensor_path = "../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230709_1315.csv"

# 读取数据
anchor_data = pd.read_csv(anchor_path)
anchor_connection = pd.read_csv(connection_path)
user = pd.read_csv(user_path)
sensor = pd.read_csv(sensor_path)

# 设定参数1
start = 2100
window_size = 400
start_label=4
end_label = 5        # 目标锚点标签
num_rows = len(user)
current_path = []
path_segments = {}

# 规划路径
current_path, path_segments = Navigation.find_path_with_lengths(start_label, end_label)
print("[导航路径] 路径节点序列:", current_path)
print(path_segments)
time.sleep(1)

# 开始路径跟踪循环
while start <= num_rows - window_size:
    user_segments = User_path_1.analyze_segment_turns(sensor, start_idx=start, end_idx=start + window_size)
    print("[用户轨迹] 当前窗口:", user_segments)
    # time.sleep(2)

    path_segments, current_path, deviated = User_path_1.compare_path_with_trajectory(path_segments, user_segments, current_path)

    if deviated:
        print("[偏离路径] 用户偏离了路径，正在重新定位...")
        # time.sleep(2)

        if len(current_path) == 0:
            print("[异常] 模糊锚点为空，终止导航")
            # time.sleep(2)
            break

        fuzzy_label = current_path[0]
        connected_row = anchor_connection[anchor_connection["Cluster_Label"] == fuzzy_label]
        if connected_row.empty:
            print(f"[异常] 未找到模糊锚点 {fuzzy_label} 的拓扑信息，终止导航")
            # time.sleep(2)
            break

        connected_classes = ast.literal_eval(connected_row["Connected_Classes"].values[0])
        relevant_labels = connected_classes + [fuzzy_label]

        anchor_data = anchor_data[anchor_data["label"].isin(relevant_labels)]

        start = start - window_size
        break  # 跳出 tracking 循环，回到初始定位
    else:
        print("路径信息：",path_segments )
        print("[正常路径] 用户未偏离路径，继续导航")
        # time.sleep(1)
        start += window_size

        finished, error = User_path_1.check_navigation_finished(path_segments)

        if finished:
            print(f"是否到达终点: {finished}, 导航误差: {error} m")
            print(start)
            # time.sleep(2)
            break

