import matplotlib.pyplot as plt
import pandas as pd
import Window_preprocessing
import matplotlib
import Navigation
import Location
import User_path
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
# 用户情况
user_path = "../data/collectionData_02/交叉_小米MAX3_移动卡/20230706_2104_merged.csv"
sensor_path = "../data/collectionData/交叉_小米MAX3_移动卡/sensor_20230706_2104.csv"

# 读取数据
anchor_data = pd.read_csv(anchor_path)
anchor_connection = pd.read_csv(connection_path)
user = pd.read_csv(user_path)
sensor = pd.read_csv(sensor_path)

# 设定参数
start = 500
window_size = 600
step = 600
end_label = 5  # 目标锚点标签
num_rows = len(user)
current_path = []
path_segments = {}

# 初始定位阶段
while start <= num_rows - window_size:
    if start==0:
        result = Location.detect_stable_anchor(user, anchor_data, start=start, num_rows=len(user))
        time.sleep(1)
    else:
        result = Location.detect_stable_anchor_middle(user, anchor_data, start=start, num_rows=len(user))
        time.sleep(1)

    if not result:
        print("[跟丢] 未检测到稳定锚点，导航失败")
        time.sleep(1)
        break

    best_label, position = result
    print(f"[初始定位] 成功检测到稳定锚点 {best_label}，位置 {position}")
    start=position
    time.sleep(1)

    # 规划路径
    current_path, path_segments = Navigation.find_path_with_lengths(best_label, end_label)
    print("[导航路径] 路径节点序列:", current_path)
    print(path_segments)
    time.sleep(1)

    # 开始路径跟踪循环
    while start <= num_rows - window_size:
        user_segments = User_path.analyze_segment_turns(sensor, start_idx=start, end_idx=start + window_size)
        print("[用户轨迹] 当前窗口:", user_segments)
        time.sleep(1)

        ok, new_segments, new_nodes = User_path.update_path_and_nodes(user_segments, path_segments, current_path)

        if not ok:
            print(start)
            print("[偏离路径] 用户偏离了路径，正在重新定位...")
            time.sleep(1)

            if len(new_nodes) == 0:
                print("[异常] 模糊锚点为空，终止导航")
                time.sleep(1)
                break

            fuzzy_label = new_nodes[0]
            connected_row = anchor_connection[anchor_connection["Cluster_Label"] == fuzzy_label]
            if connected_row.empty:
                print(f"[异常] 未找到模糊锚点 {fuzzy_label} 的拓扑信息，终止导航")
                time.sleep(1)
                break

            connected_classes = ast.literal_eval(connected_row["Connected_Classes"].values[0])
            relevant_labels = connected_classes + [fuzzy_label]

            anchor_data = anchor_data[anchor_data["label"].isin(relevant_labels)]

            start = start - window_size
            break  # 跳出 tracking 循环，回到初始定位
        else:
            print("[正常路径] 用户未偏离路径，继续导航")
            time.sleep(1)
            current_path = new_nodes
            path_segments = new_segments
            start += window_size

            if len(current_path) <= 2:
                print("[完成导航] 用户即将到达目标区域，导航完成")
                time.sleep(1)
                break

