import matplotlib.pyplot as plt
import ast
import math
import pandas as pd
from collections import deque
import matplotlib
matplotlib.use("TkAgg")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1.读取数据
df = pd.read_csv('Paths.csv')

# 2.构建图（正向 + 反向路径）
def reverse_path_situation(path_situation):
    reversed_path = []
    for direction, length in reversed(path_situation):
        new_dir = direction if direction == 0 else -direction
        reversed_path.append((new_dir, length))
    return reversed_path

graph = {}
for _, row in df.iterrows():
    start, end = row['Start_Anchor'], row['End_Anchor']
    path = ast.literal_eval(row['Path_Situation'])
    graph.setdefault(start, []).append((end, path))
    graph.setdefault(end, []).append((start, reverse_path_situation(path)))

# 3.计算路径坐标
def get_path_coordinates(start_point, path_situation, initial_direction=0):
    coords = [start_point]
    x, y = start_point
    direction = initial_direction  # 初始方向

    # 遍历路径上的每个步骤，更新方向并计算新的坐标
    for turn, length in path_situation:
        if turn == 1:  # 右转
            direction -= 90
        elif turn == -1:  # 左转
            direction += 90
        direction %= 360  # 保证方向在0到360度之间

        rad = math.radians(direction)
        x += length * math.cos(rad)
        y += length * math.sin(rad)
        coords.append((x, y))

    return coords, direction  # 返回路径点与最终方向

# 4.绘制完整路径图
def draw_all_paths(graph, start_id=0):
    plt.figure(figsize=(10, 10))
    plt.title("锚点路径网络")
    plt.axis('equal')

    positions = {}
    orientations = {}
    visited_edges = set()
    visited_nodes = set()

    positions[start_id] = (0, 0)
    orientations[start_id] = 0
    queue = deque()
    queue.append((start_id, (0, 0)))

    while queue:
        current_id, current_pos = queue.popleft()
        current_dir = orientations[current_id]
        visited_nodes.add(current_id)

        for neighbor_id, path_situation in graph.get(current_id, []):
            edge_key = tuple(sorted((current_id, neighbor_id)))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            coords, final_dir = get_path_coordinates(current_pos, path_situation, current_dir)
            end_pos = coords[-1]

            if neighbor_id not in positions:
                positions[neighbor_id] = end_pos
                orientations[neighbor_id] = final_dir
                queue.append((neighbor_id, end_pos))

            # 绘图
            xs, ys = zip(*coords)
            plt.plot(xs, ys, 'b')
            plt.plot(xs[0], ys[0], 'ro')  # 起点
            plt.plot(xs[-1], ys[-1], 'bo')  # 终点
            plt.text(xs[0], ys[0], str(current_id), fontsize=9, ha='right', va='bottom')
            plt.text(xs[-1], ys[-1], str(neighbor_id), fontsize=9, ha='left', va='top')

    # 补充孤立点
    for anchor in graph.keys():
        if anchor not in positions:
            print(f"未连接的孤立点：{anchor}")
            positions[anchor] = (9999, 9999)
            plt.plot(9999, 9999, 'ko')
            plt.text(9999, 9999, str(anchor), fontsize=9)

    # plt.grid(True)
    plt.show()

# 5.执行绘图
draw_all_paths(graph, start_id=0)
