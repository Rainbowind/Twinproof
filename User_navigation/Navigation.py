import matplotlib.pyplot as plt
import pandas as pd
import Window_preprocessing
import matplotlib
import pandas as pd
import ast
import heapq
matplotlib.use("TkAgg")
plt.rcParams['font.family'] = 'SimHei'  # 设置为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

Anchor_path = "../path_reconstruction/Anchor_connection.csv"  # 替换为实际路径
Paths_path = "../path_reconstruction/Paths.csv"  # 替换为实际路径


def load_anchor_graph(anchor_connection_file):
    """加载锚点连接图"""
    df = pd.read_csv(anchor_connection_file)
    graph = {}
    for _, row in df.iterrows():
        node = int(row['Cluster_Label'])
        neighbors = ast.literal_eval(row['Connected_Classes'])
        graph[node] = neighbors
    return graph


def load_paths(paths_file):
    """加载路径信息"""
    df = pd.read_csv(paths_file)
    edge_info = {}

    def reverse_segment(segment):
        """反转路径段并反转方向"""
        reversed_segment = []
        for direction, length in reversed(segment):
            if direction == 1:
                reversed_direction = -1
            elif direction == -1:
                reversed_direction = 1
            else:
                reversed_direction = 0
            reversed_segment.append((reversed_direction, length))
        return reversed_segment

    for _, row in df.iterrows():
        start = int(row['Start_Anchor'])
        end = int(row['End_Anchor'])
        path = ast.literal_eval(row['Path_Situation'])  # [(dir, len)]
        edge_info[(start, end)] = path
        edge_info[(end, start)] = reverse_segment(path)  # 加入反向路径并调整方向

    return edge_info



def dijkstra(graph, edge_info, start, end):
    """执行 Dijkstra 算法"""
    heap = [(0, start, [])]  # (累积距离, 当前节点, 路径)
    visited = set()

    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]
        if node == end:
            return path
        for neighbor in graph.get(node, []):
            if (node, neighbor) in edge_info:
                segment = edge_info[(node, neighbor)]
                seg_length = sum([p[1] for p in segment])
                heapq.heappush(heap, (cost + seg_length, neighbor, path))
    return None  # 没有路径

# 寻路算法
def find_path_with_lengths(start_label, end_label,
                           anchor_connection_file=Anchor_path,
                           paths_file=Paths_path):
    """
    输入起点锚点与终点锚点编号，输出路径点序列与路径段字典。

    返回:
        Tuple[List[int], Dict[Tuple[int, int], List[Tuple[int, float]]]]
    """
    graph = load_anchor_graph(anchor_connection_file)
    edge_info = load_paths(paths_file)
    path = dijkstra(graph, edge_info, start_label, end_label)

    if path is None:
        print("⚠️ 没有从锚点 {} 到锚点 {} 的路径".format(start_label, end_label))
        return None, None

    path_lengths = {}
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        seg = edge_info.get((a, b))
        if seg:
            path_lengths[(a, b)] = seg
        else:
            print(f"⚠️ 缺失路径段: {a} -> {b}")
            return None, None

    return path, path_lengths


# path, path_lengths = find_path_with_lengths(0, 4)
# print("路径节点序列:", path)
# print("路径段详情:")
# for k, v in path_lengths.items():
#     print(f"{k}: {v}")