import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def parse_list(value):
    """把 CSV 中保存的列表字符串解析成 Python list，并去掉连续重复项。"""
    if isinstance(value, list):
        items = value
    elif pd.isna(value):
        items = []
    else:
        text = str(value).strip()
        if not text:
            items = []
        else:
            try:
                items = json.loads(text)
            except json.JSONDecodeError:
                items = ast.literal_eval(text)

    cleaned = []
    for item in items:
        if item is None:
            continue
        item = str(item).strip()
        if not item:
            continue
        if cleaned and cleaned[-1] == item:
            continue
        cleaned.append(item)
    return cleaned


def claim_path_to_file_name(claim_path):
    """把 Claim_Path 转成 merged 文件名，用于匹配全局节点表中的 File_Name。"""
    stem = str(claim_path).strip().replace("\\", "/").split("/")[-1]
    if stem.endswith("_merged.csv"):
        return stem
    if stem.endswith("_merged"):
        return f"{stem}.csv"
    return f"{stem}_merged.csv"


def parse_connected(value):
    """解析 Anchor_connection.csv 中的 Connected_Classes 字段。"""
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in ast.literal_eval(text)]


def parse_regions(value):
    """解析节点对应的区域约束；交点节点会返回多个候选区域。"""
    if pd.isna(value):
        return []
    return [region.strip() for region in str(value).split(",") if region.strip()]


def load_anchor_connection(path):
    """读取地图拓扑表，返回节点连通图和节点到区域的映射。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    graph = {}
    node_regions = {}

    for _, row in df.iterrows():
        node = int(row["Cluster_Label"])
        graph[node] = set(parse_connected(row["Connected_Classes"]))
        node_regions[node] = parse_regions(row["Environment_Constraint"])

    return graph, node_regions


def load_detected_node_sequences(global_anchor_table):
    """读取带全局编号的锚点表，按 File_Name 聚合成真实检测节点序列。"""
    df = pd.read_csv(global_anchor_table, encoding="utf-8-sig")
    required = {"File_Name", "Global_Anchor_ID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Global anchor table is missing columns: {sorted(missing)}")

    sequences = {}
    for file_name, group in df.groupby("File_Name", sort=False):
        nodes = []
        for value in group["Global_Anchor_ID"].dropna():
            node = int(value)
            # 连续重复节点不提供新的拓扑移动信息，压缩掉。
            if nodes and nodes[-1] == node:
                continue
            nodes.append(node)
        sequences[str(file_name)] = nodes
    return sequences


def shortest_path(start, end, graph):
    """用 BFS 查找两个全局节点之间的最短可达路径。"""
    if start == end:
        return [start]
    if start not in graph or end not in graph:
        return []

    queue = [(start, [start])]
    visited = {start}
    while queue:
        node, path = queue.pop(0)
        for neighbor in graph.get(node, set()):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == end:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))
    return []


def expand_reachable_path(node_sequence, graph):
    """把检测到的节点序列展开为地图中实际可达的节点路径。"""
    if not node_sequence:
        return []
    if len(node_sequence) == 1:
        return list(node_sequence)

    expanded = [node_sequence[0]]
    for current_node, next_node in zip(node_sequence[:-1], node_sequence[1:]):
        # 检测序列中的相邻节点不一定直接相连，允许通过地图最短路径连接。
        path = shortest_path(current_node, next_node, graph)
        if not path:
            return []
        expanded.extend(path[1:])
    return expanded


def nodes_to_region_candidates(node_sequence, node_regions):
    """把全局节点路径转成区域候选序列。一个节点可能对应多个区域。"""
    return [node_regions.get(node, []) for node in node_sequence]


def lcs_region_candidates(claim_trace, detected_region_candidates):
    """计算支持多候选区域匹配的最长公共子序列长度。"""
    if not claim_trace or not detected_region_candidates:
        return 0

    previous = [0] * (len(detected_region_candidates) + 1)
    for claim_region in claim_trace:
        current = [0] * (len(detected_region_candidates) + 1)
        for index, candidates in enumerate(detected_region_candidates, start=1):
            if claim_region in candidates:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def sequence_similarity(claim_trace, detected_region_candidates):
    """时序相似性：比较声明区域序列和检测区域候选序列的顺序一致程度。"""
    denominator = max(len(claim_trace), len(detected_region_candidates))
    if denominator == 0:
        return 0.0
    return lcs_region_candidates(claim_trace, detected_region_candidates) / denominator


def whole_similarity(claim_trace, detected_region_candidates):
    """整体相似性：比较声明经过的区域集合和检测区域集合的重合程度。"""
    claim_regions = set(claim_trace)
    detected_regions = {
        region
        for candidates in detected_region_candidates
        for region in candidates
    }

    union = claim_regions | detected_regions
    if not union:
        return 0.0
    return len(claim_regions & detected_regions) / len(union)


def score_claim_row(row, detected_sequences, graph, node_regions):
    """计算单条攻击样本的拓扑一致性得分。"""
    claim_trace = parse_list(row.get("Claim_Trace"))
    file_name = claim_path_to_file_name(row.get("Claim_Path"))

    # 通过 Claim_Path 对应的 merged 文件名，在全局节点表中找到真实检测节点序列。
    detected_nodes = detected_sequences.get(file_name, [])

    # 可达性是硬门槛：不可达或没有检测节点时，直接置为攻击。
    reachable_node_path = expand_reachable_path(detected_nodes, graph)
    reach_flag = 1 if reachable_node_path else -1

    # 只有可达路径才会被转换成区域候选序列，用于和 Claim_Trace 比较。
    detected_region_candidates = nodes_to_region_candidates(reachable_node_path, node_regions)

    if reach_flag == -1:
        s_seq = 0.0
        s_whole = 0.0
        s_topo = 0.0
        decision = "attack_no_detected_nodes" if not detected_nodes else "attack_unreachable"
    else:
        s_seq = sequence_similarity(claim_trace, detected_region_candidates)
        s_whole = whole_similarity(claim_trace, detected_region_candidates)
        # 环境一致性由时序相似性和整体相似性各占 0.5 得到。
        s_topo = 0.5 * s_seq + 0.5 * s_whole
        decision = "reachable_scored"

    output = row.to_dict()
    output.update({
        "Matched_File_Name": file_name,
        "Detected_Global_Node_Seq": json.dumps(detected_nodes, ensure_ascii=False),
        "Reachable_Global_Node_Path": json.dumps(reachable_node_path, ensure_ascii=False),
        "Detected_Region_Candidates": json.dumps(detected_region_candidates, ensure_ascii=False),
        "Reach_Flag": reach_flag,
        "S_seq": round(s_seq, 6),
        "S_whole": round(s_whole, 6),
        "S_topo": round(s_topo, 6),
        "Topo_Decision": decision,
    })
    return output


def summarize(scored):
    """汇总当前攻击数据集的拓扑一致性计算结果。"""
    return pd.DataFrame([{
        "Sample_Count": len(scored),
        "Reachable_Count": int((scored["Reach_Flag"] == 1).sum()),
        "Unreachable_Count": int((scored["Reach_Flag"] == -1).sum()),
        "Mean_S_seq": round(scored["S_seq"].mean(), 6),
        "Mean_S_whole": round(scored["S_whole"].mean(), 6),
        "Mean_S_topo": round(scored["S_topo"].mean(), 6),
        "Median_S_topo": round(scored["S_topo"].median(), 6),
        "Min_S_topo": round(scored["S_topo"].min(), 6),
        "Max_S_topo": round(scored["S_topo"].max(), 6),
    }])


def default_global_anchor_table(project_root):
    """自动选择 anchor_with_global_id 目录下的全局编号表，避免中文路径硬编码。"""
    table_dir = project_root / "Claim_Detection" / "anchor_with_global_id"
    tables = sorted(table_dir.glob("*_global.csv"))
    if not tables:
        raise FileNotFoundError(f"No *_global.csv file found in {table_dir}")
    return tables[0]


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate topology consistency for forged trace claims.")
    parser.add_argument("--claims", type=Path, default=project_root / "claim" / "forged_trace_claims.csv")
    parser.add_argument("--global-anchor-table", type=Path, default=default_global_anchor_table(project_root))
    parser.add_argument("--anchor-connection", type=Path, default=project_root / "path_reconstruction" / "Anchor_connection.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Claim_Detection" / "results" / "topology" / "forged_trace")
    args = parser.parse_args()

    # graph 用于检查节点可达性，node_regions 用于把节点序列转换成区域候选序列。
    graph, node_regions = load_anchor_connection(args.anchor_connection)
    detected_sequences = load_detected_node_sequences(args.global_anchor_table)
    claims = pd.read_csv(args.claims, encoding="utf-8-sig")

    scored = pd.DataFrame([
        score_claim_row(row, detected_sequences, graph, node_regions)
        for _, row in claims.iterrows()
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "scores.csv"
    summary_path = args.output_dir / "summary.csv"
    scored.to_csv(scores_path, index=False, encoding="utf-8-sig")
    summarize(scored).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved topology scores to: {scores_path}")
    print(f"Saved summary to: {summary_path}")
    print(summarize(scored).to_string(index=False))


if __name__ == "__main__":
    main()
