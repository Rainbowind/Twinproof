import argparse
import ast
import json
from pathlib import Path

import pandas as pd


def parse_list(value):
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
    stem = str(claim_path).strip().replace("\\", "/").split("/")[-1]
    if stem.endswith("_merged"):
        return f"{stem}.csv"
    if stem.endswith("_merged.csv"):
        return stem
    return f"{stem}_merged.csv"


def parse_connected(value):
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    return [int(v) for v in ast.literal_eval(text)]


def parse_regions(value):
    if pd.isna(value):
        return []
    return [region.strip() for region in str(value).split(",") if region.strip()]


def load_anchor_connection(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    graph = {}
    node_regions = {}

    for _, row in df.iterrows():
        node = int(row["Cluster_Label"])
        graph[node] = set(parse_connected(row["Connected_Classes"]))
        node_regions[node] = parse_regions(row["Environment_Constraint"])

    return graph, node_regions


def load_detected_node_sequences(global_anchor_table):
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
            if nodes and nodes[-1] == node:
                continue
            nodes.append(node)
        sequences[str(file_name)] = nodes
    return sequences


def reachability_flag(node_sequence, graph):
    return 1 if expand_reachable_path(node_sequence, graph) else -1


def shortest_path(start, end, graph):
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
    if not node_sequence:
        return []
    if len(node_sequence) == 1:
        return list(node_sequence)

    expanded = [node_sequence[0]]
    for current_node, next_node in zip(node_sequence[:-1], node_sequence[1:]):
        path = shortest_path(current_node, next_node, graph)
        if not path:
            return []
        expanded.extend(path[1:])
    return expanded


def nodes_to_region_candidates(node_sequence, node_regions):
    return [node_regions.get(node, []) for node in node_sequence]


def candidate_match(claim_region, detected_candidates):
    return claim_region in detected_candidates


def lcs_region_candidates(claim_trace, detected_region_candidates):
    if not claim_trace or not detected_region_candidates:
        return 0

    previous = [0] * (len(detected_region_candidates) + 1)
    for claim_region in claim_trace:
        current = [0] * (len(detected_region_candidates) + 1)
        for index, candidates in enumerate(detected_region_candidates, start=1):
            if candidate_match(claim_region, candidates):
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]



def sequence_similarity(claim_trace, detected_region_candidates):
    denominator = max(len(claim_trace), len(detected_region_candidates))
    if denominator == 0:
        return 0.0
    return lcs_region_candidates(claim_trace, detected_region_candidates) / denominator


def whole_similarity(claim_trace, detected_region_candidates):
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
    claim_trace = parse_list(row.get("Claim_Trace"))
    file_name = claim_path_to_file_name(row.get("Claim_Path"))
    detected_nodes = detected_sequences.get(file_name, [])
    reachable_node_path = expand_reachable_path(detected_nodes, graph)
    reach_flag = 1 if reachable_node_path else -1
    detected_region_candidates = nodes_to_region_candidates(reachable_node_path, node_regions)

    if reach_flag == -1:
        s_seq = 0.0
        s_whole = 0.0
        s_topo = 0.0
        decision = "attack_no_detected_nodes" if not detected_nodes else "attack_unreachable"
    else:
        s_seq = sequence_similarity(claim_trace, detected_region_candidates)
        s_whole = whole_similarity(claim_trace, detected_region_candidates)
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


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Calculate topology consistency for forged trace claims.")
    parser.add_argument("--claims", type=Path, default=project_root / "claim" / "forged_trace_claims.csv")
    parser.add_argument(
        "--global-anchor-table",
        type=Path,
        default=project_root / "Claim_Detection" / "anchor_with_global_id" / "anchor_combined_交叉_小米MAX3_移动卡_global.csv",
    )
    parser.add_argument("--anchor-connection", type=Path, default=project_root / "path_reconstruction" / "Anchor_connection.csv")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Claim_Detection" / "results" / "topology" / "forged_trace")
    args = parser.parse_args()

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
