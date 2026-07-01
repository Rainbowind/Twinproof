import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MEG_COLUMNS = [f"Column_{index}" for index in range(1, 401)]


def row_zscore(values):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    arr = arr.ffill().bfill().fillna(0.0).to_numpy()
    std = arr.std()
    if std == 0:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def downsample(values, step):
    return np.asarray(values, dtype=float)[::step]


def dtw_distance(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0

    for left_value in left:
        current = np.full(len(right) + 1, np.inf)
        for j, right_value in enumerate(right, start=1):
            cost = abs(left_value - right_value)
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current

    normalizer = max(len(left) + len(right), 1)
    return float(previous[-1] / normalizer)


def feature_id_from_path(path):
    match = re.search(r"anchor_feature_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse global anchor id from file name: {path.name}")
    return int(match.group(1))


def load_anchor_prototypes(feature_dir, sample_step):
    prototypes = {}
    for feature_path in sorted(feature_dir.glob("anchor_feature_*.csv")):
        df = pd.read_csv(feature_path, usecols=MEG_COLUMNS, encoding="utf-8-sig")
        processed = np.vstack([row_zscore(row) for row in df.to_numpy()])
        prototype = np.nanmedian(processed, axis=0)
        prototypes[feature_id_from_path(feature_path)] = downsample(prototype, sample_step)

    if not prototypes:
        raise FileNotFoundError(f"No anchor_feature_*.csv files found in {feature_dir}")
    return prototypes


def match_signal_to_global_id(signal, prototypes, sample_step):
    processed_signal = downsample(row_zscore(signal), sample_step)
    distances = {
        global_id: dtw_distance(processed_signal, prototype)
        for global_id, prototype in prototypes.items()
    }
    best_global_id = min(distances, key=distances.get)
    return best_global_id


def add_global_anchor_id(input_path, output_path, feature_dir, sample_step):
    prototypes = load_anchor_prototypes(feature_dir, sample_step)
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    missing_cols = [col for col in MEG_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input table is missing MEG columns, first missing column: {missing_cols[0]}")

    global_ids = []
    for _, row in df.iterrows():
        global_id = match_signal_to_global_id(row[MEG_COLUMNS].values, prototypes, sample_step)
        global_ids.append(global_id)

    if "Global_Anchor_ID" in df.columns:
        df = df.drop(columns=["Global_Anchor_ID"])
    df = df.copy()
    df["Global_Anchor_ID"] = global_ids

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def main():
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "Find_Anchor" / "anchor" / "anchor_combined_交叉_小米MAX3_移动卡.csv"
    default_output = (
        project_root
        / "Claim_Detection"
        / "anchor_with_global_id"
        / "anchor_combined_交叉_小米MAX3_移动卡_global.csv"
    )
    default_feature_dir = project_root / "path_reconstruction" / "Anchor_feature_parking"

    parser = argparse.ArgumentParser(description="Match local anchor rows to global anchor ids by MEG-DTW.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--feature-dir", type=Path, default=default_feature_dir)
    parser.add_argument("--sample-step", type=int, default=4)
    args = parser.parse_args()

    result = add_global_anchor_id(args.input, args.output, args.feature_dir, args.sample_step)
    counts = result["Global_Anchor_ID"].value_counts().sort_index()

    print(f"Saved labeled anchor table to: {args.output}")
    print("Global_Anchor_ID distribution:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
