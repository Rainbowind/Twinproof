import os
import pandas as pd
import ast

# 根目录路径（修改为你的一级文件夹路径）
root_folder = r"../data/collectionData_new"

RSSI_INVALID = 2147483647


def safe_eval_list(value):
    """
    安全解析 Cell_Type, Cell_ID, Cell_RSSI 这种字符串列表。
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    # 如果是标准 Python 列表字符串
    if value.startswith('[') and value.endswith(']'):
        try:
            return ast.literal_eval(value)
        except Exception:
            # 如果 literal_eval 失败，就按逗号手动切分
            return [v.strip().strip('"').strip("'") for v in value[1:-1].split(',') if v.strip()]
    else:
        return [value.strip().strip('"').strip("'")]


def is_invalid_rssi(r):
    """
    判断 RSSI 是否为异常值（2147483647），兼容字符串/浮点/空值。
    """
    if pd.isna(r):
        return False
    try:
        return int(r) == RSSI_INVALID
    except Exception:
        return str(r).strip() == str(RSSI_INVALID)


def remove_gsm_columns(row):
    """
    从单行中删除 Gsm 相关列，保留 Lte 的 Cell_Type, Cell_ID, Cell_RSSI。
    同时删去 RSSI 异常（2147483647）的数据（包括对应ID）。
    """
    types = safe_eval_list(row['Cell_Type'])
    ids = safe_eval_list(row['Cell_ID'])
    rssis = safe_eval_list(row['Cell_RSSI'])

    # 统一长度（防止三列长度不一致导致 zip 截断丢数据）
    n = min(len(types), len(ids), len(rssis))
    types, ids, rssis = types[:n], ids[:n], rssis[:n]

    # 先筛选 Lte，再筛掉 RSSI 异常值
    filtered = []
    for t, i, r in zip(types, ids, rssis):
        if str(t).lower() != 'lte':
            continue
        if is_invalid_rssi(r):
            continue
        filtered.append((t, i, r))

    if filtered:
        new_types, new_ids, new_rssis = zip(*filtered)
        return pd.Series([list(new_types), list(new_ids), list(new_rssis)])
    else:
        return pd.Series([[], [], []])


def process_signal_file(signal_path):
    """
    处理单个 signal 文件：
    - 如果全是 Gsm，删除 signal 和 sensor 文件；
    - 如果混合 Lte 和 Gsm，删除 Gsm 列并保存；
    - 额外：删除 RSSI 异常（2147483647）的数据（含对应ID）。
    """
    try:
        df = pd.read_csv(signal_path)
        if 'Cell_Type' not in df.columns:
            print(f"⚠️ {signal_path} 没有 Cell_Type 列，跳过")
            return

        # 检查是否全是 Gsm（保留你原逻辑）
        all_types = df['Cell_Type'].astype(str).str.lower()
        if all(all_types.str.contains('gsm')):
            print(f"🗑 全是 Gsm，删除文件: {signal_path}")
            os.remove(signal_path)

            sensor_file = os.path.join(
                os.path.dirname(signal_path),
                os.path.basename(signal_path).replace("signal", "sensor")
            )
            if os.path.exists(sensor_file):
                os.remove(sensor_file)
                print(f"✅ 删除对应 sensor 文件: {sensor_file}")
            else:
                print(f"❌ 找不到对应 sensor 文件: {sensor_file}")
            return

        # 如果有 Lte，删除 Gsm 列 + 删除异常 RSSI 条目
        new_cols = df.apply(remove_gsm_columns, axis=1)
        df['Cell_Type'] = new_cols[0]
        df['Cell_ID'] = new_cols[1]
        df['Cell_RSSI'] = new_cols[2]

        df.to_csv(signal_path, index=False)
        print(f"✅ 处理并保存: {signal_path}")

    except Exception as e:
        print(f"读取 {signal_path} 出错: {e}")
        with open(signal_path, 'r', encoding='utf-8') as f:
            print("出错文件前5行：")
            for _ in range(5):
                print(f.readline())


def clean_collection_data(root_folder):
    """
    遍历 root_folder 中的所有 signal 文件并处理。
    """
    for dirpath, _, files in os.walk(root_folder):
        signal_files = [f for f in files if f.startswith("signal") and f.endswith(".csv")]

        for signal_file in signal_files:
            signal_path = os.path.join(dirpath, signal_file)
            process_signal_file(signal_path)


# 执行
clean_collection_data(root_folder)
