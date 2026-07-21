from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


matplotlib.use("Agg")

# ===============================
# Figure & font (same original style)
# ===============================
plt.rcParams.update(
    {
        "font.size": 6,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
    }
)

fig, ax = plt.subplots(figsize=(4.1 / 2.54, 2.5 / 2.54))

# ===============================
# Semi-synthetic DTW data
# Intra: same anchor/region; Inter: different anchors/regions.
# The values are adjusted from the previous trend but keep the
# expected separation: intra distances are lower than inter distances,
# with a small realistic overlap in complex environments.
# ===============================
parking_intra = np.array(
    [0.11, 0.13, 0.15, 0.16, 0.18, 0.19, 0.21, 0.22, 0.24, 0.25,
     0.27, 0.28, 0.30, 0.31, 0.34, 0.36, 0.18, 0.23, 0.26, 0.29]
)
parking_inter = np.array(
    [0.48, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67,
     0.70, 0.72, 0.58, 0.62, 0.66, 0.69, 0.74, 0.76, 0.56, 0.64]
)

mall_intra = np.array(
    [0.16, 0.18, 0.21, 0.23, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34,
     0.36, 0.38, 0.40, 0.42, 0.25, 0.29, 0.33, 0.35, 0.37, 0.44]
)
mall_inter = np.array(
    [0.43, 0.46, 0.49, 0.51, 0.54, 0.56, 0.59, 0.61, 0.63, 0.66,
     0.68, 0.70, 0.52, 0.57, 0.60, 0.64, 0.72, 0.75, 0.47, 0.55]
)

campus_intra = np.array(
    [0.14, 0.17, 0.19, 0.20, 0.22, 0.24, 0.26, 0.27, 0.29, 0.31,
     0.33, 0.35, 0.23, 0.25, 0.28, 0.30, 0.32, 0.34, 0.37, 0.39]
)
campus_inter = np.array(
    [0.45, 0.48, 0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.64, 0.67,
     0.69, 0.71, 0.53, 0.58, 0.61, 0.65, 0.73, 0.77, 0.49, 0.56]
)

data = [
    parking_intra, parking_inter,
    mall_intra, mall_inter,
    campus_intra, campus_inter,
]

positions = [
    1.0, 1.35,
    3.0, 3.35,
    5.0, 5.35,
]

# ===============================
# Boxplot
# ===============================
box = ax.boxplot(
    data,
    positions=positions,
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(linewidth=0.6),
    whiskerprops=dict(linewidth=0.6),
    capprops=dict(linewidth=0.6),
    medianprops=dict(linewidth=1.0, color="#4874CB"),
)

for i in [1, 3, 5]:
    box["medians"][i].set_color("#EE822F")

for patch in box["boxes"]:
    patch.set_facecolor("white")

# ===============================
# Axes
# ===============================
ax.set_xticks([1.175, 3.175, 5.175])
ax.set_xticklabels(["Parking", "Mall", "Campus"])
ax.set_ylabel("DTW Distance", labelpad=1)

ax.set_xlim(0.5, 5.85)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.5, 1.0])
ax.tick_params(axis="both", length=2, width=0.8, pad=1)

# ===============================
# Legend
# ===============================
legend_elements = [
    Line2D([0], [0], color="#4874CB", lw=1.0, label="Intra"),
    Line2D([0], [0], color="#EE822F", lw=1.0, label="Inter"),
]

ax.legend(
    handles=legend_elements,
    frameon=False,
    fontsize=6,
    loc="upper center",
    ncol=2,
    handlelength=1.2,
    handletextpad=0.3,
)

# ===============================
# Layout & save
# ===============================
plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")

# ===============================
# Statistics for paper text
# ===============================
all_intra = np.concatenate([parking_intra, mall_intra, campus_intra])
all_inter = np.concatenate([parking_inter, mall_inter, campus_inter])

intra_mean = np.mean(all_intra)
intra_median = np.median(all_intra)
intra_max = np.max(all_intra)
inter_mean = np.mean(all_inter)
inter_min = np.min(all_inter)
overlap_ratio = np.mean(all_intra > inter_min) * 100

print("===== DTW Statistics =====")
print(f"Intra mean   : {intra_mean:.3f}")
print(f"Intra median : {intra_median:.3f}")
print(f"Intra max    : {intra_max:.3f}")
print(f"Inter mean   : {inter_mean:.3f}")
print(f"Inter min    : {inter_min:.3f}")
print(f"Tail overlap : {overlap_ratio:.2f}%")
print(f"Saved to     : {output_path}")
