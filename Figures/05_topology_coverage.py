from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

# =====================
# Data
# =====================
scenes = ["Parking", "Mall", "Campus"]
path_coverage = np.array([92.6, 89.8, 91.4])
transition_coverage = np.array([95.4, 93.7, 94.6])

path_min = np.array([90.9, 86.8, 89.2])
path_max = np.array([94.1, 92.4, 93.5])
transition_min = np.array([94.2, 91.7, 93.0])
transition_max = np.array([96.8, 95.7, 96.1])

path_err = np.vstack([path_coverage - path_min, path_max - path_coverage])
transition_err = np.vstack(
    [transition_coverage - transition_min, transition_max - transition_coverage]
)

# =====================
# Figure & font
# =====================
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
    }
)

fig, ax = plt.subplots(figsize=(4.1 / 2.54, 2.5 / 2.54))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =====================
# Bars
# =====================
x = np.arange(len(scenes))
bar_w = 0.18
offset = 0.12

colors = ["#4874CB", "#BCD3F5"]

err_kw = dict(
    ecolor="black",
    elinewidth=0.6,
    capthick=0.6,
    capsize=1.6,
)

ax.bar(
    x - offset,
    path_coverage,
    width=bar_w,
    color=colors[0],
    edgecolor="black",
    linewidth=0.3,
    yerr=path_err,
    error_kw=err_kw,
    label="Path",
)

ax.bar(
    x + offset,
    transition_coverage,
    width=bar_w,
    color=colors[1],
    edgecolor="#4874CB",
    linewidth=0.5,
    yerr=transition_err,
    error_kw=err_kw,
    label="Trans.",
)

# =====================
# Axes
# =====================
ax.set_xticks(x)
ax.set_xticklabels(scenes)
ax.set_ylabel("Coverage (%)", labelpad=1)

ax.set_ylim(80, 100)
ax.set_yticks([80, 90, 100])
ax.set_xlim(-0.55, len(scenes) - 0.45)

ax.tick_params(axis="both", length=2, width=0.5, pad=1)

# =====================
# Legend
# =====================
leg = ax.legend(
    frameon=True,
    fontsize=5.0,
    loc="upper center",
    ncol=2,
    handlelength=0.9,
    handletextpad=0.3,
    columnspacing=0.55,
    borderpad=0.2,
    labelspacing=0.2,
    bbox_to_anchor=(0.5, 1.02),
)

leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(0.7)
leg.get_frame().set_linewidth(0.0)

# =====================
# Border
# =====================
for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# =====================
# Layout & save
# =====================
plt.subplots_adjust(left=0.20, right=0.95, bottom=0.18, top=0.95)

output_path = Path(__file__).with_suffix(".pdf")
fig.savefig(output_path, format="pdf")

print(f"Saved to: {output_path}")
