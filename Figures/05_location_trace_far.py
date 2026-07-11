from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


matplotlib.use("Agg")

# ===================== Raw data =====================
attack_labels = ["Forged", "Replay", "Proxy", "Trans."]
method_labels = ["Long-term", "Short-term"]

far_values = np.array(
    [
        [2.61, 6.21],
        [7.14, 4.32],
        [3.62, 2.16],
        [3.13, 1.26],
    ]
)

# ===================== Blue gradient + texture =====================
colors = [
    "#4874CB",
    "#BCD3F5",
]

hatches = [
    "",
    "/////",
]

# ===================== Figure size: cm to inch =====================
cm_to_in = 1 / 2.54
fig_w = 4.1 * cm_to_in
fig_h = 2.5 * cm_to_in

plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 4.8,
        "font.family": "sans-serif",
        "hatch.linewidth": 0.2,
    }
)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ===================== Bar layout =====================
bar_w = 0.08
cluster_gap = 0.22

group_xs = []
start = 0.0
for _ in attack_labels:
    xs = start + np.array([0, 1]) * bar_w
    group_xs.append(xs)
    start = xs[-1] + bar_w + cluster_gap


def draw_group(xs, values):
    for i, x in enumerate(xs):
        ax.bar(
            x,
            values[i],
            width=bar_w,
            color=colors[i],
            edgecolor="black",
            linewidth=0.3,
            hatch=hatches[i],
            align="center",
        )


for attack_index, xs in enumerate(group_xs):
    draw_group(xs, far_values[attack_index])

# ===================== Axes =====================
group_centers = [np.mean(xs) for xs in group_xs]

ax.set_xticks(group_centers)
ax.set_xticklabels(attack_labels)

ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_ylim(0, 8)
ax.set_yticks([0, 2, 4, 6, 8])

ax.tick_params(axis="both", pad=1, width=0.5, length=2)

for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# ===================== Legend =====================
legend_handles = [
    Patch(facecolor=colors[0], edgecolor="black", linewidth=0.3, hatch=hatches[0], label=method_labels[0]),
    Patch(facecolor=colors[1], edgecolor="black", linewidth=0.3, hatch=hatches[1], label=method_labels[1]),
]

legend = ax.legend(
    handles=legend_handles,
    loc="upper right",
    ncol=1,
    frameon=True,
    borderpad=0.18,
    handlelength=1.0,
    handletextpad=0.35,
    columnspacing=0.75,
    labelspacing=0.2,
    bbox_to_anchor=(0.99, 0.99),
)

frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_alpha(0.8)
frame.set_edgecolor("none")
frame.set_linewidth(0.0)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.2,
    right=0.95,
    bottom=0.15,
    top=0.95,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
