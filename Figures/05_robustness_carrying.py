from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


matplotlib.use("Agg")

# ===================== Raw data =====================
category_labels = ["Hand-held", "Pocket", "Bag"]
attack_labels = ["Forged", "Replay", "Proxy", "Trans."]

far_values = np.array(
    [
        [2.61, 7.14, 3.62, 3.13],
        [3.18, 8.36, 4.71, 3.92],
        [3.47, 8.92, 5.16, 4.28],
    ]
)

# ===================== Blue gradient =====================
colors = [
    "#4874CB",
    "#7FA6E6",
    "#BCD3F5",
    "#FFFFFF",
]

hatches = [
    "",
    "//",
    "\\\\\\\\",
    "",
]

# ===================== Figure size: cm to inch =====================
cm_to_in = 1 / 2.54
fig_w = 8.4 * cm_to_in
fig_h = 5.0 * cm_to_in

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
x = np.arange(len(category_labels))
bar_w = 0.15
offsets = (np.arange(len(attack_labels)) - (len(attack_labels) - 1) / 2) * bar_w

for i, label in enumerate(attack_labels):
    ax.bar(
        x + offsets[i],
        far_values[:, i],
        width=bar_w,
        color=colors[i],
        edgecolor="black",
        linewidth=0.3,
        hatch=hatches[i],
        label=label,
        align="center",
    )

# ===================== Axes =====================
ax.set_xticks(x)
ax.set_xticklabels(category_labels)
ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_ylim(0, 10)
ax.set_yticks([0, 2, 4, 6, 8, 10])

ax.tick_params(axis="both", pad=1, width=0.5, length=2)

for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# ===================== Legend =====================
legend_handles = [
    Patch(facecolor=colors[i], edgecolor="black", linewidth=0.3, hatch=hatches[i], label=attack_labels[i])
    for i in range(len(attack_labels))
]

legend = ax.legend(
    handles=legend_handles,
    loc="upper right",
    ncol=2,
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
    left=0.12,
    right=0.98,
    bottom=0.17,
    top=0.96,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)

