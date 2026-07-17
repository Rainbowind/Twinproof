from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

# ===================== Raw data =====================
frr_values = np.array([1, 2, 3, 4, 5, 6])

series = {
    "Forged": [10.50, 7.13, 6.12, 4.42, 2.61, 2.71],
    "Replay": [15.12, 13.21, 10.81, 8.96, 7.14, 6.33],
    "Proxy": [11.53, 9.50, 6.47, 5.43, 3.62, 3.04],
    "Trans.": [11.04, 8.43, 6.43, 6.95, 3.13, 2.83],
    "Overall": [12.36, 10.34, 7.58, 6.21, 4.12, 3.83],
}

# ===================== Line styles =====================
line_styles = {
    "Forged": {
        "color": "#4874CB",
        "linestyle": "-",
        "marker": "o",
    },
    "Replay": {
        "color": "#668DD5",
        "linestyle": "--",
        "marker": "s",
    },
    "Proxy": {
        "color": "#8BADE2",
        "linestyle": "-.",
        "marker": "^",
    },
    "Trans.": {
        "color": "#AFC9ED",
        "linestyle": ":",
        "marker": "D",
    },
    "Overall": {
        "color": "#EE822F",
        "linestyle": "-",
        "marker": "P",
    },
}

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
    }
)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ===================== Lines =====================
for label, values in series.items():
    style = line_styles[label]
    is_overall = label == "Overall"
    ax.plot(
        frr_values,
        values,
        label=label,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=1.2 if is_overall else 0.9,
        markersize=3.2 if is_overall else 2.8,
        markerfacecolor="white" if not is_overall else style["color"],
        markeredgecolor=style["color"],
        markeredgewidth=0.6,
        zorder=3 if is_overall else 2,
    )

# ===================== Axes =====================
ax.set_xlabel("FRR (%)", labelpad=1)
ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_xlim(0.8, 6.2)
ax.set_xticks(frr_values)
ax.set_ylim(0, 16)
ax.set_yticks([0, 4, 8, 12, 16])

ax.tick_params(axis="both", pad=1, width=0.5, length=2)

for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# ===================== Legend =====================
handles, labels = ax.get_legend_handles_labels()
handle_map = dict(zip(labels, handles))
legend_handles = [
    handle_map["Forged"],
    handle_map["Replay"],
    handle_map["Proxy"],
    handle_map["Trans."],
    handle_map["Overall"],
]
legend_labels = ["Forged", "Replay", "Proxy", "Trans.", "Overall"]

legend = ax.legend(
    legend_handles,
    legend_labels,
    loc="upper right",
    ncol=2,
    frameon=True,
    borderpad=0.25,
    handlelength=1.4,
    handletextpad=0.35,
    columnspacing=0.55,
    labelspacing=0.18,
    bbox_to_anchor=(1.01, 1.03),
)

frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_alpha(0.65)
frame.set_edgecolor("none")
frame.set_linewidth(0.0)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.20,
    right=0.95,
    bottom=0.26,
    top=0.95,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
