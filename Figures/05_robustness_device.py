from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

# ===================== Raw data =====================
device_labels = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
overall_far = np.array([3.91, 4.05, 4.20, 4.32, 3.99, 4.45, 4.13, 4.57])
far_error = np.array([0.42, 0.55, 0.48, 0.66, 0.46, 0.78, 0.40, 0.72])

# ===================== Figure size: cm to inch =====================
cm_to_in = 1 / 2.54
fig_w = 8.4 * cm_to_in
fig_h = 5.0 * cm_to_in

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "sans-serif",
    }
)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ===================== Bars =====================
x = np.arange(len(device_labels))
ax.bar(
    x,
    overall_far,
    width=0.42,
    color="#4874CB",
    edgecolor="black",
    linewidth=0.3,
    yerr=far_error,
    error_kw={
        "ecolor": "black",
        "elinewidth": 0.8,
        "capsize": 2.0,
        "capthick": 0.8,
    },
)

ax.axhline(
    overall_far.mean(),
    color="#999999",
    linestyle="--",
    linewidth=0.6,
    zorder=0,
)

# ===================== Axes =====================
ax.set_xticks(x)
ax.set_xticklabels(device_labels)
ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_ylim(0, 6)
ax.set_yticks([0, 2, 4, 6])

ax.tick_params(axis="both", pad=1, width=0.5, length=2)

for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.17,
    right=0.97,
    bottom=0.21,
    top=0.94,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
