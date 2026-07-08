import os

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ===================== Data =====================
frr = ["1", "2", "3", "4", "5", "6"]
far = np.array([6.41, 4.24, 3.55, 3.15, 2.51, 1.86])

# ===================== Figure size =====================
cm_to_in = 1 / 2.54
fig_w = 3.5 * cm_to_in
fig_h = 1.6 * cm_to_in

# ===================== Global style =====================
plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 6,
    "font.family": "sans-serif"
})

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ===================== Bar chart =====================
x = np.arange(len(frr))
bar_w = 0.34

ax.bar(
    x,
    far,
    width=bar_w,
    color="#4874CB",
    edgecolor="none",
    linewidth=0.3,
    zorder=3
)

# ===================== Axes =====================
ax.set_xticks(x)
ax.set_xticklabels(frr)

ax.set_xlabel("FRR (%)", labelpad=1)
ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_ylim(0, 10)
ax.set_yticks([0, 2, 4, 6, 8, 10])

ax.tick_params(axis="both", pad=1, width=0.5, length=2)

# ===================== Spines =====================
for sp in ["left", "bottom", "top", "right"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.22,
    right=0.96,
    bottom=0.34,
    top=0.94
)

# ===================== Save =====================
out_name = os.path.splitext(os.path.basename(__file__))[0] + ".pdf"
out_path = os.path.join(os.path.dirname(__file__), out_name)
plt.savefig(out_path, format="pdf")
plt.show()
