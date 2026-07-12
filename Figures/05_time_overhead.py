from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use("Agg")

# ===================== Raw data =====================
columns = ["Component", "Description", "Overhead"]
rows = [
    [r"$S_{topo}$", "Topological consistency", "1.8 ms"],
    [r"$S_{time}$", "Temporal consistency", "0.6 ms"],
    [r"$S_{signal}$", "Cross-signal consistency", "4.7 ms"],
    [r"$S_{curr}$", "Currentness consistency", "225.4 ms"],
    ["Total", "End-to-end verification", "232.5 ms"],
]

# ===================== Figure size: cm to inch =====================
cm_to_in = 1 / 2.54
fig_w = 8.4 * cm_to_in
fig_h = 2.9 * cm_to_in

plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    }
)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ===================== Manual three-line table =====================
x_left = 0.04
x_right = 0.96
col_x = [0.18, 0.54, 0.86]

top_y = 0.88
header_y = 0.76
mid_y = 0.66
row_ys = [0.55, 0.45, 0.35, 0.25, 0.15]
bottom_y = 0.08

ax.hlines(top_y, x_left, x_right, colors="black", linewidth=0.9)
ax.hlines(mid_y, x_left, x_right, colors="black", linewidth=0.6)
ax.hlines(bottom_y, x_left, x_right, colors="black", linewidth=0.9)

for x, text in zip(col_x, columns):
    ax.text(x, header_y, text, ha="center", va="center", fontweight="bold")

for row_y, row in zip(row_ys, rows):
    is_total = row[0] == "Total"
    weight = "bold" if is_total else "normal"
    ax.text(col_x[0], row_y, row[0], ha="center", va="center", fontweight=weight)
    ax.text(col_x[1], row_y, row[1], ha="center", va="center", fontweight=weight)
    ax.text(col_x[2], row_y, row[2], ha="center", va="center", fontweight=weight)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.04,
    top=0.96,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
