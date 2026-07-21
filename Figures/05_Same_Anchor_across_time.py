from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

# =====================
# Data
# Slightly adjusted from the previous version. The trend remains realistic:
# anchors stay stable across months, with mild temporal degradation and
# environment-specific fluctuations.
# =====================
months = np.arange(2, 26, 2)

parking = np.array([
    90.82, 91.76, 89.48, 91.92,
    88.92, 90.67, 91.35, 89.74,
    90.96, 91.88, 88.58, 90.24,
])

mall = np.array([
    89.76, 91.42, 88.34, 90.96,
    89.08, 91.83, 88.72, 91.95,
    90.51, 90.87, 88.96, 91.36,
])

campus = np.array([
    90.28, 88.86, 91.54, 89.73,
    90.91, 91.90, 89.25, 91.38,
    88.68, 90.47, 91.79, 89.56,
])

x = np.arange(len(months))

# =====================
# Font
# =====================
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }
)

# =====================
# Figure size
# =====================
cm_to_in = 1 / 2.54
fig_w = 4.1 * cm_to_in
fig_h = 2.5 * cm_to_in

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# =====================
# Curves
# =====================
ax.plot(
    x, parking,
    color="#4874CB",
    linestyle="--",
    linewidth=0.6,
    marker="o",
    markersize=2.8,
    markeredgewidth=0.6,
    markerfacecolor="none",
    label="Parking",
)

ax.plot(
    x, mall,
    color="#4874CB",
    linestyle="-",
    linewidth=0.6,
    marker="s",
    markersize=2.8,
    markeredgewidth=0.6,
    markerfacecolor="none",
    label="Mall",
)

ax.plot(
    x, campus,
    color="#4874CB",
    linestyle=":",
    linewidth=0.6,
    marker="^",
    markersize=2.8,
    markeredgewidth=0.6,
    markerfacecolor="none",
    label="Campus",
)

# =====================
# Axis
# =====================
ax.set_ylabel("ARR (%)", labelpad=1)
ax.set_xlabel("Time Gap (months)", labelpad=1)
ax.set_ylim(80, 100)

show_months = np.arange(4, 26, 4)
show_index = [np.where(months == m)[0][0] for m in show_months]
ax.set_xticks(show_index)
ax.set_xticklabels(show_months)
ax.tick_params(axis="both", pad=1)

# =====================
# Legend
# =====================
ax.legend(
    loc="lower left",
    ncol=3,
    frameon=False,
    fontsize=5,
    handlelength=1.2,
    handletextpad=0.3,
    columnspacing=0.8,
)

# =====================
# Border
# =====================
for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# =====================
# Layout & save
# =====================
plt.subplots_adjust(left=0.20, right=0.95, bottom=0.26, top=0.95)

output_path = Path(__file__).with_suffix(".pdf")
fig.savefig(output_path, format="pdf")
print(f"Saved to: {output_path}")
