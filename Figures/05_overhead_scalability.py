from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# Raw data
# =========================
data_path = Path(__file__).with_name("05_overhead_scalability_data.csv")
data = pd.read_csv(data_path)

x = data["Area_km2"].to_numpy()
y_min = data["Total_min"].to_numpy()
y_mean = data["Total_mean"].to_numpy()
y_max = data["Total_max"].to_numpy()

# =========================
# Paper plotting style
# =========================
plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.6,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.family": "sans-serif",
    }
)

fig, ax = plt.subplots(figsize=(4.1 / 2.54, 2.5 / 2.54))

color = "#4874CB"

# =========================
# Mean latency line
# =========================
ax.plot(
    x,
    y_mean,
    color=color,
    linestyle="--",
    linewidth=1,
    zorder=3,
)

# =========================
# Hollow markers + min-max error bars
# =========================
y_err = [
    y_mean - y_min,
    y_max - y_mean,
]

ax.errorbar(
    x,
    y_mean,
    yerr=y_err,
    fmt="o",
    markersize=3.2,
    markerfacecolor="white",
    markeredgecolor=color,
    markeredgewidth=0.8,
    ecolor=color,
    elinewidth=0.8,
    capsize=2.0,
    capthick=0.8,
    linestyle="none",
    zorder=4,
)

# =========================
# Axes and style
# =========================
ax.set_xlabel(r"Area (k m$^2$)", labelpad=1)
ax.set_ylabel("Latency (ms)", labelpad=1)

ax.set_xlim(16, 124)
ax.set_xticks([20, 40, 60, 80, 100, 120])

ax.set_ylim(150, 290)
ax.set_yticks([150, 180, 210, 240, 270])

ax.tick_params(axis="x", pad=1)
ax.tick_params(axis="y", pad=1)

for spine in ax.spines.values():
    spine.set_linewidth(0.6)

# =========================
# Layout
# =========================
plt.subplots_adjust(
    left=0.22,
    right=0.95,
    bottom=0.27,
    top=0.95,
)

# =========================
# Export
# =========================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
