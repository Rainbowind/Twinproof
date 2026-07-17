from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ===================== Raw data =====================
data_path = Path(__file__).with_name("05_robustness_window_time_attacks_data.csv")
data = pd.read_csv(data_path)

times = data["Time_s"]
series = {
    "Forged": data["Forged"],
    "Replay": data["Replay"],
    "Proxy": data["Proxy"],
    "Trans.": data["Trans."],
}

line_styles = {
    "Forged": {"color": "#4874CB", "linestyle": "-", "marker": "s"},
    "Replay": {"color": "#EE822F", "linestyle": "--", "marker": "^"},
    "Proxy": {"color": "#4874CB", "linestyle": "-.", "marker": "D"},
    "Trans.": {"color": "#EE822F", "linestyle": ":", "marker": "v"},
}

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
    }
)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ===================== Lines =====================
for label, values in series.items():
    style = line_styles[label]
    ax.plot(
        times,
        values,
        label=label,
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        linewidth=0.9,
        markersize=2.7,
        markerfacecolor="white",
        markeredgecolor=style["color"],
        markeredgewidth=0.6,
    )

# ===================== Axes =====================
ax.set_xlabel("Window Length (s)", labelpad=1)
ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_xlim(1.5, 20.5)
ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
ax.set_ylim(0, 8)
ax.set_yticks([0, 2, 4, 6, 8])

ax.tick_params(axis="both", pad=1, width=0.5, length=2)

for spine in ["left", "right", "top", "bottom"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(0.6)

# ===================== Legend =====================
legend = ax.legend(
    loc="upper right",
    ncol=2,
    frameon=True,
    borderpad=0.25,
    handlelength=2.2,
    handletextpad=0.45,
    columnspacing=0.8,
    labelspacing=0.25,
    bbox_to_anchor=(0.99, 0.99),
)

frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_alpha(0.8)
frame.set_edgecolor("none")
frame.set_linewidth(0.0)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.14,
    right=0.97,
    bottom=0.19,
    top=0.96,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
