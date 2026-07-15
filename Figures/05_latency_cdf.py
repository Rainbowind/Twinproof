from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


matplotlib.use("Agg")

# ===================== Raw data =====================
# Synthetic end-to-end latency samples for CDF visualization.
# Each scenario contains 200 claim-verification latency samples.
# Most samples are concentrated in the normal response interval. A few samples
# form low-latency and high-latency outliers, while the three scenarios remain
# largely overlapped to reflect realistic runtime fluctuations.
rng = np.random.default_rng(20260715)

parking = np.concatenate(
    [
        rng.normal(300, 16, 165),
        rng.normal(330, 14, 20),
        np.array([222, 226, 230, 234, 238, 242, 246, 250, 254, 258], dtype=float),
        rng.normal(390, 22, 5),
    ]
)
indoor_road = np.concatenate(
    [
        rng.normal(291, 16, 165),
        rng.normal(320, 14, 20),
        np.array([238, 242, 246, 250, 254, 258, 262, 266, 270, 274], dtype=float),
        rng.normal(378, 24, 5),
    ]
)
mall = np.concatenate(
    [
        rng.normal(294, 17, 165),
        rng.normal(326, 15, 20),
        np.array([254, 258, 262, 266, 270, 274, 278, 282, 286, 290], dtype=float),
        rng.normal(385, 25, 5),
    ]
)

latencies = {
    "Parking": np.clip(np.rint(parking), 220, 445).astype(int),
    "Indoor road": np.clip(np.rint(indoor_road), 236, 440).astype(int),
    "Mall": np.clip(np.rint(mall), 252, 450).astype(int),
}

# ===================== Export raw data =====================
data_path = Path(__file__).with_name("05_latency_cdf_data.csv")
pd.DataFrame(latencies).to_csv(data_path, index=False)

# ===================== Plot settings =====================
data = [latencies["Parking"], latencies["Mall"], latencies["Indoor road"]]
labels = ["Parking", "Mall", "Indoor road"]

colors = {
    "Parking": "#EE822F",
    "Mall": "#4874CB",
    "Indoor road": "#75BD42",
}


def compute_cdf(values):
    x_sorted = np.sort(values)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y


plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 1.0,
    }
)

# ===================== Figure size =====================
fig_w = 4.1 / 2.54
fig_h = 2.5 / 2.54
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# ===================== CDF curves =====================
for values, label in zip(data, labels):
    x, y = compute_cdf(values)
    ax.plot(
        x,
        y,
        label=label,
        color=colors[label],
    )

# ===================== Axes =====================
ax.set_xlim(200, 450)
ax.set_xticks([200, 250, 300, 350, 400, 450])
ax.set_ylim(0, 1.0)

ax.set_xlabel("Latency (ms)", labelpad=1)
ax.set_ylabel("CDF", labelpad=1)

ax.tick_params(axis="x", pad=1)
ax.tick_params(axis="y", pad=1)

for spine in ax.spines.values():
    spine.set_linewidth(0.6)

# ===================== Legend =====================
legend = ax.legend(
    frameon=True,
    fontsize=6,
    loc="lower right",
)
legend.get_frame().set_alpha(0.7)
legend.get_frame().set_linewidth(0)

# ===================== Layout =====================
plt.subplots_adjust(
    left=0.25,
    right=0.95,
    bottom=0.26,
    top=0.95,
)

# ===================== Export =====================
output_path = Path(__file__).with_suffix(".pdf")
plt.savefig(output_path, format="pdf")
plt.close(fig)
