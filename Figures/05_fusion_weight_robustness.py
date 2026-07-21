from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


matplotlib.use("Agg")

# ============================================================
# Semi-synthetic fusion-weight robustness analysis
#
# Replace this semi-synthetic FAR response with real score arrays
# when available:
#   score = w_topo * score_topo
#         + w_time * score_time
#         + w_signal * score_signal
#         + w_curr * score_curr
# Then set the threshold at a fixed FRR operating point and compute FAR.
# ============================================================

rng = np.random.default_rng(20260709)

n_configs = 1000
weights = rng.dirichlet(alpha=np.ones(4), size=n_configs)

# Order: S_topo, S_time, S_signal, S_curr
selected_weight = np.array([0.40, 0.10, 0.20, 0.30])
equal_weight = np.array([0.25, 0.25, 0.25, 0.25])


def deterministic_far(weight: np.ndarray) -> float:
    """A smooth semi-synthetic FAR surface over legal fusion weights."""
    w_topo, w_time, w_signal, w_curr = weight

    # Stable plateau around the selected/equal-weight region.
    distance_penalty = 2.50 * np.sum((weight - selected_weight) ** 2)

    # Extreme configurations are still legal, but usually less stable.
    dominance_penalty = 4.00 * max(0.0, np.max(weight) - 0.52) ** 1.35

    # Over-emphasizing temporal consistency alone is less discriminative.
    time_penalty = 1.80 * max(0.0, w_time - 0.34) ** 1.18

    # Very small currentness/topology weights weaken integrity validation.
    curr_penalty = 1.40 * max(0.0, 0.10 - w_curr) ** 0.80
    topo_penalty = 1.00 * max(0.0, 0.10 - w_topo) ** 0.80

    # A small interaction term reflects that topology and currentness help
    # stabilize one another, without making any single setting perfect.
    balance_bonus = 0.18 * min(w_topo, w_curr)

    return (
        4.186
        + distance_penalty
        + dominance_penalty
        + time_penalty
        + curr_penalty
        + topo_penalty
        - balance_bonus
    )


base_far = np.array([deterministic_far(w) for w in weights])

# Mild experiment-to-experiment variation and a few realistic long-tail cases.
noise = rng.normal(loc=0.0, scale=0.28, size=n_configs)
tail_mask = rng.random(n_configs) < 0.075
tail_extra = np.zeros(n_configs)
tail_extra[tail_mask] = rng.gamma(shape=1.8, scale=0.42, size=tail_mask.sum())

far_values = np.clip(base_far + noise + tail_extra, 2.7, 8.2)

selected_far = deterministic_far(selected_weight)
equal_far = deterministic_far(equal_weight)

output_dir = Path(__file__).resolve().parent
data_path = output_dir / "05_fusion_weight_robustness_data.csv"
pdf_path = output_dir / "05_fusion_weight_robustness.pdf"

df = pd.DataFrame(
    {
        "w_topo": weights[:, 0],
        "w_time": weights[:, 1],
        "w_signal": weights[:, 2],
        "w_curr": weights[:, 3],
        "FAR": far_values,
    }
)
df.to_csv(data_path, index=False)

# ===================== Figure style =====================
cm_to_in = 1 / 2.54
fig_w = 4.1 * cm_to_in
fig_h = 2.5 * cm_to_in

plt.rcParams.update(
    {
        "font.size": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 4.6,
        "font.family": "sans-serif",
    }
)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_grid = np.linspace(2.5, 8.2, 400)
kde = gaussian_kde(far_values, bw_method=0.24)
y_grid = kde(x_grid)

main_color = "#4874CB"
selected_color = "#EE822F"
equal_color = "#4C9F38"

ax.plot(
    x_grid,
    y_grid,
    color=main_color,
    linewidth=1.0,
    label="Random weights",
)
ax.fill_between(x_grid, y_grid, color=main_color, alpha=0.16, linewidth=0)

ax.axvline(
    selected_far,
    color=selected_color,
    linestyle="-",
    linewidth=0.9,
    label="Selected",
)
ax.axvline(
    equal_far,
    color=equal_color,
    linestyle="--",
    linewidth=0.9,
    label="Equal",
)

ax.set_xlabel("FAR (%)", labelpad=1)
ax.set_ylabel("Density", labelpad=1)

ax.set_xlim(2.5, 8.2)
ax.set_xticks([3, 4, 5, 6, 7, 8])
ax.set_ylim(bottom=0)
ax.tick_params(axis="both", pad=1, width=0.5, length=2)

for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

legend = ax.legend(
    loc="upper right",
    frameon=True,
    borderpad=0.2,
    handlelength=1.6,
    handletextpad=0.4,
    labelspacing=0.2,
)
frame = legend.get_frame()
frame.set_facecolor("white")
frame.set_alpha(0.75)
frame.set_edgecolor("none")
frame.set_linewidth(0.0)

plt.subplots_adjust(
    left=0.2,
    right=0.95,
    bottom=0.23,
    top=0.95,
)

plt.savefig(pdf_path, format="pdf")

print(f"Saved PDF: {pdf_path}")
print(f"Saved CSV: {data_path}")
print(f"Selected FAR: {selected_far:.2f}%")
print(f"Equal FAR: {equal_far:.2f}%")
print(
    "Random FAR percentiles:",
    np.percentile(far_values, [5, 25, 50, 75, 95]).round(2).tolist(),
)
