from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")

# =====================
# Data
# Each claim consists of consecutive 8-second observation windows.
# =====================
output_dir = Path(__file__).resolve().parent
data_path = output_dir / "05_claim_duration_robustness_data.csv"

df = pd.read_csv(data_path)

windows = df["Windows"].to_numpy()
forged = df["Forged"].to_numpy()
replay = df["Replay"].to_numpy()
proxy = df["Proxy"].to_numpy()
transplant = df["Trans."].to_numpy()
overall = df["Overall"].to_numpy()

# =====================
# Figure & font
# =====================
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

cm_to_in = 1 / 2.54
fig_w = 4.1 * cm_to_in
fig_h = 2.5 * cm_to_in

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =====================
# Curves
# =====================
series = [
    ("Forged", forged, "#2F5FB8", "-", "s", 0.65),
    ("Replay", replay, "#4874CB", "--", "^", 0.65),
    ("Proxy", proxy, "#638ED8", "-.", "D", 0.65),
    ("Trans.", transplant, "#86AEEA", ":", "v", 0.75),
    ("Overall", overall, "#EE822F", "-", "o", 0.9),
]

for label, values, color, linestyle, marker, linewidth in series:
    ax.plot(
        windows,
        values,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=2.4,
        markeredgewidth=0.55,
        markerfacecolor="white",
        label=label,
    )

# =====================
# Axes
# =====================
ax.set_xlabel("Number of Windows", labelpad=1)
ax.set_ylabel("FAR (%)", labelpad=1)

ax.set_xlim(1, 8)
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
ax.set_ylim(0, 15)
ax.set_yticks([0, 5, 10, 15])

ax.tick_params(axis="both", length=2, width=0.5, pad=1)

# =====================
# Legend
# =====================
leg = ax.legend(
    loc="upper right",
    ncol=2,
    frameon=True,
    handlelength=1.5,
    handletextpad=0.35,
    columnspacing=0.65,
    borderpad=0.18,
    labelspacing=0.2,
)

leg.get_frame().set_facecolor("white")
leg.get_frame().set_alpha(0.72)
leg.get_frame().set_linewidth(0.0)

# =====================
# Border
# =====================
for sp in ["left", "right", "top", "bottom"]:
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)

# =====================
# Layout & save
# =====================
plt.subplots_adjust(left=0.20, right=0.95, bottom=0.22, top=0.95)

pdf_path = output_dir / "05_claim_duration_robustness.pdf"
fig.savefig(pdf_path, format="pdf")

print(f"Saved PDF: {pdf_path}")
print(f"Loaded CSV: {data_path}")
