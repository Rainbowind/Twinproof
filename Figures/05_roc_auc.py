from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


RANDOM_SEED = 20260715
GENUINE_SAMPLES = 600
ATTACK_SAMPLES = 400

ATTACK_DISTRIBUTIONS = {
    "Forged": (0.700, 0.095),
    "Replay": (0.740, 0.100),
    "Proxy": (0.725, 0.095),
    "Trans.": (0.715, 0.090),
}

CURVE_ORDER = ["Overall", "Forged", "Replay", "Proxy", "Trans."]
PLOT_ORDER = ["Forged", "Replay", "Proxy", "Trans.", "Overall"]

LINE_STYLES = {
    "Forged": {"color": "#4874CB", "linestyle": "-"},
    "Replay": {"color": "#668DD5", "linestyle": "--"},
    "Proxy": {"color": "#8BADE2", "linestyle": "-."},
    "Trans.": {"color": "#AFC9ED", "linestyle": ":"},
    "Overall": {"color": "#EE822F", "linestyle": "-"},
}


def generate_scores():
    rng = np.random.default_rng(RANDOM_SEED)
    genuine = np.clip(rng.normal(0.860, 0.075, GENUINE_SAMPLES), 0.0, 1.0)

    frames = [
        pd.DataFrame(
            {
                "Sample_ID": [f"genuine_{index + 1:04d}" for index in range(GENUINE_SAMPLES)],
                "Class_Name": "Genuine",
                "Label": 1,
                "S_total": genuine,
                "Source": "synthetic_for_roc",
            }
        )
    ]

    for class_name, (mean, std) in ATTACK_DISTRIBUTIONS.items():
        values = np.clip(rng.normal(mean, std, ATTACK_SAMPLES), 0.0, 1.0)
        prefix = class_name.lower().replace(".", "").replace(" ", "_")
        frames.append(
            pd.DataFrame(
                {
                    "Sample_ID": [f"{prefix}_{index + 1:04d}" for index in range(ATTACK_SAMPLES)],
                    "Class_Name": class_name,
                    "Label": 0,
                    "S_total": values,
                    "Source": "synthetic_for_roc",
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def calculate_curve(scores, class_name=None):
    if class_name is None:
        selected = scores
    else:
        selected = scores[scores["Class_Name"].isin(["Genuine", class_name])]

    far, tpr, thresholds = roc_curve(
        selected["Label"],
        selected["S_total"],
        pos_label=1,
        drop_intermediate=False,
    )
    return far, tpr, thresholds, auc(far, tpr)


def main():
    scores = generate_scores()

    score_path = Path(__file__).with_name("05_roc_auc_scores.csv")
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")

    curves = {}
    point_frames = []
    for curve_name in CURVE_ORDER:
        class_name = None if curve_name == "Overall" else curve_name
        far, tpr, thresholds, area = calculate_curve(scores, class_name)
        curves[curve_name] = (far, tpr, area)
        point_frames.append(
            pd.DataFrame(
                {
                    "Curve": curve_name,
                    "FAR": far,
                    "TPR": tpr,
                    "Threshold": thresholds,
                    "AUC": area,
                }
            )
        )

    point_path = Path(__file__).with_name("05_roc_auc_points.csv")
    pd.concat(point_frames, ignore_index=True).to_csv(
        point_path,
        index=False,
        encoding="utf-8-sig",
    )

    cm_to_in = 1 / 2.54
    fig_w = 4.1 * cm_to_in
    fig_h = 2.5 * cm_to_in

    plt.rcParams.update(
        {
            "font.size": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 3.5,
            "font.family": "sans-serif",
        }
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    line_handles = {}
    for curve_name in PLOT_ORDER:
        far, tpr, area = curves[curve_name]
        style = LINE_STYLES[curve_name]
        is_overall = curve_name == "Overall"
        line, = ax.plot(
            far,
            tpr,
            label=f"{curve_name} (AUC={area:.3f})",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.3 if is_overall else 0.9,
            drawstyle="steps-post",
            zorder=3 if is_overall else 2,
        )
        line_handles[curve_name] = line

    ax.plot(
        [0, 1],
        [0, 1],
        color="#A6A6A6",
        linestyle="--",
        linewidth=0.6,
        zorder=1,
    )

    ax.set_xlabel("FAR", labelpad=1)
    ax.set_ylabel("TPR", labelpad=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="both", pad=1, width=0.5, length=2)

    for spine in ["left", "right", "top", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.6)

    legend_handles = [line_handles[name] for name in CURVE_ORDER]
    legend_labels = [line_handles[name].get_label() for name in CURVE_ORDER]
    legend = ax.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        frameon=True,
        borderpad=0.18,
        handlelength=1.0,
        handletextpad=0.25,
        labelspacing=0.14,
        bbox_to_anchor=(0.99, 0.02),
    )

    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.82)
    frame.set_edgecolor("none")
    frame.set_linewidth(0.0)

    plt.subplots_adjust(
        left=0.20,
        right=0.95,
        bottom=0.26,
        top=0.95,
    )

    output_path = Path(__file__).with_suffix(".pdf")
    plt.savefig(output_path, format="pdf")
    plt.close(fig)

    print(f"[done] figure: {output_path}")
    print(f"[done] scores: {score_path}")
    print(f"[done] points: {point_path}")
    for curve_name in CURVE_ORDER:
        print(f"{curve_name}: AUC={curves[curve_name][2]:.4f}")


if __name__ == "__main__":
    main()
