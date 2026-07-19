from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("docs/JAIR")
INK = "#1f2937"
MUTED = "#5b6570"
ARROW = "#59636e"
COLORS = {
    "data": ("#e8f2fb", "#3f7cac"),
    "select": ("#e8f6f3", "#23856d"),
    "density": ("#edf6e9", "#568a45"),
    "model": ("#f0eafa", "#7154a3"),
    "eval": ("#fbeceb", "#b95c55"),
}


def box(ax, x, y, width, height, label, kind, size=8.5):
    fill, edge_color = COLORS[kind]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.06",
            facecolor=fill,
            edgecolor=edge_color,
            linewidth=1.15,
            zorder=2,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        linespacing=1.25,
        zorder=3,
    )
    return (x, y, width, height)


def band(ax, x, y, width, height, title, fill, edge_color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(
        x + 0.18,
        y + height - 0.18,
        title,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=INK,
    )


def center(rect):
    x, y, width, height = rect
    return x + width / 2, y + height / 2


def edge(rect, target):
    x, y, width, height = rect
    center_x, center_y = center(rect)
    dx, dy = target[0] - center_x, target[1] - center_y
    if abs(dx) / max(width, 1e-9) > abs(dy) / max(height, 1e-9):
        return (
            x + width if dx > 0 else x,
            center_y + dy * (width / 2) / max(abs(dx), 1e-9),
        )
    return (
        center_x + dx * (height / 2) / max(abs(dy), 1e-9),
        y + height if dy > 0 else y,
    )


def arrow(ax, start, end, label=None, label_offset=(0, 0), color=ARROW):
    start_point = edge(start, center(end))
    end_point = edge(end, center(start))
    ax.add_patch(
        FancyArrowPatch(
            start_point,
            end_point,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=1,
        )
    )
    if label:
        ax.text(
            (start_point[0] + end_point[0]) / 2 + label_offset[0],
            (start_point[1] + end_point[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7.7,
            color=MUTED,
            zorder=4,
        )


def elbow(ax, points, label=None, label_xy=None, color=ARROW):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=1.15,
            zorder=1,
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=color,
            shrinkA=0,
            shrinkB=2,
            zorder=1,
        )
    )
    if label and label_xy:
        ax.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=7.7,
            color=MUTED,
            zorder=4,
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.4, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 15.4)
    ax.set_ylim(0, 5.7)
    ax.axis("off")

    ax.text(
        7.7,
        5.45,
        "FairReweighing workflow",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )

    dataset = box(ax, 0.25, 4.02, 1.35, 0.78, "Dataset\n$(X,A,Y)$", "data", size=9.0)
    split = box(ax, 1.88, 4.02, 1.42, 0.78, "Outer split\ntrain / test", "data", size=8.8)
    arrow(ax, dataset, split)

    band(ax, 3.58, 2.55, 11.52, 2.48, "TRAINING DATA ONLY", "#f8fafb", "#aeb8c2")
    select = box(
        ax,
        3.92,
        3.22,
        2.75,
        1.16,
        "$\\mathbf{1}$  Dynamic selection\n"
        "inner fit / validation split\n"
        "grid search $r$ or $h$\n"
        "minimize $\\hat C_{sep}^{xfit}+0.01$ MSE",
        "select",
        size=8.2,
    )
    density = box(
        ax,
        7.02,
        3.22,
        2.70,
        1.16,
        "$\\mathbf{2}$  Density estimation\n"
        "z-score $A$, $Y$, $(A,Y)$\n"
        "radius counts or Gaussian KDE\n"
        "$\\hat\\rho_A$, $\\hat\\rho_Y$, $\\hat\\rho_{A,Y}$",
        "density",
        size=8.2,
    )
    train = box(
        ax,
        10.08,
        3.22,
        3.02,
        1.16,
        "$\\mathbf{3}$  Reweight and train\n"
        "$\\hat w_i\\propto\\hat\\rho_A(a_i)\\hat\\rho_Y(y_i)"
        "/\\hat\\rho_{A,Y}(a_i,y_i)$\n"
        "fit weighted model $M^*$",
        "model",
        size=8.3,
    )
    arrow(ax, split, select, label="train", label_offset=(0, 0.14))
    arrow(
        ax,
        select,
        density,
        label="$r^*$ or $h^*$",
        label_offset=(0, 0.14),
        color=COLORS["select"][1],
    )
    arrow(ax, density, train)

    band(ax, 0.25, 0.20, 14.85, 1.92, "HELD-OUT TEST EVALUATION", "#fbf9f7", "#c6b9ad")
    test = box(
        ax,
        0.62,
        0.72,
        1.62,
        0.76,
        "Test data\n$(X_{te},A_{te},Y_{te})$",
        "data",
        size=8.4,
    )
    predict = box(
        ax,
        2.58,
        0.72,
        1.72,
        0.76,
        "Predictions\n$\\hat Y=M^*(X_{te},A_{te})$",
        "model",
        size=8.3,
    )
    evaluate = box(
        ax,
        4.70,
        0.56,
        4.62,
        1.08,
        "$\\mathbf{4}$  Evaluation\n"
        "Prediction: MSE, MAE, $R^2$ or accuracy, F1\n"
        "Separation: $K$-fold held-out $\\hat C_{sep}^{xfit}$ log-ratios",
        "eval",
        size=8.5,
    )
    report = box(
        ax,
        9.72,
        0.72,
        2.18,
        0.76,
        "Report\nprediction performance\nand $\\hat C_{sep}^{xfit}$",
        "eval",
        size=8.1,
    )

    elbow(
        ax,
        [(2.59, 4.02), (2.59, 2.32), (1.43, 2.32), (1.43, 1.48)],
        label="test",
        label_xy=(2.38, 2.42),
    )
    elbow(
        ax,
        [(11.59, 3.22), (11.59, 2.32), (3.44, 2.32), (3.44, 1.48)],
        label="$M^*$",
        label_xy=(7.52, 2.42),
        color=COLORS["model"][1],
    )
    arrow(ax, test, predict)
    arrow(ax, predict, evaluate)
    arrow(ax, evaluate, report)

    ax.text(
        7.7,
        0.30,
        "Test data are used only for final prediction and evaluation.",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color=MUTED,
    )

    for suffix, options in (("pdf", {}), ("png", {"dpi": 210})):
        fig.savefig(
            OUTPUT_DIR / f"fig1_crossfit_workflow.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **options,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
