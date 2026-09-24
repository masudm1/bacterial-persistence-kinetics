"""Generate the PINN architecture diagram used as Figure 3."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
    }
)


def box(ax, xy, width, height, face, edge, title, body):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.08,rounding_size=0.10",
        linewidth=1.7,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#202124",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=9.2,
        color="#303236",
        linespacing=1.25,
    )
    return patch


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "linewidth": 1.55,
            "color": "#3c4043",
            "mutation_scale": 14,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def make_figure(output=HERE / "fig2_architecture.png"):
    fig, ax = plt.subplots(figsize=(12.0, 5.1))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    box(
        ax,
        (0.25, 2.25),
        1.75,
        1.5,
        "#f2f3f5",
        "#5f6368",
        "Input",
        "Normalized time\n(t/T)",
    )
    box(
        ax,
        (2.75, 1.65),
        3.0,
        2.7,
        "#e8eaed",
        "#4f5358",
        "State network",
        "2 hidden layers\n32 tanh units per layer\nDouble precision",
    )
    box(
        ax,
        (6.55, 1.65),
        3.2,
        2.7,
        "#e8f0fe",
        "#2457a6",
        "Latent log states",
        "u(t) = log N(t)\nv(t) = log P(t)\nInitial state imposed exactly",
    )
    box(
        ax,
        (10.55, 1.35),
        3.15,
        3.3,
        "#e6f4ea",
        "#287a3d",
        "Composite objective",
        "Log-scale data mismatch\n+\nODE residuals from\nautomatic differentiation",
    )
    box(
        ax,
        (6.55, 0.10),
        3.2,
        1.05,
        "#fce8e6",
        "#a52a2a",
        "Trainable rates",
        "alpha > 0,  beta > 0 via softplus\nTreatment-only fits: D < 0",
    )

    arrow(ax, (2.02, 3.0), (2.73, 3.0))
    arrow(ax, (5.77, 3.0), (6.53, 3.0))
    arrow(ax, (9.77, 3.0), (10.53, 3.0))
    arrow(ax, (8.15, 1.17), (10.53, 1.85))

    ax.text(
        8.15,
        5.35,
        "Physics-informed estimation of hidden population states",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#202124",
    )
    ax.text(
        8.15,
        4.90,
        "The neural state representation and kinetic rates are optimized jointly",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#5f6368",
    )

    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    make_figure()
