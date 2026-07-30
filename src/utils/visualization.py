"""Plotting utilities for EDA, augmentation, training curves, and evaluation.

Static (matplotlib) figures for notebooks, README, and the LaTeX presentation.
Palette follows the project's data-viz conventions: fixed categorical hues
(never cycled), a single-hue sequential ramp for magnitude (no rainbow
colormaps), muted gridlines, and legends whenever more than one series appears.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.config import FIGURES_DIR

# Categorical slots (fixed order — see project data-viz conventions)
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_RED = "#e34948"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"]
)


def _style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def savefig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    return path


def plot_class_distribution(counts, title="Training Images per Class"):
    fig, ax = plt.subplots(figsize=(10, 4))
    sorted_counts = np.sort(counts)[::-1]

    # Sequential single-hue encoding (magnitude, not identity): darker bar = more
    # images. Redundant with bar height by design — it makes the imbalance
    # readable at a glance without a 102-color categorical palette, which would
    # be visual noise (see project data-viz conventions). Clamped to 0.25-1.0 so
    # the lightest bar stays visible against the surface instead of receding to
    # near-white.
    span = sorted_counts.max() - sorted_counts.min()
    norm = (sorted_counts - sorted_counts.min()) / span if span > 0 else np.ones_like(sorted_counts, dtype=float)
    bar_colors = SEQUENTIAL_BLUE(0.25 + norm * 0.75)

    ax.bar(range(len(sorted_counts)), sorted_counts, color=bar_colors, width=1.0)
    _style_axis(ax)
    ax.set_xlabel("Class (sorted by image count)", color=INK_SECONDARY)
    ax.set_ylabel("Image count", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    return fig


def plot_image_size_distribution(widths, heights, title="Original Image Dimensions (sample)"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, heights, color=COLOR_BLUE, alpha=0.5, s=14, edgecolors="none")
    _style_axis(ax)
    ax.set_xlabel("Width (px)", color=INK_SECONDARY)
    ax.set_ylabel("Height (px)", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    return fig


def plot_aspect_ratio_distribution(aspect_ratios, title="Aspect Ratio (width / height)"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(aspect_ratios, bins=40, color=COLOR_BLUE, edgecolor=SURFACE, linewidth=0.3)
    ax.axvline(1.0, color=COLOR_ORANGE, linewidth=1.5, linestyle="--", label="Square (1:1)")
    _style_axis(ax)
    ax.set_xlabel("Width / height", color=INK_SECONDARY)
    ax.set_ylabel("Number of images", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_resolution_histograms(heights, widths, title_prefix="Image Resolution"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(heights, bins=40, color=COLOR_BLUE, edgecolor=SURFACE, linewidth=0.3)
    axes[0].set_title(f"{title_prefix}: Height", color=INK_PRIMARY, loc="left")
    axes[0].set_xlabel("Height (px)", color=INK_SECONDARY)
    axes[0].set_ylabel("Number of images", color=INK_SECONDARY)
    _style_axis(axes[0])

    axes[1].hist(widths, bins=40, color=COLOR_ORANGE, edgecolor=SURFACE, linewidth=0.3)
    axes[1].set_title(f"{title_prefix}: Width", color=INK_PRIMARY, loc="left")
    axes[1].set_xlabel("Width (px)", color=INK_SECONDARY)
    axes[1].set_ylabel("Number of images", color=INK_SECONDARY)
    _style_axis(axes[1])

    fig.tight_layout()
    return fig


def plot_labeled_image_examples(images, labels, title, ncols=None):
    n = len(images)
    ncols = ncols or min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.5))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].astype("uint8"))
            ax.set_title(labels[i], fontsize=8, color=INK_PRIMARY)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()
    return fig


def plot_sample_grid(images, labels, class_names, title, n=12, ncols=6):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.3))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].astype("uint8"))
            ax.set_title(class_names[labels[i]], fontsize=8, color=INK_PRIMARY)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()
    return fig


def plot_augmentation_examples(original, augmented_variants, title="Augmentation Examples"):
    n = len(augmented_variants) + 1
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.6))
    axes[0].imshow(original.astype("uint8"))
    axes[0].set_title("Original", fontsize=9, color=INK_PRIMARY)
    for i, img in enumerate(augmented_variants, start=1):
        axes[i].imshow(img.astype("uint8"))
        axes[i].set_title(f"Augmented {i}", fontsize=9, color=INK_PRIMARY)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, color=INK_PRIMARY, fontsize=12)
    fig.tight_layout()
    return fig


def plot_training_curves(history_dict, title_prefix, metric="accuracy"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history_dict[metric], color=COLOR_BLUE, linewidth=2, label="Train")
    axes[0].plot(
        history_dict[f"val_{metric}"], color=COLOR_ORANGE, linewidth=2, label="Validation"
    )
    axes[0].set_title(f"{title_prefix}: {metric.capitalize()}", color=INK_PRIMARY, loc="left")
    axes[0].legend(frameon=False, labelcolor=INK_SECONDARY)
    _style_axis(axes[0])

    axes[1].plot(history_dict["loss"], color=COLOR_BLUE, linewidth=2, label="Train")
    axes[1].plot(history_dict["val_loss"], color=COLOR_ORANGE, linewidth=2, label="Validation")
    axes[1].set_title(f"{title_prefix}: Loss", color=INK_PRIMARY, loc="left")
    axes[1].legend(frameon=False, labelcolor=INK_SECONDARY)
    _style_axis(axes[1])

    fig.tight_layout()
    return fig


def plot_model_comparison(names, accuracies, title="Test Accuracy: Baseline vs. Transfer Learning"):
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLOR_BLUE, COLOR_ORANGE][: len(names)]
    bars = ax.bar(names, accuracies, color=colors, width=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            fontsize=10,
            color=INK_PRIMARY,
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, loc="left", fontsize=12)
    _style_axis(ax)
    fig.tight_layout()
    return fig


def plot_most_confused_pairs(pairs, title="Most Frequently Confused Class Pairs"):
    labels = [f"{t} -> {p}" for t, p, _ in pairs]
    counts = [c for _, _, c in pairs]
    fig, ax = plt.subplots(figsize=(8, max(4, len(pairs) * 0.35)))
    ax.barh(labels[::-1], counts[::-1], color=COLOR_RED)
    ax.set_xlabel("Test-set count", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, loc="left", fontsize=12)
    _style_axis(ax)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_per_class_f1_extremes(per_class, title="Per-Class F1: Best and Worst 10 Classes", k=10):
    worst = per_class[:k]
    best = per_class[-k:]
    combined = worst + best
    names = [c for c, _, _ in combined]
    f1s = [f for _, f, _ in combined]
    colors = [COLOR_RED] * len(worst) + [COLOR_AQUA] * len(best)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, f1s, color=colors)
    ax.set_xlabel("F1-score", color=INK_SECONDARY)
    ax.set_xlim(0, 1.0)
    ax.set_title(title, color=INK_PRIMARY, loc="left", fontsize=12)
    _style_axis(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_prediction_examples(images, true_names, pred_names, confidences, correct, title):
    n = len(images)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.6))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].astype("uint8"))
            color = COLOR_AQUA if correct[i] else COLOR_RED
            label = f"True: {true_names[i]}\nPred: {pred_names[i]} ({confidences[i]:.0%})"
            ax.set_title(label, fontsize=7.5, color=color)
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13)
    fig.tight_layout()
    return fig


def plot_reliability_diagram(bin_confidences, bin_accuracies, bin_counts, title="Reliability Diagram"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle="--", label="Perfect calibration")

    sizes = 20 + 200 * (bin_counts / max(bin_counts.max(), 1))
    has_data = bin_counts > 0
    ax.scatter(
        bin_confidences[has_data], bin_accuracies[has_data],
        s=sizes[has_data], color=COLOR_BLUE, alpha=0.85, edgecolors="none", zorder=3,
        label="Model (bin size = # predictions)",
    )
    ax.plot(bin_confidences[has_data], bin_accuracies[has_data], color=COLOR_BLUE, linewidth=1, alpha=0.5, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style_axis(ax)
    ax.set_xlabel("Confidence (predicted probability)", color=INK_SECONDARY)
    ax.set_ylabel("Actual accuracy", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_confidence_distribution(conf_correct, conf_incorrect, title="Prediction Confidence: Correct vs. Incorrect"):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf_correct, bins=bins, color=COLOR_AQUA, alpha=0.75, label=f"Correct (n={len(conf_correct)})")
    ax.hist(conf_incorrect, bins=bins, color=COLOR_RED, alpha=0.75, label=f"Incorrect (n={len(conf_incorrect)})")
    _style_axis(ax)
    ax.set_xlabel("Prediction confidence", color=INK_SECONDARY)
    ax.set_ylabel("Number of test examples", color=INK_SECONDARY)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_risk_coverage_curve(thresholds, coverages, accuracies, title="Confidence-Threshold Trade-off"):
    # Single shared axis (not dual-axis): both series are already fractions in
    # [0, 1], so indexing them to a common scale is both correct and avoids
    # the two-y-scale anti-pattern.
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(thresholds, coverages, color=COLOR_BLUE, linewidth=2, marker="o", markersize=3, label="Coverage (fraction kept)")
    ax.plot(thresholds, accuracies, color=COLOR_ORANGE, linewidth=2, marker="o", markersize=3, label="Selective accuracy (on kept predictions)")

    _style_axis(ax)
    ax.set_xlabel("Confidence threshold (minimum to keep a prediction)", color=INK_SECONDARY)
    ax.set_ylabel("Fraction", color=INK_SECONDARY)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower left")
    fig.tight_layout()
    return fig


def plot_pipeline_overview(title="Project Pipeline"):
    """A linear box-and-arrow diagram of the end-to-end workflow, from raw
    dataset to final selected model. Purely illustrative (no data plotted).
    """
    stages = [
        "Dataset", "EDA", "Preprocessing", "Data\nAugmentation",
        "Baseline\nCNN", "Transfer\nLearning", "Fine-tuning",
        "Evaluation", "Best\nModel",
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(1.6 * n, 2.2))

    box_w, box_h, gap = 1.3, 0.9, 0.35
    for i, stage in enumerate(stages):
        x = i * (box_w + gap)
        is_last = i == n - 1
        face = COLOR_ORANGE if is_last else COLOR_BLUE
        rect = plt.Rectangle(
            (x, 0), box_w, box_h, facecolor=face, alpha=0.15 if not is_last else 0.25,
            edgecolor=face, linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2, box_h / 2, stage, ha="center", va="center",
            fontsize=8.5, color=INK_PRIMARY, zorder=3,
        )
        if not is_last:
            arrow_x = x + box_w
            ax.annotate(
                "", xy=(arrow_x + gap, box_h / 2), xytext=(arrow_x, box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_BLUE, linewidth=1.3),
                zorder=1,
            )

    ax.set_xlim(-0.2, n * (box_w + gap))
    ax.set_ylim(-0.3, box_h + 0.3)
    ax.axis("off")
    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left")
    fig.tight_layout()
    return fig
