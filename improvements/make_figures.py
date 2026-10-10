"""Report-ready figures for the improvement analyses.

Four figures, each answering one question the submitted report leaves open:

1. risk_coverage.png      Does any gate rank its own errors better than plain
                          softmax confidence?
2. selective_gain.png     What does abstention buy, at coverage levels a
                          reader can hold the systems to?
3. control_baselines.png  How much of each label is recoverable from the
                          publisher alone?
4. effective_sample.png   How many independent evidence clusters is each test
                          set actually made of?

Palette is the validated five-slot categorical set (adjacent-pair CVD Delta E
9.1, normal-vision 19.6). Three slots sit below 3:1 against the light surface,
so every series carries a direct label and the underlying CSVs act as the table
view, which is the documented relief for that warning.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.patheffects

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "improvements"

# The analyses number three settings (zero-shot, source, combined). The report
# keeps its own two-study numbering and treats zero-shot as a diagnostic, so
# FIGURE_NAMING=report relabels every study and writes to a separate folder.
NAMING = os.environ.get("FIGURE_NAMING", "analysis")
REPORT_NAMES = {
    "Study 1 - Zero-shot UAE": "Zero-shot UAE",
    "Study 2 - Source-domain": "Study 1 - Source-domain",
    "Study 3 - Combined adaptation": "Study 2 - Combined adaptation",
}
FIGURES_DIR = RESULTS_DIR / ("figures_report" if NAMING == "report" else "figures")


def display(study: str) -> str:
    return REPORT_NAMES.get(study, study) if NAMING == "report" else study


def tick(study: str) -> str:
    """Two-line axis label, e.g. 'S2' over 'Combined adaptation'."""
    name = display(study)
    if " - " in name:
        head, tail = name.split(" - ", 1)
        return head.replace("Study ", "S") + "\n" + tail
    return name

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2dd"

# Validated categorical slots, assigned in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# The five selectors carried into the figures, in a fixed order so a colour
# always means the same selector across every panel.
SELECTOR_ORDER = [
    "MaxProb (confidence baseline)",
    "Margin (confidence baseline)",
    "FAAV-Light (shipped rule gate)",
    "FAAV-Full (shipped TF-IDF gate)",
    "FAAV-Rich (corrected gate)",
]
SHORT = {
    "MaxProb (confidence baseline)": "MaxProb",
    "Margin (confidence baseline)": "Margin",
    "FAAV-Light (shipped rule gate)": "FAAV-Light",
    "FAAV-Full (shipped TF-IDF gate)": "FAAV-Full",
    "FAAV-Rich (corrected gate)": "FAAV-Rich",
}
COLOR = {name: SERIES[i] for i, name in enumerate(SELECTOR_ORDER)}


def style_axes(ax) -> None:
    """Recessive grid and axes; the data carries the ink."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def new_figure(width: float, height: float):
    figure = plt.figure(figsize=(width, height), facecolor=SURFACE, dpi=200)
    return figure


def figure_risk_coverage() -> None:
    path = RESULTS_DIR / "risk_coverage_curves.csv"
    if not path.exists():
        return
    curves = pd.read_csv(path)
    study = "Study 3 - Combined adaptation"
    tasks = ["binary", "three_class"]

    figure = new_figure(11, 4.4)
    for index, task in enumerate(tasks):
        ax = figure.add_subplot(1, 2, index + 1)
        style_axes(ax)
        subset = curves[(curves["Study"] == study) & (curves["Task"] == task)]
        if subset.empty:
            continue
        for selector in SELECTOR_ORDER:
            # With two classes, max-probability and top-two margin are monotone
            # transformations of each other and produce identical curves, so
            # plotting both would draw one line twice and imply agreement
            # between two independent methods.
            if task == "binary" and selector.startswith("Margin"):
                continue
            series = subset[subset["Selector"] == selector].sort_values("Coverage")
            if series.empty:
                continue
            ax.plot(
                series["Coverage"],
                series["Risk"],
                color=COLOR[selector],
                linewidth=2.0,
                label=SHORT[selector],
                zorder=3,
            )
            # Label where the curves are furthest apart. Every curve meets at
            # coverage 1.0 by definition, so labelling the right-hand end
            # stacks all five on top of one another.
            anchor = series.iloc[(series["Coverage"] - 0.45).abs().argmin()]
            ax.annotate(
                SHORT[selector],
                xy=(anchor["Coverage"], anchor["Risk"]),
                xytext=(0, 5),
                textcoords="offset points",
                color=TEXT_SECONDARY,
                fontsize=7.5,
                ha="center",
                path_effects=[
                    matplotlib.patheffects.withStroke(
                        linewidth=2.5, foreground=SURFACE
                    )
                ],
            )
        ax.set_xlabel("Coverage (fraction of test set accepted)", fontsize=9,
                      color=TEXT_SECONDARY)
        ax.set_ylabel("Risk (error rate among accepted)", fontsize=9,
                      color=TEXT_SECONDARY)
        ax.set_title(
            f"{task.replace('_', '-')} task",
            fontsize=10,
            color=TEXT_PRIMARY,
            loc="left",
        )
        ax.set_xlim(0.05, 1.02)
    figure.suptitle(
        f"Risk-coverage curves, {display(study)} "
        "(lower is better; a useful gate bends below MaxProb)",
        fontsize=11,
        color=TEXT_PRIMARY,
        x=0.02,
        ha="left",
    )
    figure.legend(
        handles=[
            plt.Line2D([], [], color=COLOR[s], linewidth=2.0, label=SHORT[s])
            for s in SELECTOR_ORDER
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=[0, 0.08, 1, 0.93])
    figure.savefig(FIGURES_DIR / "risk_coverage.png", facecolor=SURFACE)
    plt.close(figure)


def figure_selective_gain() -> None:
    path = RESULTS_DIR / "selective_at_matched_coverage.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    study = "Study 3 - Combined adaptation"
    frame = frame[frame["Study"] == study]

    figure = new_figure(11, 4.4)
    for index, task in enumerate(["binary", "three_class"]):
        ax = figure.add_subplot(1, 2, index + 1)
        style_axes(ax)
        subset = frame[frame["Task"] == task]
        if subset.empty:
            continue
        targets = sorted(subset["Target coverage"].unique())
        width = 0.8 / max(len(SELECTOR_ORDER), 1)
        for slot, selector in enumerate(SELECTOR_ORDER):
            series = subset[subset["Selector"] == selector].sort_values(
                "Target coverage"
            )
            if series.empty:
                continue
            positions = [
                i + slot * width - 0.4 + width / 2 for i in range(len(targets))
            ]
            ax.bar(
                positions,
                series["selective_macro_f1"],
                width=width * 0.86,  # 2px-equivalent surface gap between bars
                color=COLOR[selector],
                label=SHORT[selector],
                zorder=3,
            )
        full = subset["selective_macro_f1"].min()
        baseline = frame[frame["Task"] == task]["Macro F1 gain over full coverage"]
        reference = (
            subset["selective_macro_f1"] - subset["Macro F1 gain over full coverage"]
        ).iloc[0]
        ax.axhline(
            reference,
            color=TEXT_SECONDARY,
            linewidth=1.2,
            linestyle="--",
            zorder=4,
        )
        ax.annotate(
            f"full coverage {reference:.3f}",
            xy=(len(targets) - 0.5, reference),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=TEXT_SECONDARY,
            ha="right",
        )
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([f"{t:.0%}" for t in targets])
        ax.set_xlabel("Target coverage", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylabel("Selective macro F1", fontsize=9, color=TEXT_SECONDARY)
        ax.set_title(
            f"{task.replace('_', '-')} task",
            fontsize=10,
            color=TEXT_PRIMARY,
            loc="left",
        )
        low = max(reference - 0.06, 0.0)
        ax.set_ylim(low, 1.0)
    figure.suptitle(
        f"What abstention buys: selective macro F1 at matched coverage, {display(study)}",
        fontsize=11,
        color=TEXT_PRIMARY,
        x=0.02,
        ha="left",
    )
    figure.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=COLOR[s], label=SHORT[s])
            for s in SELECTOR_ORDER
        ],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=[0, 0.08, 1, 0.93])
    figure.savefig(FIGURES_DIR / "selective_gain.png", facecolor=SURFACE)
    plt.close(figure)


def figure_control_baselines() -> None:
    path = RESULTS_DIR / "control_baselines.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    controls = [
        "Majority class",
        "Source lookup (provenance)",
    ]
    figure = new_figure(11, 4.6)
    for index, task in enumerate(["binary", "three_class"]):
        ax = figure.add_subplot(1, 2, index + 1)
        style_axes(ax)
        subset = frame[frame["Task"] == task]
        studies = list(subset["Study"].unique())
        width = 0.26
        for slot, control in enumerate(controls):
            values = [
                subset[
                    (subset["Study"] == study) & (subset["Control"] == control)
                ]["Macro F1"].iloc[0]
                for study in studies
            ]
            ax.bar(
                [i + slot * width - width for i in range(len(studies))],
                values,
                width=width * 0.88,
                color=SERIES[slot],
                label=control,
                zorder=3,
            )
        best = [
            subset[subset["Study"] == study]["Reported best macro F1"].iloc[0]
            for study in studies
        ]
        ax.bar(
            [i + 2 * width - width for i in range(len(studies))],
            best,
            width=width * 0.88,
            color=SERIES[2],
            label="Reported best model",
            zorder=3,
        )
        for i, value in enumerate(best):
            ax.annotate(
                f"{value:.3f}",
                xy=(i + 2 * width - width, value),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=TEXT_SECONDARY,
            )
        ax.set_xticks(range(len(studies)))
        ax.set_xticklabels([tick(s) for s in studies], fontsize=8)
        ax.set_ylabel("Macro F1", fontsize=9, color=TEXT_SECONDARY)
        ax.set_title(
            f"{task.replace('_', '-')} task",
            fontsize=10,
            color=TEXT_PRIMARY,
            loc="left",
        )
        ax.set_ylim(0, 1.0)
    figure.suptitle(
        "Provenance control: how much of the label the publisher alone explains",
        fontsize=11,
        color=TEXT_PRIMARY,
        x=0.02,
        ha="left",
    )
    figure.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SERIES[i], label=name)
            for i, name in enumerate(
                ["Majority class", "Source lookup (provenance)", "Reported best model"]
            )
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_SECONDARY,
    )
    figure.tight_layout(rect=[0, 0.08, 1, 0.93])
    figure.savefig(FIGURES_DIR / "control_baselines.png", facecolor=SURFACE)
    plt.close(figure)


def figure_effective_sample() -> None:
    path = RESULTS_DIR / "evidence_structure.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame[frame["Split"].str.contains("test|full")]
    frame = frame.sort_values("Rows")

    figure = new_figure(9.5, 4.8)
    ax = figure.add_subplot(1, 1, 1)
    style_axes(ax)
    positions = range(len(frame))
    ax.barh(
        [p + 0.19 for p in positions],
        frame["Rows"],
        height=0.34,
        color=SERIES[0],
        label="Rows",
        zorder=3,
    )
    ax.barh(
        [p - 0.19 for p in positions],
        frame["Effective N (evidence clusters)"],
        height=0.34,
        color=SERIES[1],
        label="Effective N (evidence clusters)",
        zorder=3,
    )
    for position, (_, row) in zip(positions, frame.iterrows()):
        ax.annotate(
            f"{row['Effective N (evidence clusters)']:.0f}"
            f"  ({row['CI inflation factor']:.1f}x wider CI)",
            xy=(row["Effective N (evidence clusters)"], position - 0.19),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(frame["Split"], fontsize=8)
    ax.set_xscale("log")
    # Headroom on the right so the Rows bars are not clipped and the inline
    # effective-N labels have room to sit beside their bars.
    ax.set_xlim(10, frame["Rows"].max() * 3.2)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_xlabel(
        "Sample size (log scale)", fontsize=9, color=TEXT_SECONDARY
    )
    ax.set_title(
        "Repeated evidence shrinks every test set's effective sample size",
        fontsize=11,
        color=TEXT_PRIMARY,
        loc="left",
    )
    # Legend above the plot: inside the axes it overlapped the lowest label.
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_SECONDARY,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "effective_sample.png", facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_control_baselines()
    figure_effective_sample()
    figure_risk_coverage()
    figure_selective_gain()
    written = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    print("figures written to", FIGURES_DIR)
    for name in written:
        print("  ", name)


if __name__ == "__main__":
    main()
