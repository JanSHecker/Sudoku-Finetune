import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS = Path("artifacts/qwen2.5-1.5b-banking77-qlora/metrics.json")
OUTPUT = Path("results-seaborn.png")


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    baseline = metrics["baseline"]
    tuned = metrics["fine_tuned"]
    labels = pd.DataFrame.from_dict(tuned["per_label"], orient="index")
    labels.index.name = "intent"
    labels = labels.reset_index()
    baseline_labels = pd.DataFrame.from_dict(baseline["per_label"], orient="index")
    baseline_labels.index.name = "intent"
    baseline_labels = baseline_labels.reset_index()
    intent_scores = baseline_labels.merge(labels, on="intent", suffixes=("_base", "_qlora"))
    assert len(intent_scores) == 77

    comparison = pd.DataFrame(
        [
            {"Metric": name, "Model": model, "Score": source[key]}
            for name, key in [
                ("Accuracy", "accuracy"),
                ("Macro F1", "macro_f1"),
                ("Macro recall", "macro_recall"),
                ("Valid labels", "valid_label_rate"),
            ]
            for model, source in [("Base", baseline), ("QLoRA", tuned)]
        ]
    )
    confusions = pd.DataFrame(tuned["top_confusions"])
    confusions["Pair"] = confusions["expected"] + "  →  " + confusions["predicted"]
    confusions = confusions.sort_values("count")
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f6f3",
            "axes.facecolor": "#f5f6f3",
            "axes.edgecolor": "#8c9690",
            "grid.color": "#d9dedb",
            "text.color": "#17201b",
            "axes.labelcolor": "#17201b",
            "xtick.color": "#59635d",
            "ytick.color": "#59635d",
            "font.family": "DejaVu Sans",
        }
    )

    figure = plt.figure(figsize=(18, 28), layout="constrained")
    grid = figure.add_gridspec(3, 2, height_ratios=[1.2, 1.7, 5.5])
    comparison_ax = figure.add_subplot(grid[0, 0])
    scatter_ax = figure.add_subplot(grid[0, 1])
    confusion_ax = figure.add_subplot(grid[1, :])
    heatmap_ax = figure.add_subplot(grid[2, :])

    sns.barplot(
        data=comparison,
        x="Score",
        y="Metric",
        hue="Model",
        palette={"Base": "#a3aaa6", "QLoRA": "#23744d"},
        ax=comparison_ax,
    )
    comparison_ax.set(title="Overall metrics", xlabel="Score", ylabel="", xlim=(0, 1.08))
    comparison_ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for container in comparison_ax.containers:
        comparison_ax.bar_label(
            container,
            labels=[f"{value:.1%}" for value in container.datavalues],
            padding=4,
            fontsize=9,
        )
    comparison_ax.legend(title="", frameon=False, loc="lower right")
    sns.despine(ax=comparison_ax)

    sns.scatterplot(
        data=intent_scores,
        x="f1_base",
        y="f1_qlora",
        color="#23744d",
        s=72,
        edgecolor="white",
        linewidth=0.7,
        ax=scatter_ax,
    )
    scatter_ax.plot([0, 1], [0, 1], color="#8c9690", linewidth=1, linestyle="--")
    scatter_ax.set(
        title="Intent-level F1: base versus QLoRA",
        xlabel="Base F1",
        ylabel="QLoRA F1",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    scatter_ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    scatter_ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    annotation_positions = {
        "card_payment_not_recognised": (6, -14),
        "pending_transfer": (6, -14),
        "topping_up_by_card": (6, 8),
        "balance_not_updated_after_bank_transfer": (6, 8),
    }
    for row in intent_scores.nsmallest(4, "f1_qlora").itertuples():
        x_offset, y_offset = annotation_positions[row.intent]
        scatter_ax.annotate(
            row.intent.replace("_", " "),
            (row.f1_base, row.f1_qlora),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left",
            fontsize=7,
        )
    sns.despine(ax=scatter_ax)

    sns.barplot(
        data=confusions,
        x="count",
        y="Pair",
        color="#ae6329",
        ax=confusion_ax,
    )
    confusion_ax.set(
        title="Most frequent remaining confusions",
        xlabel="Test examples",
        ylabel="",
        xlim=(0, confusions["count"].max() + 1.5),
    )
    confusion_ax.bar_label(confusion_ax.containers[0], padding=4)
    confusion_ax.grid(axis="y", visible=False)
    sns.despine(ax=confusion_ax)

    heatmap_data = intent_scores.set_index("intent")[
        ["precision_base", "recall_base", "f1_base", "precision_qlora", "recall_qlora", "f1_qlora"]
    ].sort_values("f1_qlora")
    heatmap_data.columns = [
        "Base precision",
        "Base recall",
        "Base F1",
        "QLoRA precision",
        "QLoRA recall",
        "QLoRA F1",
    ]
    sns.heatmap(
        heatmap_data,
        cmap=sns.light_palette("#23744d", as_cmap=True),
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        linewidths=0.4,
        linecolor="#e3e7e4",
        cbar_kws={"label": "Score", "shrink": 0.35},
        annot_kws={"fontsize": 7},
        ax=heatmap_ax,
    )
    heatmap_ax.axvline(3, color="#f5f6f3", linewidth=4)
    heatmap_ax.set(title="All 77 intent scores, ordered by QLoRA F1", xlabel="", ylabel="")
    heatmap_ax.tick_params(axis="y", labelsize=7)
    heatmap_ax.tick_params(axis="x", rotation=0)

    figure.suptitle(
        "QLoRA BANKING77 evaluation\nQwen2.5-1.5B-Instruct · 3,076 held-out examples",
        fontsize=20,
        fontweight="bold",
    )
    figure.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    print(f"Saved {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
