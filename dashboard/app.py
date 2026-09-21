"""Streamlit dashboard for model training and benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.catalog import artifact_fingerprint, discover_runs, summarize_run
from dashboard.metrics import UNPARSED_LABEL


st.set_page_config(
    page_title="Fine-tune evaluation lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#13202b; --muted:#607180; --line:#d5e0e5; --signal:#ee714b; --teal:#1d9b92; --violet:#6559d6; --paper:#f3f7f8; }
        .stApp { background:var(--paper); color:var(--ink); }
        .block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1500px; }
        h1,h2,h3,h4,p,div,span,label { font-family:'Space Grotesk',system-ui,sans-serif; }
        code, .metric-value, [data-testid="stMetricValue"] { font-family:'IBM Plex Mono',ui-monospace,monospace !important; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-top:3px solid var(--teal); padding:1rem 1.1rem; min-height:7.2rem; box-shadow:0 4px 16px rgba(19,32,43,.04); }
        [data-testid="stMetricLabel"] { color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }
        [data-testid="stMetricValue"] { color:var(--ink); font-size:1.7rem; }
        .eyebrow { color:var(--signal); font:600 .72rem 'IBM Plex Mono',monospace; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.55rem; }
        .hero { border-bottom:1px solid var(--line); padding-bottom:1.4rem; margin-bottom:1.4rem; }
        .hero h1 { font-size:clamp(2rem,4vw,3.6rem); letter-spacing:-.06em; line-height:.98; margin:0; }
        .hero p { color:var(--muted); max-width:62rem; margin:.8rem 0 0; font-size:1rem; }
        .section-label { color:var(--muted); font:600 .72rem 'IBM Plex Mono',monospace; letter-spacing:.11em; text-transform:uppercase; margin:1.5rem 0 .65rem; }
        .metadata { background:#e5eef1; padding:.7rem .9rem; border-left:3px solid var(--violet); color:#38505e; font-size:.82rem; }
        .note { background:#fff6ed; border-left:3px solid var(--signal); padding:.8rem 1rem; color:#70412f; font-size:.88rem; }
        [data-testid="stSidebar"] { background:#15232d; }
        [data-testid="stSidebar"] * { color:#e9f0f2; }
        [data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stTextInput label { color:#aac0c8; }
        [data-testid="stSidebar"] hr { border-color:#31434e; }
        .small-caption { color:var(--muted); font-size:.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}%}"


def seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    return f"{value:.2f}s" if value < 100 else f"{value / 60:.1f}m"


def safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def cached_summaries(root: str, fingerprint: tuple[tuple[str, int, int], ...]) -> list[dict[str, Any]]:
    del fingerprint
    return [summarize_run(run) for run in discover_runs(root)]


def metric(summary: dict[str, Any], name: str) -> float | None:
    classification = summary["derived"].get("classification") or {}
    if name == "macro_f1":
        return classification.get("macro", {}).get("f1")
    if name == "macro_recall":
        return classification.get("macro", {}).get("recall")
    if name == "weighted_f1":
        return classification.get("weighted", {}).get("f1")
    if name == "accuracy":
        return classification.get("accuracy")
    if name == "parse_rate":
        return summary["derived"].get("rates", {}).get("parseable")
    for field in ("sound_deduction", "valid_deduction", "valid_rule", "exact_target"):
        if name == field:
            return summary["derived"].get("rates", {}).get(field)
    for field in ("schema_valid", "task_correct", "exact_answer"):
        if name == field:
            return summary["derived"].get("rates", {}).get(field)
    return None


def display_name(summary: dict[str, Any]) -> str:
    return summary["run"].display_name


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Fine-tune evaluation lab / live artifact index</div>
          <h1>Measure what the model<br>gets wrong.</h1>
          <p>Compare training runs and benchmarks without letting class imbalance hide behind a single accuracy number.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata(summary: dict[str, Any]) -> None:
    run = summary["run"]
    data = summary["data"]
    bits = [
        f"<b>{run.kind}</b>",
        f"model: <b>{run.model}</b>",
        f"schema: <code>{run.schema}</code>",
        f"artifact: <code>{run.path.name}</code>",
    ]
    if run.split:
        bits.append(f"split: <b>{run.split}</b>")
    if data.get("dataset_sha256"):
        bits.append(f"dataset: <code>{str(data['dataset_sha256'])[:12]}…</code>")
    st.markdown('<div class="metadata">' + " &nbsp; · &nbsp; ".join(bits) + "</div>", unsafe_allow_html=True)


def render_quality_cards(summary: dict[str, Any]) -> None:
    classification = summary["derived"].get("classification") or {}
    macro = classification.get("macro", {})
    weighted = classification.get("weighted", {})
    cards = [
        ("Macro F1", pct(macro.get("f1")), "Equal weight per class"),
        ("Balanced accuracy", pct(macro.get("recall")), "Macro recall"),
        ("Weighted F1", pct(weighted.get("f1")), "Support-weighted"),
        ("Micro accuracy", pct(classification.get("accuracy")), "Aggregate examples"),
    ]
    if summary["run"].kind == "deduction":
        cards += [
            ("Sound deductions", pct(metric(summary, "sound_deduction")), "Independent validator"),
            ("Parse rate", pct(metric(summary, "parse_rate")), "Structured output"),
        ]
    elif summary["run"].kind == "representation":
        cards += [
            ("Exact answers", pct(metric(summary, "exact_answer")), "Free generation"),
            ("Parse rate", pct(metric(summary, "parse_rate")), "Structured output"),
        ]
    for columns, row in ((6, cards),):
        cols = st.columns(columns)
        for col, (label, value, help_text) in zip(cols, row):
            col.metric(label, value, help=help_text)


def classification_frame(classification: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for label, item in classification.get("per_class", {}).items():
        rows.append(
            {
                "class": label,
                "support": item.get("support"),
                "precision": item.get("precision"),
                "recall": item.get("recall"),
                "F1": item.get("f1"),
                "specificity": item.get("specificity"),
                "balanced accuracy": item.get("balanced_accuracy"),
            }
        )
    return pd.DataFrame(rows)


def render_class_chart(classification: dict[str, Any], title: str = "Per-class performance") -> None:
    frame = classification_frame(classification)
    if frame.empty:
        st.info("No class-level predictions are available for this artifact.")
        return
    figure = go.Figure()
    for field, color in (("precision", "#ee714b"), ("recall", "#1d9b92"), ("F1", "#6559d6")):
        figure.add_trace(
            go.Bar(
                name=field.title(),
                x=frame["class"],
                y=frame[field],
                marker_color=color,
                hovertemplate="%{x}<br>" + field + ": %{y:.1%}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        barmode="group",
        yaxis={"tickformat": ".0%", "range": [0, 1]},
        xaxis={"title": None, "tickangle": -45},
        height=430,
        margin={"l": 10, "r": 10, "t": 50, "b": 110},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_overview(summary: dict[str, Any]) -> None:
    render_metadata(summary)
    if summary["run"].kind == "training":
        render_training_summary(summary)
        return
    render_quality_cards(summary)
    classification = summary["derived"].get("classification") or {}
    if not classification:
        st.info("This artifact has aggregate metrics but no prediction records to derive class metrics from.")
        render_precomputed(summary)
        return
    st.markdown('<div class="section-label">Class balance is part of the result</div>', unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        render_class_chart(classification)
    with right:
        support_frame = pd.DataFrame(
            [
                {"class": label, "support": support}
                for label, support in classification.get("support", {}).items()
            ]
        ).set_index("class")
        st.bar_chart(support_frame, color="#6559d6", height=290)
        imbalance = classification.get("imbalance_ratio")
        baseline = classification.get("majority_baseline")
        st.markdown(
            f'<div class="note">Support ratio: <b>{imbalance:.1f}×</b> between largest and smallest class. '
            f'Majority baseline: <b>{pct(baseline)}</b>. Macro scores prevent this baseline from looking like capability.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-label">Per-class table</div>', unsafe_allow_html=True)
    frame = classification_frame(classification)
    st.dataframe(
        frame.style.format(
            {field: "{:.1%}" for field in ("precision", "recall", "F1", "specificity", "balanced accuracy")}
        ),
        hide_index=True,
        use_container_width=True,
    )
    if summary["run"].kind == "representation":
        render_representation_diagnostics(summary)


def render_precomputed(summary: dict[str, Any]) -> None:
    overall = summary.get("overall", {})
    if overall:
        st.json(overall, expanded=False)


def render_training_summary(summary: dict[str, Any]) -> None:
    training = summary.get("training", {})
    validation = summary.get("validation", {})
    cols = st.columns(6)
    values = [
        ("Train loss", training.get("train_loss")),
        ("Validation loss", validation.get("eval_loss")),
        ("Token accuracy", validation.get("eval_mean_token_accuracy")),
        ("Runtime", seconds(summary["data"].get("runtime_seconds"))),
        ("Throughput", training.get("train_samples_per_second")),
        ("Peak VRAM", f"{summary['data'].get('peak_vram_gib', 0):.1f} GiB" if summary["data"].get("peak_vram_gib") else "n/a"),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, pct(value) if label == "Token accuracy" else (f"{value:.4f}" if isinstance(value, float) else value or "n/a"))
    st.markdown('<div class="section-label">Run configuration</div>', unsafe_allow_html=True)
    st.json(summary["data"].get("configuration", {}), expanded=False)


def render_representation_diagnostics(summary: dict[str, Any]) -> None:
    lists = summary["derived"].get("lists", {})
    if not lists:
        return
    st.markdown('<div class="section-label">List-level diagnostics</div>', unsafe_allow_html=True)
    rows = []
    for task, data in lists.items():
        rows.append(
            {
                "task": task,
                "empty cases": data.get("empty_cases"),
                "non-empty cases": data.get("nonempty_cases"),
                "balanced exact": data.get("balanced_exact_rate"),
                "macro precision": data.get("macro", {}).get("precision"),
                "macro recall": data.get("macro", {}).get("recall"),
                "macro F1": data.get("macro", {}).get("f1"),
                "micro precision": data.get("micro", {}).get("precision"),
                "micro recall": data.get("micro", {}).get("recall"),
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame.style.format(
            {column: "{:.1%}" for column in frame.columns if column not in {"task", "empty cases", "non-empty cases"}}
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_compare(summaries: list[dict[str, Any]]) -> None:
    choices = {summary["run"].run_id: display_name(summary) for summary in summaries}
    selected_ids = st.multiselect(
        "Runs to compare",
        list(choices),
        default=list(choices)[: min(3, len(choices))],
        format_func=choices.get,
    )
    metric_names = {
        "Macro F1": "macro_f1",
        "Balanced accuracy / macro recall": "macro_recall",
        "Weighted F1": "weighted_f1",
        "Micro accuracy": "accuracy",
        "Parse rate": "parse_rate",
        "Sound deduction rate": "sound_deduction",
        "Valid deduction rate": "valid_deduction",
        "Valid rule rate": "valid_rule",
        "Exact answer rate": "exact_answer",
    }
    selected_metric = st.selectbox("Comparison metric", list(metric_names))
    selected = [summary for summary in summaries if summary["run"].run_id in selected_ids]
    rows = []
    for summary in selected:
        value = metric(summary, metric_names[selected_metric])
        rows.append({"run": display_name(summary), "value": value, "kind": summary["run"].kind})
    if not rows:
        st.info("Select at least one run.")
        return
    frame = pd.DataFrame(rows)
    figure = go.Figure(
        go.Bar(
            x=frame["run"],
            y=frame["value"],
            marker_color=["#ee714b" if value is not None and value < .5 else "#1d9b92" for value in frame["value"]],
            text=[pct(value) for value in frame["value"]],
            textposition="outside",
        )
    )
    figure.update_layout(
        title=selected_metric,
        yaxis={"tickformat": ".0%", "range": [0, 1]},
        height=420,
        margin={"l": 10, "r": 10, "t": 55, "b": 120},
        xaxis={"tickangle": -35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)
    frame["value"] = frame["value"].map(pct)
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.markdown('<div class="note">Compare macro metrics before accuracy. A run can improve aggregate accuracy while getting worse on minority task families.</div>', unsafe_allow_html=True)


def render_confusion(classification: dict[str, Any]) -> None:
    labels = classification.get("matrix_labels", [])
    matrix = classification.get("confusion_matrix", {})
    if not labels or not matrix:
        st.info("No confusion matrix is available.")
        return
    normalize = st.checkbox("Normalize each target row", value=True)
    values = []
    for target in labels:
        row = matrix.get(target, {})
        total = sum(row.values())
        values.append(
            [((row.get(predicted, 0) / total) if normalize and total else row.get(predicted, 0)) for predicted in labels]
        )
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            colorscale=[[0, "#edf3f5"], [.5, "#7c8ce1"], [1, "#6559d6"]],
            text=[[f"{value:.1%}" if normalize else str(int(value)) for value in row] for row in values],
            texttemplate="%{text}",
            hovertemplate="target %{y}<br>prediction %{x}<br>%{z}<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(440, len(labels) * 28),
        margin={"l": 10, "r": 10, "t": 20, "b": 120},
        xaxis={"tickangle": -45, "title": "Predicted class"},
        yaxis={"title": "Target class"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)


def render_imbalance(summary: dict[str, Any]) -> None:
    classification = summary["derived"].get("classification") or {}
    if not classification:
        st.info("This run does not expose class-level predictions.")
        return
    render_metadata(summary)
    macro = classification.get("macro", {})
    weighted = classification.get("weighted", {})
    cols = st.columns(4)
    for col, label, value in (
        (cols[0], "Macro precision", macro.get("precision")),
        (cols[1], "Macro recall", macro.get("recall")),
        (cols[2], "Macro F1", macro.get("f1")),
        (cols[3], "Weighted F1", weighted.get("f1")),
    ):
        col.metric(label, pct(value))
    st.markdown('<div class="section-label">Where predictions land</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.5])
    with left:
        frame = classification_frame(classification)
        st.dataframe(frame.style.format({"precision": "{:.1%}", "recall": "{:.1%}", "F1": "{:.1%}"}), hide_index=True, use_container_width=True)
    with right:
        render_confusion(classification)


def render_detail(summary: dict[str, Any]) -> None:
    render_metadata(summary)
    st.markdown('<div class="section-label">Artifact metadata</div>', unsafe_allow_html=True)
    data = summary["data"]
    metadata = {
        "metrics file": str(summary["run"].path),
        "predictions file": str(summary["run"].predictions_path or "not found"),
        "model revision": data.get("model_revision"),
        "adapter": data.get("adapter"),
        "dataset": data.get("dataset"),
        "dataset hash": data.get("dataset_sha256"),
        "examples": data.get("examples"),
        "elapsed seconds": data.get("elapsed_seconds"),
    }
    st.json(metadata, expanded=False)
    records = summary["derived"].get("prediction_records", [])
    if not records:
        st.info("No prediction-level artifact was found for this run.")
        return
    st.markdown('<div class="section-label">Prediction browser</div>', unsafe_allow_html=True)
    targets = sorted({str(row.get("target_class", "unknown")) for row in records})
    target = st.selectbox("Target class", ["All"] + targets)
    status = st.selectbox("Outcome", ["All", "unparsed", "sound", "valid", "incorrect"])
    filtered = records
    if target != "All":
        filtered = [row for row in filtered if str(row.get("target_class")) == target]
    if status == "unparsed":
        filtered = [row for row in filtered if not row.get("parseable")]
    elif status == "sound":
        filtered = [row for row in filtered if row.get("sound_deduction")]
    elif status == "valid":
        filtered = [row for row in filtered if row.get("valid_deduction") or row.get("exact_answer")]
    elif status == "incorrect":
        filtered = [row for row in filtered if not (row.get("valid_deduction") or row.get("exact_answer") or row.get("sound_deduction"))]
    rows = []
    for row in filtered:
        rows.append(
            {
                "id": row.get("id"),
                "target": row.get("target_class"),
                "predicted": row.get("predicted_class"),
                "parseable": row.get("parseable"),
                "sound": row.get("sound_deduction", row.get("schema_valid")),
                "valid": row.get("valid_deduction", row.get("exact_answer")),
                "seconds": row.get("generation_seconds"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=360)
    if filtered:
        selected_id = st.selectbox("Inspect raw record", [str(row.get("id")) for row in filtered])
        selected = next(row for row in filtered if str(row.get("id")) == selected_id)
        st.json(selected, expanded=False)


def main() -> None:
    inject_style()
    with st.sidebar:
        st.markdown("### ◈ Evaluation lab")
        artifact_root = st.text_input("Artifact directory", "artifacts")
        if st.button("Rescan artifacts", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        view = st.radio("View", ["Overview", "Compare runs", "Imbalance lens", "Run detail"])

    root = Path(artifact_root)
    summaries = cached_summaries(str(root), artifact_fingerprint(root)) if root.exists() else []
    render_header()
    if not summaries:
        st.warning(f"No metric artifacts found under `{root}`.")
        return

    labels = {summary["run"].run_id: display_name(summary) for summary in summaries}
    with st.sidebar:
        selected_id = st.selectbox("Focused run", list(labels), format_func=labels.get)
    focused = next(summary for summary in summaries if summary["run"].run_id == selected_id)

    if view == "Overview":
        render_overview(focused)
    elif view == "Compare runs":
        render_compare(summaries)
    elif view == "Imbalance lens":
        render_imbalance(focused)
    else:
        render_detail(focused)


if __name__ == "__main__":
    main()
