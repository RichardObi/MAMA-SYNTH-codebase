#  Copyright 2025 mama-sia-eval contributors
#  Licensed under the Apache License, Version 2.0

"""
Streamlit-based web interface for the MAMA-SYNTH evaluation pipeline.

Launch with::

    streamlit run src/mama_sia_eval/webapp.py

Or from the package::

    python -m mama_sia_eval.webapp

The interface allows participants to:
  1. Upload ground-truth and prediction directories (or select paths).
  2. Configure evaluation options (masks, labels, metric toggles).
  3. Run the full evaluation pipeline.
  4. View interactive results: tables, charts, segmentation overlays.
  5. Download the results JSON and report files.

Requirements:
    pip install streamlit plotly
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["main"]


def _check_streamlit() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "The web interface requires Streamlit.\n"
            "Install it with:  pip install streamlit plotly\n"
            "Then run:  streamlit run src/mama_sia_eval/webapp.py",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """Entry point for the Streamlit web application."""
    _check_streamlit()
    import streamlit as st
    import numpy as np

    st.set_page_config(
        page_title="MAMA-SYNTH Evaluation",
        page_icon="\U0001f9ea",
        layout="wide",
    )

    st.title("\U0001f9ea MAMA-SYNTH Evaluation Dashboard")
    st.markdown(
        "Evaluate pre- to post-contrast breast DCE-MRI synthesis "
        "across four metric groups: **CLF**, **SEG**, **ROI**, **FULL**."
    )

    # ------------------------------------------------------------------
    # Sidebar: Configuration
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("\u2699\ufe0f Configuration")

        mode = st.radio(
            "Input mode",
            ["Local paths", "Upload results JSON"],
            help="Choose to run the pipeline or visualise existing results.",
        )

        if mode == "Local paths":
            gt_path = st.text_input(
                "Ground-truth directory",
                value="",
                placeholder="/path/to/ground-truth",
            )
            pred_path = st.text_input(
                "Predictions directory",
                value="",
                placeholder="/path/to/predictions",
            )
            masks_path = st.text_input(
                "Masks directory (optional)",
                value="",
                placeholder="/path/to/masks",
            )
            labels_path = st.text_input(
                "Labels file (optional)",
                value="",
                placeholder="/path/to/labels.json or labels.csv",
            )

            st.subheader("Metric toggles")
            enable_lpips = st.checkbox("LPIPS", value=False)
            enable_frd = st.checkbox("FRD", value=False)
            enable_seg = st.checkbox("Segmentation", value=True)
            enable_clf = st.checkbox("Classification", value=False)
            roi_margin = st.slider("ROI margin (mm)", 0.0, 30.0, 10.0, 1.0)

            run_btn = st.button("\u25b6\ufe0f Run Evaluation", type="primary")
        else:
            run_btn = False

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------

    if mode == "Upload results JSON":
        uploaded = st.file_uploader("Upload metrics.json", type=["json"])
        if uploaded is not None:
            results = json.load(uploaded)
            _render_results(st, results)
        return

    if not run_btn:
        st.info(
            "Configure paths in the sidebar and click **Run Evaluation** to start."
        )
        return

    # Validate inputs
    if not gt_path or not pred_path:
        st.error("Ground-truth and predictions paths are required.")
        return

    gt = Path(gt_path)
    pred = Path(pred_path)
    if not gt.exists():
        st.error(f"Ground-truth path not found: {gt}")
        return
    if not pred.exists():
        st.error(f"Predictions path not found: {pred}")
        return

    # Run evaluation
    with st.spinner("Running evaluation pipeline\u2026"):
        try:
            from mama_sia_eval.evaluation import MamaSiaEval

            with tempfile.TemporaryDirectory() as tmpdir:
                out_file = Path(tmpdir) / "metrics.json"
                evaluator = MamaSiaEval(
                    ground_truth_path=gt,
                    predictions_path=pred,
                    output_file=out_file,
                    masks_path=Path(masks_path) if masks_path else None,
                    labels_path=Path(labels_path) if labels_path else None,
                    roi_margin_mm=roi_margin,
                    enable_lpips=enable_lpips,
                    enable_frd=enable_frd,
                    enable_segmentation=enable_seg,
                    enable_classification=enable_clf,
                )
                results = evaluator.evaluate()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            return

    st.success("Evaluation complete!")
    _render_results(st, results)


def _render_results(st: Any, results: dict[str, Any]) -> None:
    """Render the results dashboard."""
    import numpy as np

    # Download button
    json_str = json.dumps(results, indent=2, default=str)
    st.download_button(
        "\u2b07\ufe0f Download metrics.json",
        data=json_str,
        file_name="metrics.json",
        mime="application/json",
    )

    # Summary metrics
    st.header("\U0001f4ca Summary")
    _render_aggregate(st, results)

    # Group metrics
    tabs = st.tabs(["Full Image", "Tumor ROI", "Segmentation", "Classification", "Per Case"])

    with tabs[0]:
        _render_group(st, results.get("full_image", {}), "Full-Image Metrics")

    with tabs[1]:
        _render_group(st, results.get("roi", {}), "Tumor ROI Metrics")

    with tabs[2]:
        _render_group(st, results.get("segmentation", {}), "Segmentation Metrics")

    with tabs[3]:
        _render_classification(st, results.get("classification", {}))

    with tabs[4]:
        _render_per_case(st, results.get("cases", []))

    # Missing predictions warning
    missing = results.get("missing_predictions", [])
    if missing:
        st.warning(
            f"**{len(missing)}** ground truth images had no matching predictions. "
            "Worst-score imputation was applied."
        )
        with st.expander("Missing cases"):
            st.write(missing)


def _render_aggregate(st: Any, results: dict[str, Any]) -> None:
    """Render top-level aggregate metrics as KPI cards."""
    agg = results.get("aggregate", {})
    cols = st.columns(min(len(agg), 6))
    for i, (metric, stats) in enumerate(sorted(agg.items())):
        if isinstance(stats, dict) and "mean" in stats:
            col = cols[i % len(cols)]
            col.metric(
                label=metric.upper(),
                value=f"{stats['mean']:.4f}",
                delta=f"\u00b1{stats['std']:.4f}" if "std" in stats else None,
            )


def _render_group(st: Any, group: dict[str, Any], title: str) -> None:
    """Render a metric group as a bar chart + table."""
    if not group:
        st.info("No data available for this group.")
        return

    st.subheader(title)

    # Try plotly interactive chart
    try:
        import plotly.graph_objects as go

        names, means, stds = [], [], []
        for metric, stats in sorted(group.items()):
            if isinstance(stats, dict) and "mean" in stats:
                names.append(metric.upper())
                means.append(stats["mean"])
                stds.append(stats.get("std", 0))
            elif isinstance(stats, (int, float)):
                names.append(metric.upper())
                means.append(stats)
                stds.append(0)

        if names:
            fig = go.Figure(
                go.Bar(
                    x=names, y=means,
                    error_y=dict(type="data", array=stds),
                    marker_color="#4C72B0",
                )
            )
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.write("Install `plotly` for interactive charts.")

    # Table
    import pandas as pd

    rows = []
    for metric, stats in sorted(group.items()):
        if isinstance(stats, dict):
            rows.append({"Metric": metric.upper(), **stats})
        else:
            rows.append({"Metric": metric.upper(), "value": stats})

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_classification(st: Any, clf: dict[str, Any]) -> None:
    """Render classification results."""
    if not clf:
        st.info("Classification not run (provide labels and enable CLF).")
        return

    st.subheader("Classification Results")
    for key, val in sorted(clf.items()):
        if isinstance(val, (int, float)):
            st.metric(key, f"{val:.4f}")
        elif isinstance(val, str):
            st.info(val)


def _render_per_case(st: Any, cases: list[dict[str, Any]]) -> None:
    """Render per-case metrics table."""
    if not cases:
        st.info("No per-case data available.")
        return

    import pandas as pd

    st.subheader("Per-Case Results")
    df = pd.DataFrame(cases)
    if "_imputed" in df.columns:
        df["imputed"] = df["_imputed"].fillna(False)
        df = df.drop(columns=["_imputed"])

    # Highlight best values: higher-is-better metrics → max, others → min
    _HIGHER_IS_BETTER = {"psnr", "ssim", "ncc", "dice"}
    numeric_cols = [c for c in df.columns if c not in ("case_id", "imputed")]
    higher_cols = [c for c in numeric_cols if c in _HIGHER_IS_BETTER]
    lower_cols = [c for c in numeric_cols if c not in _HIGHER_IS_BETTER]

    styled = df.style
    if higher_cols:
        styled = styled.highlight_max(subset=higher_cols, color="#d4edda")
    if lower_cols:
        styled = styled.highlight_min(subset=lower_cols, color="#d4edda")

    st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()
