#  Copyright 2025 mama-synth-eval contributors
#  Licensed under the Apache License, Version 2.0

"""
Result visualization for the MAMA-SYNTH evaluation pipeline.

Generates:
  - Summary tables (text and HTML)
  - Metric bar charts (per-case and aggregate)
  - Radar / spider plots comparing metric groups
  - Segmentation overlay images
  - Ranking leaderboard

All plotting uses ``matplotlib`` (required) and optionally ``plotly``
for interactive HTML output.

Usage::

    from eval.visualization import ResultVisualizer
    viz = ResultVisualizer(results_json="metrics.json", output_dir="reports")
    viz.generate_all()
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ResultVisualizer"]


class ResultVisualizer:
    """Generate visual reports from MAMA-SYNTH evaluation results.

    Args:
        results: Either a dict (from ``MamaSynthEval.evaluate()``) or a
                 path to the JSON file produced by the CLI.
        output_dir: Directory where report artefacts are saved.
    """

    def __init__(
        self,
        results: dict[str, Any] | str | Path,
        output_dir: str | Path = "reports",
    ) -> None:
        if isinstance(results, (str, Path)):
            with open(results) as f:
                self._results: dict[str, Any] = json.load(f)
        else:
            self._results = results

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> list[Path]:
        """Generate every available visualisation.

        Returns:
            List of file paths created.
        """
        created: list[Path] = []
        created.extend(self.summary_table())
        created.extend(self.metric_bar_charts())
        created.extend(self.radar_plot())
        created.extend(self.per_case_table())
        logger.info(f"Generated {len(created)} report artefacts in {self.output_dir}")
        return created

    # ------------------------------------------------------------------
    # 1. Summary table
    # ------------------------------------------------------------------

    def summary_table(self) -> list[Path]:
        """Produce a plain-text and HTML summary table of aggregate metrics."""
        rows: list[tuple[str, str, str, str]] = []

        for section, label in [
            ("aggregate", "Per-Case"),
            ("full_image", "Full Image"),
            ("roi", "Tumor ROI"),
            ("segmentation", "Segmentation"),
        ]:
            block = self._results.get(section, {})
            for metric, stats in sorted(block.items()):
                if isinstance(stats, dict) and "mean" in stats:
                    rows.append((
                        label,
                        metric.upper(),
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                    ))
                elif isinstance(stats, (int, float)):
                    rows.append((label, metric.upper(), f"{stats:.4f}", "-"))

        clf = self._results.get("classification", {})
        for k, v in sorted(clf.items()):
            if isinstance(v, (int, float)):
                rows.append(("Classification", k, f"{v:.4f}", "-"))

        # Plain text
        txt_path = self.output_dir / "summary.txt"
        lines = [f"{'Group':<18} {'Metric':<12} {'Mean':>10} {'Std':>10}"]
        lines.append("-" * 52)
        for grp, met, mean, std in rows:
            lines.append(f"{grp:<18} {met:<12} {mean:>10} {std:>10}")
        txt_path.write_text("\n".join(lines) + "\n")

        # HTML
        html_path = self.output_dir / "summary.html"
        html_rows = "".join(
            f"<tr><td>{g}</td><td>{m}</td><td>{mn}</td><td>{s}</td></tr>"
            for g, m, mn, s in rows
        )
        html = (
            "<html><head><style>"
            "table{border-collapse:collapse;font-family:monospace;}"
            "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
            "th{background:#f0f0f0}"
            "</style></head><body>"
            "<h2>MAMA-SYNTH Evaluation Summary</h2>"
            "<table>"
            "<tr><th>Group</th><th>Metric</th><th>Mean</th><th>Std</th></tr>"
            f"{html_rows}"
            "</table></body></html>"
        )
        html_path.write_text(html)

        return [txt_path, html_path]

    # ------------------------------------------------------------------
    # 2. Metric bar charts
    # ------------------------------------------------------------------

    def metric_bar_charts(self) -> list[Path]:
        """Create bar chart PNGs for each metric group."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed — skipping bar charts.")
            return []

        created: list[Path] = []

        for section, title in [
            ("aggregate", "Per-Case Aggregate Metrics"),
            ("full_image", "Full-Image Metrics"),
            ("roi", "Tumor ROI Metrics"),
            ("segmentation", "Segmentation Metrics"),
        ]:
            block = self._results.get(section, {})
            names, means, stds = [], [], []
            for metric, stats in sorted(block.items()):
                if isinstance(stats, dict) and "mean" in stats:
                    names.append(metric.upper())
                    means.append(stats["mean"])
                    stds.append(stats.get("std", 0))

            if not names:
                continue

            fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 4))
            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_title(title)
            ax.set_ylabel("Value")
            fig.tight_layout()

            path = self.output_dir / f"bar_{section}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            created.append(path)

        return created

    # ------------------------------------------------------------------
    # 3. Radar / spider plot
    # ------------------------------------------------------------------

    def radar_plot(self) -> list[Path]:
        """Create a radar plot of group-level scores (normalised 0–1)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return []

        # Collect one representative number per group.
        # For each metric, record whether higher is better so the radar
        # plot consistently maps "better" to a larger radius.
        _LOWER_IS_BETTER = {"mse", "mae", "nmse", "hd95", "lpips", "frd"}

        group_scores: dict[str, float] = {}
        lower_flags: dict[str, bool] = {}
        for section, key in [
            ("full_image", "mse"),
            ("roi", "mse"),
            ("segmentation", "dice"),
        ]:
            block = self._results.get(section, {})
            stats = block.get(key)
            if isinstance(stats, dict) and "mean" in stats:
                label = f"{section}_{key}"
                group_scores[label] = stats["mean"]
                lower_flags[label] = key in _LOWER_IS_BETTER

        clf = self._results.get("classification", {})
        for k, v in clf.items():
            if "auroc" in k and isinstance(v, (int, float)):
                group_scores[k] = v
                lower_flags[k] = False  # AUROC higher is better

        if len(group_scores) < 3:
            return []

        labels = list(group_scores.keys())
        values = list(group_scores.values())
        # Normalise to 0-1, inverting "lower is better" metrics so
        # larger radius always means better performance.
        vmin, vmax = min(values), max(values)
        if vmax > vmin:
            normed = [(v - vmin) / (vmax - vmin) for v in values]
        else:
            normed = [1.0] * len(values)
        for i, lbl in enumerate(labels):
            if lower_flags.get(lbl, False):
                normed[i] = 1.0 - normed[i]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        normed += normed[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles, normed, "o-", linewidth=2, color="#4C72B0")
        ax.fill(angles, normed, alpha=0.25, color="#4C72B0")
        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            [l.replace("_", "\n") for l in labels],
        )
        ax.set_title("Metric Group Radar", y=1.08)
        fig.tight_layout()

        path = self.output_dir / "radar.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return [path]

    # ------------------------------------------------------------------
    # 4. Per-case table
    # ------------------------------------------------------------------

    def per_case_table(self) -> list[Path]:
        """Write a CSV and HTML table with per-case metrics."""
        cases = self._results.get("cases", [])
        if not cases:
            return []

        import csv as csv_mod

        keys = [k for k in cases[0].keys() if k != "_imputed"]

        csv_path = self.output_dir / "per_case.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(
                f, fieldnames=keys, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(cases)

        # HTML
        header = "".join(f"<th>{k}</th>" for k in keys)
        body = ""
        for c in cases:
            cls = ' class="imputed"' if c.get("_imputed") else ""
            cells = ""
            for k in keys:
                v = c.get(k, "")
                cells += f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
            body += f"<tr{cls}>{cells}</tr>"

        html_path = self.output_dir / "per_case.html"
        html = (
            "<html><head><style>"
            "table{border-collapse:collapse;font-family:monospace;font-size:12px}"
            "th,td{border:1px solid #ccc;padding:3px 6px;text-align:right}"
            "th{background:#f0f0f0}"
            ".imputed{background:#ffe0e0}"
            "</style></head><body>"
            "<h2>Per-Case Metrics</h2>"
            f"<table><tr>{header}</tr>{body}</table>"
            "</body></html>"
        )
        html_path.write_text(html)

        return [csv_path, html_path]

    # ------------------------------------------------------------------
    # 5. Segmentation overlay (static)
    # ------------------------------------------------------------------

    @staticmethod
    def segmentation_overlay(
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        output_path: str | Path,
        title: str = "Segmentation Overlay",
    ) -> Path:
        """Save a side-by-side image with GT and predicted mask overlay.

        Args:
            image: 2D image array.
            gt_mask: Ground truth binary mask.
            pred_mask: Predicted binary mask.
            output_path: Where to save the PNG.
            title: Figure title.

        Returns:
            Path to the saved image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(image, cmap="gray")
        axes[1].contour(gt_mask, colors=["lime"], linewidths=1.5)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(image, cmap="gray")
        axes[2].contour(pred_mask, colors=["red"], linewidths=1.5)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    # ------------------------------------------------------------------
    # Interactive HTML (Plotly)
    # ------------------------------------------------------------------

    def interactive_bar_chart(self) -> list[Path]:
        """Generate an interactive Plotly bar chart (standalone HTML)."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed — skipping interactive charts.")
            return []

        fig = go.Figure()

        for section, label, colour in [
            ("aggregate", "Per-Case", "#4C72B0"),
            ("full_image", "Full Image", "#DD8452"),
            ("segmentation", "Segmentation", "#55A868"),
        ]:
            block = self._results.get(section, {})
            for metric, stats in sorted(block.items()):
                if isinstance(stats, dict) and "mean" in stats:
                    fig.add_trace(go.Bar(
                        name=f"{label} {metric.upper()}",
                        x=[f"{label}<br>{metric.upper()}"],
                        y=[stats["mean"]],
                        error_y=dict(type="data", array=[stats.get("std", 0)]),
                        marker_color=colour,
                    ))

        fig.update_layout(
            title="MAMA-SYNTH Evaluation Results",
            barmode="group",
            template="plotly_white",
        )
        path = self.output_dir / "interactive_results.html"
        fig.write_html(str(path))
        return [path]
