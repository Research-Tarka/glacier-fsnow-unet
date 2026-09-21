"""Standalone HTML reports for the interpretation analyses.

Every figure is written as a self-contained HTML file with the plotting library
inlined, so a report can be opened from a filesystem, emailed, or attached to a
review with no server, no network fetch and no build step. The cost is roughly
3 MB of inlined JavaScript per file, which is the right trade for an artifact
that has to still render years after the run that produced it.

These files are a *rendering*, not a deliverable dataset. Every number they
show also exists in the JSON summary written alongside them, and any per-pixel
map they display is persisted in zarr -- so nothing here is the only copy of
anything, and a reader who needs the values parses the JSON rather than the
HTML.

Plotly rather than matplotlib for this specific job: these figures are read
interactively (hovering a band to read its exact attribution, zooming a
threshold sweep around the chosen point), which a static raster cannot do, and
its `write_html(include_plotlyjs="inline")` produces the single self-contained
file described above in one call. The training module's own exports stay
tabular CSV/JSON precisely because they are meant to be re-read by code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

__all__ = [
    "PlotlyUnavailableError",
    "write_ambiguity_report",
    "write_shap_global_report",
    "write_shap_per_class_report",
    "write_grad_cam_report",
]


class PlotlyUnavailableError(RuntimeError):
    """Raised when plotly is not installed but an HTML report was requested."""


_INSTALL_HINT = (
    "plotly is required for the interpretation HTML reports.\n"
    "Install it with: pip install -r requirements-torch.txt"
)


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PlotlyUnavailableError(_INSTALL_HINT) from exc
    return go


def _write(figure: Any, destination: str | Path, title: str) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.update_layout(title=title, template="plotly_white")
    figure.write_html(str(path), include_plotlyjs="inline", full_html=True)
    return path


def write_ambiguity_report(
    sweep: Sequence[dict[str, float]],
    chosen_threshold: float,
    tolerance: float,
    baseline_miou: float,
    destination: str | Path,
) -> Path:
    """Plot macro-mIoU against the ambiguity threshold, marking the choice.

    The tolerance band is drawn explicitly, because the chosen point is not the
    curve's maximum -- it is the rightmost point still inside that band, and a
    reader who does not see the band will read the choice as a mistake.
    """
    go = _plotly()

    thresholds = [float(record["threshold"]) for record in sweep]
    mious = [float(record["macro_miou"]) for record in sweep]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=thresholds,
            y=mious,
            mode="lines",
            name="macro mIoU with the priority rule",
            hovertemplate="threshold %{x:.4f}<br>mIoU %{y:.5f}<extra></extra>",
        )
    )
    figure.add_hline(
        y=baseline_miou,
        line_dash="dash",
        annotation_text=f"plain argmax baseline ({baseline_miou:.5f})",
    )
    figure.add_hrect(
        y0=baseline_miou - abs(tolerance),
        y1=max(mious) if mious else baseline_miou,
        fillcolor="green",
        opacity=0.08,
        line_width=0,
        annotation_text=f"within tolerance ({tolerance:g})",
    )
    figure.add_vline(
        x=chosen_threshold,
        line_dash="dot",
        annotation_text=f"chosen: {chosen_threshold:.4f}",
    )
    figure.update_layout(
        xaxis_title="confidence gap threshold  P(top1) - P(top2)",
        yaxis_title="macro mIoU",
    )
    return _write(figure, destination, "Ambiguity threshold calibration")


def write_shap_global_report(
    feature_names: Sequence[str],
    macro_values: Sequence[float],
    destination: str | Path,
    method: str = "expected_gradients",
) -> Path:
    """Rank the input bands by mean absolute attribution across all classes."""
    go = _plotly()

    order = np.argsort(np.asarray(macro_values, dtype=float))
    names = [str(feature_names[i]) for i in order]
    values = [float(macro_values[i]) for i in order]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            hovertemplate="%{y}<br>%{x:.6g}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title=f"mean |attribution| ({method})",
        yaxis_title="input band",
        height=max(360, 28 * len(names) + 160),
    )
    return _write(figure, destination, "Band attribution, all classes")


def write_shap_per_class_report(
    feature_names: Sequence[str],
    class_names: Sequence[str],
    values: np.ndarray,
    destination: str | Path,
    method: str = "expected_gradients",
) -> Path:
    """Heatmap the `(band, class)` attribution table.

    A band can be indispensable for one class and irrelevant to the other
    three; a single ranked bar chart averages exactly that structure away,
    which is why this figure exists alongside the global one.
    """
    go = _plotly()

    matrix = np.asarray(values, dtype=float)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[str(name) for name in class_names],
            y=[str(name) for name in feature_names],
            colorscale="RdBu",
            zmid=0.0,
            colorbar={"title": "attribution"},
            hovertemplate="band %{y}<br>class %{x}<br>%{z:.6g}<extra></extra>",
        )
    )
    figure.update_layout(
        xaxis_title="class",
        yaxis_title="input band",
        height=max(400, 30 * matrix.shape[0] + 180),
    )
    return _write(
        figure, destination, f"Band attribution per class ({method})"
    )


def write_grad_cam_report(
    class_names: Sequence[str],
    mean_maps: Sequence[np.ndarray],
    tiles_per_class: Sequence[int],
    destination: str | Path,
    target_layer: str = "enc4",
) -> Path:
    """One mean localisation heatmap per class, side by side.

    Averaging over tiles rather than showing single examples is deliberate:
    one tile's map is as much about that tile's terrain as about the model, and
    a reviewer picking the most striking example is how an interpretation
    report becomes an illustration of a conclusion already reached.
    """
    go = _plotly()  # raises the install hint before the subplots import does
    from plotly.subplots import make_subplots

    names = [str(name) for name in class_names]
    figure = make_subplots(
        rows=1,
        cols=max(1, len(names)),
        subplot_titles=[
            f"{name} ({count} tiles)" for name, count in zip(names, tiles_per_class)
        ],
    )
    for index, array in enumerate(mean_maps):
        figure.add_trace(
            go.Heatmap(
                z=np.asarray(array, dtype=float),
                colorscale="Inferno",
                zmin=0.0,
                zmax=1.0,
                showscale=index == len(names) - 1,
                hovertemplate="row %{y}<br>col %{x}<br>%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=index + 1,
        )
    for index in range(len(names)):
        figure.update_yaxes(autorange="reversed", row=1, col=index + 1)
    figure.update_layout(height=420, width=max(520, 320 * len(names)))
    return _write(
        figure,
        destination,
        f"Grad-CAM mean localisation per class (layer {target_layer})",
    )
