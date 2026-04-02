"""Multi-sample QA group reporting for MEGqc derivatives.

This module builds one HTML report that compares two or more datasets while
reusing the same derivative-loading and run-level aggregation logic as the
single-dataset group report.

Design rules implemented here
-----------------------------
1) Cohort QA overview and QA metrics across tasks use one subtab per dataset.
2) QA metrics details merges datasets in shared distribution/fingerprint plots;
   heatmaps and topographic maps are shown in dataset-specific subtabs.
3) Cummulative distributions combines all datasets and tasks in the same ECDF
   plots.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import meg_qc
from meg_qc.calculation.meg_qc_pipeline import resolve_analysis_root
from meg_qc.plotting.meg_qc_group_plots import (
    ChTypeAccumulator,
    TopomapPayload,
    _build_accumulators_for_runs,
    _build_cohort_overview_section,
    _build_condition_effect_section,
    _build_subtabs_html,
    _combine_accumulators,
    _discover_run_records,
    _epoch_consistency_notes,
    _epoch_values_from_profiles,
    _figure_to_div,
    _figure_block,
    _finite_array,
    _heatmap_variants_by_condition_from_acc,
    _lazy_payload_script_tags_html,
    _load_settings_snapshot,
    _make_ecdf_figure,
    _plot_summary_distribution_recordings,
    _pooled_topomap_payload,
    _reset_lazy_figure_store,
    _run_rows_dataframe,
    _summary_shared_controls_panel_html,
    plot_density_distribution,
    plot_box_with_subject_jitter,
    plot_heatmap_sorted_channels_windows,
    plot_topomap_3d_if_available,
    plot_histogram_distribution,
    plot_topomap_if_available,
    plot_violin_with_subject_jitter,
)

MAX_POINTS_SCATTER = 4000


def _adaptive_legend_layout(n_entries: int) -> dict:
    """Return legend + margin + height kwargs for non-overlapping zones.

    Title is rendered as independent HTML above the figure (see
    ``_figure_to_div``).  The legend is **always horizontal**, placed above
    the plot area with ``y > 1`` so it lives in the top-margin band.  The
    top margin grows dynamically to accommodate multi-row wrapping, so
    legend and plot content never overlap.
    """
    entries_per_row = 3  # conservative for typical label widths
    n_rows = max(1, -(-n_entries // entries_per_row))
    legend_height_px = n_rows * 26 + 20
    margin_t = max(50, legend_height_px)
    return {
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
            "itemsizing": "constant",
            "tracegroupgap": 4,
        },
        "margin": {"l": 55, "r": 30, "t": margin_t, "b": 60},
        "height": max(620, margin_t + 60 + 400),
    }


@dataclass
class SampleBundle:
    """Container with all information needed for one dataset sample."""

    sample_id: str
    dataset_path: str
    derivatives_root: str
    reports_dir: Path
    settings_snapshot: str
    tab_accumulators: Dict[str, ChTypeAccumulator]


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token or "item"


def _safe_sample_id(dataset_path: str) -> str:
    return _sanitize_token(os.path.basename(os.path.normpath(dataset_path)))


def _strip_outer_section(html: str) -> str:
    txt = str(html)
    match = re.match(r"^\s*<section>(.*)</section>\s*$", txt, flags=re.S)
    if match:
        txt = match.group(1)
    txt = re.sub(r"^\s*<h2>.*?</h2>\s*", "", txt, count=1, flags=re.S)
    return txt


def _collect_sample_bundle(
    dataset_path: str,
    derivatives_base: Optional[str] = None,
    n_jobs: int = 1,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> Optional[SampleBundle]:
    """Load one dataset into MAG/GRAD/combined accumulators."""
    (
        _output_root,
        derivatives_root,
        megqc_root,
        _resolved_analysis_id,
        analysis_segments,
    ) = resolve_analysis_root(
        dataset_path=dataset_path,
        external_derivatives_root=derivatives_base,
        analysis_mode=analysis_mode,
        analysis_id=analysis_id,
        create_if_missing=True,
    )
    sample_id = _safe_sample_id(dataset_path)

    # Reports always go to the resolved (external) path.
    reports_dir = Path(megqc_root) / "reports"

    # Source: prefer external path, fall back to original dataset (Scenario C).
    source_megqc_root = megqc_root
    calculation_dir = Path(source_megqc_root) / "calculation"
    if not calculation_dir.exists() and derivatives_base is not None:
        original_megqc_root = os.path.join(
            dataset_path, "derivatives", "Meg_QC", *analysis_segments
        )
        original_calc_dir = Path(original_megqc_root) / "calculation"
        if original_calc_dir.exists():
            source_megqc_root = original_megqc_root
            calculation_dir = original_calc_dir
            print(
                f"___MEGqc___: Multi-sample QA: Scenario C — "
                f"reading calculation from original dataset: {calculation_dir}"
            )

    if not calculation_dir.exists():
        print(f"___MEGqc___: Multi-sample QA: skipping {sample_id}, calculation folder missing: {calculation_dir}")
        return None

    run_records = _discover_run_records(calculation_dir)
    if not run_records:
        print(f"___MEGqc___: Multi-sample QA: skipping {sample_id}, no run TSV derivatives found.")
        return None

    acc_by_type = _build_accumulators_for_runs(run_records, n_jobs=n_jobs)

    combined_acc = _combine_accumulators(acc_by_type)
    # Build tab accumulators, filtering out types with no data.
    tab_accumulators: Dict[str, ChTypeAccumulator] = {}
    if acc_by_type["mag"].run_count > 0 or acc_by_type["grad"].run_count > 0:
        tab_accumulators["Combined (mag+grad)"] = combined_acc
    if acc_by_type["mag"].run_count > 0:
        tab_accumulators["MAG"] = acc_by_type["mag"]
    if acc_by_type["grad"].run_count > 0:
        tab_accumulators["GRAD"] = acc_by_type["grad"]
    if acc_by_type["eeg"].run_count > 0:
        tab_accumulators["EEG"] = acc_by_type["eeg"]
    if all(acc.run_count == 0 for acc in tab_accumulators.values()):
        print(f"___MEGqc___: Multi-sample QA: skipping {sample_id}, no usable run summaries.")
        return None

    return SampleBundle(
        sample_id=sample_id,
        dataset_path=dataset_path,
        derivatives_root=derivatives_root,
        reports_dir=reports_dir,
        settings_snapshot=_load_settings_snapshot(source_megqc_root),
        tab_accumulators=tab_accumulators,
    )


def _tab_dataframe(bundles: Sequence[SampleBundle], tab_name: str) -> pd.DataFrame:
    """Stack run-level summaries for one channel-type tab across datasets."""
    frames: List[pd.DataFrame] = []
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None or not acc.run_rows:
            continue
        df = _run_rows_dataframe(acc.run_rows)
        if df.empty:
            continue
        df = df.copy()
        df["sample_id"] = bundle.sample_id
        df["subject_original"] = df["subject"].astype(str)
        df["subject"] = df["sample_id"].astype(str) + "::" + df["subject"].astype(str)
        df["run_key"] = (
            df["sample_id"].astype(str)
            + "::"
            + df["run_key"].astype(str)
            + "::"
            + df["channel_type"].astype(str)
        )
        if "task" not in df.columns:
            df["task"] = "n/a"
        if "condition_label" not in df.columns:
            df["condition_label"] = "all recordings"
        df["task_label"] = np.where(df["task"].astype(str) != "n/a", df["task"].astype(str), df["condition_label"].astype(str))
        if "hover_entities" in df.columns:
            df["hover_entities"] = "sample=" + df["sample_id"].astype(str) + "<br>" + df["hover_entities"].astype(str)
        else:
            df["hover_entities"] = (
                "sample="
                + df["sample_id"].astype(str)
                + "<br>subject="
                + df["subject_original"].astype(str)
                + "<br>task="
                + df["task"].astype(str)
                + "<br>run="
                + df["run"].astype(str)
            )
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _condition_symbol_map(conditions: Sequence[str]) -> Dict[str, str]:
    symbols = ["circle", "diamond", "square", "cross", "triangle-up", "triangle-down", "x", "star"]
    return {str(cond): symbols[idx % len(symbols)] for idx, cond in enumerate(sorted(set(str(c) for c in conditions)))}


def _sample_color_map(sample_ids: Sequence[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600",
    ]
    uniq = sorted(set(str(s) for s in sample_ids))
    return {sid: palette[idx % len(palette)] for idx, sid in enumerate(uniq)}


def plot_run_fingerprint_scatter_by_sample(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Optional[go.Figure]:
    """Run fingerprint with sample colors and task legend filtering."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    if data.empty:
        return None

    symbol_map = _condition_symbol_map(data["condition_label"].astype(str).tolist())
    sample_map = _sample_color_map(data["sample_id"].astype(str).tolist())

    fig = go.Figure()
    for cond in sorted(symbol_map):
        dcond = data.loc[data["condition_label"].astype(str) == cond].copy()
        if dcond.empty:
            continue
        keep = np.arange(len(dcond), dtype=int)
        if len(dcond) > MAX_POINTS_SCATTER:
            keep = np.linspace(0, len(dcond) - 1, num=MAX_POINTS_SCATTER, dtype=int)
            keep = np.unique(keep)
            dcond = dcond.iloc[keep].copy()
        fig.add_trace(
            go.Scattergl(
                x=dcond[x_col],
                y=dcond[y_col],
                mode="markers",
                name=cond,
                legendgroup=f"task-{cond}",
                marker={
                    "size": 8,
                    "symbol": symbol_map[cond],
                    "color": [sample_map.get(str(s), "#4F6F84") for s in dcond["sample_id"].astype(str)],
                    "line": {"width": 0.6, "color": "#2b2d42"},
                    "opacity": 0.86,
                    "showscale": False,
                },
                customdata=np.stack([dcond["hover_entities"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>task=" + cond + "<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
                showlegend=True,
            )
        )

    if not fig.data:
        return None
    legend_kw = _adaptive_legend_layout(len(fig.data))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        **legend_kw,
    )
    return fig


def _run_values_by_condition(df: pd.DataFrame, value_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    if df.empty or value_col not in df.columns:
        return {}
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    labels = df["condition_label"].astype(str).tolist() if "condition_label" in df.columns else ["all recordings"] * len(df)
    for label, val in zip(labels, vals):
        if np.isfinite(val):
            out[str(label)].append(float(val))
    return dict(out)


def _run_values_labeled(df: pd.DataFrame, value_col: str) -> Dict[str, List[float]]:
    """Build value groups as: '<sample> | <task>' plus '<sample> | all tasks'."""
    out: Dict[str, List[float]] = defaultdict(list)
    if df.empty or value_col not in df.columns or "sample_id" not in df.columns:
        return {}

    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    sample_ids = df["sample_id"].astype(str).tolist()
    conds = df["condition_label"].astype(str).tolist() if "condition_label" in df.columns else ["all recordings"] * len(df)

    for sample_id, cond, val in zip(sample_ids, conds, vals):
        if not np.isfinite(val):
            continue
        cond_label = f"{sample_id} | {cond}"
        all_label = f"{sample_id} | all tasks"
        out[cond_label].append(float(val))
        out[all_label].append(float(val))
    return dict(out)


def _label_sample_condition_values(sample_id: str, values_by_condition: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    all_vals: List[float] = []
    for cond, vals in sorted(values_by_condition.items(), key=lambda x: str(x[0])):
        arr = _finite_array(vals)
        if arr.size == 0:
            continue
        label = f"{sample_id} | {cond}"
        out[label].extend(arr.tolist())
        all_vals.extend(arr.tolist())
    if all_vals:
        out[f"{sample_id} | all tasks"].extend(all_vals)
    return dict(out)


def _collect_labeled_values_from_bundles(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    extractor: Callable[[ChTypeAccumulator], Dict[str, List[float]]],
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None:
            continue
        labeled = _label_sample_condition_values(bundle.sample_id, extractor(acc))
        for label, vals in labeled.items():
            out[label].extend(vals)
    return dict(out)


def _subject_points_labeled(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """One subject point per sample-task and per sample-all_tasks."""
    if df.empty or value_col not in df.columns or "sample_id" not in df.columns or "subject" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.loc[np.isfinite(tmp[value_col])].copy()
    if tmp.empty:
        return pd.DataFrame()

    recs: List[dict] = []

    by_cond = tmp.groupby(["sample_id", "condition_label", "subject"], dropna=False)
    for (sample_id, cond, subject), group in by_cond:
        vals = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        vals = _finite_array(vals)
        if vals.size == 0:
            continue
        tasks = sorted({str(t) for t in group.get("task", pd.Series([], dtype=str)).astype(str).tolist() if str(t) != "n/a"})
        val = float(np.nanmedian(vals))
        recs.append(
            {
                "condition_label": f"{sample_id} | {cond}",
                "subject": str(subject),
                "task": ",".join(tasks) if tasks else "n/a",
                value_col: val,
                "hover_entities": (
                    f"sample={sample_id}<br>subject={subject}<br>task={cond}<br>n_recordings={len(group)}"
                    + (f"<br>tasks={', '.join(tasks)}" if tasks else "")
                    + f"<br>value={val:.3g}"
                ),
            }
        )

    by_all = tmp.groupby(["sample_id", "subject"], dropna=False)
    for (sample_id, subject), group in by_all:
        vals = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        vals = _finite_array(vals)
        if vals.size == 0:
            continue
        tasks = sorted({str(t) for t in group.get("task", pd.Series([], dtype=str)).astype(str).tolist() if str(t) != "n/a"})
        val = float(np.nanmedian(vals))
        recs.append(
            {
                "condition_label": f"{sample_id} | all tasks",
                "subject": str(subject),
                "task": ",".join(tasks) if tasks else "n/a",
                value_col: val,
                "hover_entities": (
                    f"sample={sample_id}<br>subject={subject}<br>task=all tasks<br>n_recordings={len(group)}"
                    + (f"<br>tasks={', '.join(tasks)}" if tasks else "")
                    + f"<br>value={val:.3g}"
                ),
            }
        )

    if not recs:
        return pd.DataFrame()
    return pd.DataFrame(recs)


def _distribution_views(
    values_map: Dict[str, List[float]],
    points_df: pd.DataFrame,
    point_col: str,
    *,
    title_prefix: str,
    value_label: str,
    tab_id: str,
) -> str:
    if not values_map:
        return "<p>No values are available for this panel.</p>"

    box_raw = plot_box_with_subject_jitter(
        values_map,
        points_df,
        point_col=point_col,
        title=f"{title_prefix} - boxplot",
        y_title=value_label,
    )
    violin_density = plot_violin_with_subject_jitter(
        values_map,
        points_df,
        point_col=point_col,
        title=f"{title_prefix} - violin densities",
        y_title=value_label,
        show_density_ridge=True,
        spanmode="soft",
    )
    hist = plot_histogram_distribution(
        values_map,
        title=f"{title_prefix} - histogram",
        x_title=value_label,
    )
    dens = plot_density_distribution(
        values_map,
        title=f"{title_prefix} - density",
        x_title=value_label,
    )

    return _build_subtabs_html(
        tab_id,
        [
            (
                "Boxplot",
                _figure_block(
                    box_raw,
                    (
                        "Each box is one dataset-task group summarizing median, quartiles, and whiskers. "
                        "Labels include '<dataset> | <task>' and '<dataset> | all tasks'. "
                        "Jittered dots are one robust subject summary for the selected variant."
                    ),
                    normalized_variant=True,
                    norm_mode="y",
                ),
            ),
            (
                "Violin densities",
                _figure_block(
                    violin_density,
                    (
                        "Density-smoothed violin version of the same dataset-task groups. "
                        "Jittered dots are one robust subject summary for the selected variant."
                    ),
                    normalized_variant=True,
                    norm_mode="y",
                ),
            ),
            (
                "Histogram",
                _figure_block(
                    hist,
                    "Histogram + density view of the same distribution groups.",
                    normalized_variant=True,
                    norm_mode="x",
                ),
            ),
            (
                "Density",
                _figure_block(
                    dens,
                    "Kernel-density view of the same distribution groups.",
                    normalized_variant=True,
                    norm_mode="x",
                ),
            ),
        ],
        level=4,
    )


def _condition_figure_blocks_local(
    figures_by_condition: Dict[str, Optional[go.Figure]],
    interpretation: str,
    *,
    interpretation_by_condition: Optional[Dict[str, str]] = None,
    normalized_variant: bool = False,
    norm_mode: str = "y",
) -> str:
    chunks: List[str] = []
    for condition in sorted(figures_by_condition):
        fig = figures_by_condition[condition]
        if fig is None:
            continue
        chunks.append(f"<h4>{condition}</h4>")
        note = interpretation
        if interpretation_by_condition is not None and condition in interpretation_by_condition:
            note = f"{interpretation} {interpretation_by_condition[condition]}"
        chunks.append(
            _figure_block(
                fig,
                note,
                normalized_variant=normalized_variant,
                norm_mode=norm_mode,
            )
        )
    return "".join(chunks) if chunks else "<p>No figures are available for this panel.</p>"


def _run_values_by_condition_from_acc(acc: ChTypeAccumulator, value_col: str) -> Dict[str, List[float]]:
    df = _run_rows_dataframe(acc.run_rows)
    return _run_values_by_condition(df, value_col)


def _muscle_profile_map(acc: ChTypeAccumulator) -> Dict[str, List[Dict[str, np.ndarray]]]:
    out: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
    for cond, profiles in acc.muscle_profiles_by_condition.items():
        for arr in profiles:
            vals = np.asarray(arr, dtype=float)
            out[cond].append({"q50": vals, "mean": vals, "q95": vals})
    return dict(out)


def _ordered_variants(default_label: str) -> List[str]:
    ordered = [str(default_label)]
    for label in ["Median", "Mean", "Upper tail"]:
        if label != default_label:
            ordered.append(label)
    return ordered


def _build_metric_panel(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    df_all: pd.DataFrame,
    *,
    metric_name: str,
    value_label: str,
    default_variant: str,
    run_col_by_variant: Dict[str, str],
    channel_extractor_by_variant: Dict[str, Callable[[ChTypeAccumulator], Dict[str, List[float]]]],
    epoch_extractor_by_variant: Dict[str, Callable[[ChTypeAccumulator], Dict[str, List[float]]]],
    formula_a: str,
    formula_b: str,
    formula_c: str,
    fingerprint_spec: Optional[Tuple[str, str, str, str]] = None,
    heatmap_variants_getter: Optional[Callable[[ChTypeAccumulator], Dict[str, Dict[str, np.ndarray]]]] = None,
    heatmap_epoch_notes_getter: Optional[Callable[[ChTypeAccumulator], Dict[str, str]]] = None,
    topomap_attr: Optional[str] = None,
    tab_token: str = "tab",
) -> str:
    ordered = _ordered_variants(default_variant)

    def _panel_variant_tabs(
        panel_key: str,
        values_getter: Callable[[str], Dict[str, List[float]]],
        title_stub: str,
    ) -> str:
        tabs: List[Tuple[str, str]] = []
        for variant in ordered:
            vals = values_getter(variant)
            point_col = run_col_by_variant.get(variant, run_col_by_variant.get(default_variant, ""))
            points = _subject_points_labeled(df_all, point_col)
            tabs.append(
                (
                    variant,
                    _distribution_views(
                        vals,
                        points,
                        point_col=point_col,
                        title_prefix=f"{metric_name} {title_stub} [{variant}]",
                        value_label=value_label,
                        tab_id=f"{tab_token}-{_sanitize_token(metric_name)}-{panel_key}-{_sanitize_token(variant)}",
                    ),
                )
            )
        return _build_subtabs_html(
            f"{tab_token}-{_sanitize_token(metric_name)}-{panel_key}-variants",
            tabs,
            level=4,
        )

    panel_a = (
        "<div class='fig-note'><strong>A summary definition:</strong> "
        + formula_a
        + "</div>"
        + _panel_variant_tabs(
            "A",
            lambda variant: _run_values_labeled(df_all, run_col_by_variant.get(variant, "")),
            "recording distribution",
        )
    )
    panel_b = (
        "<div class='fig-note'><strong>B summary definition:</strong> "
        + formula_b
        + "</div>"
        + _panel_variant_tabs(
            "B",
            lambda variant: _collect_labeled_values_from_bundles(
                bundles,
                tab_name,
                channel_extractor_by_variant.get(variant, lambda _acc: {}),
            ),
            "epochs-per-channel distribution",
        )
    )
    panel_c = (
        "<div class='fig-note'><strong>C summary definition:</strong> "
        + formula_c
        + "</div>"
        + _panel_variant_tabs(
            "C",
            lambda variant: _collect_labeled_values_from_bundles(
                bundles,
                tab_name,
                epoch_extractor_by_variant.get(variant, lambda _acc: {}),
            ),
            "channels-per-epoch distribution",
        )
    )

    dist_tabs = _build_subtabs_html(
        f"{tab_token}-{_sanitize_token(metric_name)}-dist-panels",
        [
            ("A: Recording distributions", panel_a),
            ("B: Epochs per channel", panel_b),
            ("C: Channels per epoch", panel_c),
        ],
        level=3,
    )

    chunks = [f"<div class='metric-block'><h3>{metric_name}</h3>{dist_tabs}"]

    if fingerprint_spec is not None:
        x_col, y_col, x_label, y_label = fingerprint_spec
        fp_fig = plot_run_fingerprint_scatter_by_sample(
            df_all,
            x_col=x_col,
            y_col=y_col,
            title=f"Run fingerprint ({metric_name})",
            x_label=x_label,
            y_label=y_label,
        )
        chunks.append(
            _figure_block(
                fp_fig,
                (
                    "Each point is one recording. Color encodes dataset and legend toggles task groups. "
                    "Use this view to compare recording-level spread across datasets in the same metric space."
                ),
            )
        )

    if heatmap_variants_getter is not None:
        sample_tabs: List[Tuple[str, str]] = []
        for bundle in bundles:
            acc = bundle.tab_accumulators.get(tab_name)
            if acc is None:
                continue
            matrices_by_condition = heatmap_variants_getter(acc)
            if not matrices_by_condition:
                continue
            notes = heatmap_epoch_notes_getter(acc) if heatmap_epoch_notes_getter is not None else None
            figures = {
                cond: plot_heatmap_sorted_channels_windows(
                    variants,
                    title=f"{metric_name} channel-epoch map ({cond})",
                    color_title=value_label,
                    summary_mode=("upper_tail" if metric_name.lower().startswith("ptp") else "median"),
                    channel_names=(
                        getattr(acc, topomap_attr, {}).get(cond).layout.names
                        if (topomap_attr is not None and getattr(acc, topomap_attr, {}).get(cond) is not None)
                        else None
                    ),
                )
                for cond, variants in sorted(matrices_by_condition.items())
            }
            sample_tabs.append(
                (
                    bundle.sample_id,
                    _condition_figure_blocks_local(
                        figures,
                        (
                            "Heatmap cell = one channel-by-epoch value. "
                            "Bottom buttons switch heatmap variant (median/mean/upper tail) and "
                            "independently switch top/right profile central summaries."
                        ),
                        interpretation_by_condition=notes,
                        normalized_variant=True,
                        norm_mode="z",
                    ),
                )
            )
        if sample_tabs:
            chunks.append("<h4>Heatmaps by dataset</h4>" + _build_subtabs_html(
                f"{tab_token}-{_sanitize_token(metric_name)}-heatmaps-by-dataset",
                sample_tabs,
                level=3,
            ))

    if topomap_attr is not None:
        sample_tabs = []
        for bundle in bundles:
            acc = bundle.tab_accumulators.get(tab_name)
            if acc is None:
                continue
            payloads: Dict[str, TopomapPayload] = getattr(acc, topomap_attr, {})
            cond_tabs: List[Tuple[str, str]] = []
            for idx, (cond, payload) in enumerate(sorted(payloads.items())):
                fig_2d = plot_topomap_if_available(
                    payload,
                    title=f"{metric_name} topographic footprint ({cond})",
                    color_title=value_label,
                )
                fig_3d = plot_topomap_3d_if_available(
                    payload,
                    title=f"{metric_name} topographic footprint (3D) ({cond})",
                    color_title=value_label,
                )
                view_tabs: List[Tuple[str, str]] = []
                if fig_2d is not None:
                    view_tabs.append(
                        (
                            "2D",
                            _figure_block(
                                fig_2d,
                                (
                                    "Each marker is one channel summary value. For Elekta triplets, overlapping points are spread "
                                    "to preserve one MAG and two GRAD channel values."
                                ),
                                normalized_variant=True,
                                norm_mode="color",
                            ),
                        )
                    )
                if fig_3d is not None:
                    view_tabs.append(
                        (
                            "3D",
                            _figure_block(
                                fig_3d,
                                (
                                    "3D view of the same channel summaries. Rotate to inspect spatial clustering while keeping "
                                    "the same value scale as the 2D map."
                                ),
                                normalized_variant=True,
                                norm_mode="color",
                            ),
                        )
                    )
                if not view_tabs:
                    continue
                cond_tabs.append(
                    (
                        cond,
                        _build_subtabs_html(
                            f"{tab_token}-{_sanitize_token(metric_name)}-topo-{bundle.sample_id}-{idx}",
                            view_tabs,
                            level=4,
                        ),
                    )
                )
            html = _build_subtabs_html(
                f"{tab_token}-{_sanitize_token(metric_name)}-topo-conds-{bundle.sample_id}",
                cond_tabs,
                level=3,
            ) if cond_tabs else "<p>Topographic maps not shown: channel positions not available in stored outputs.</p>"
            sample_tabs.append((bundle.sample_id, html))
        if sample_tabs:
            chunks.append("<h4>Topographic maps by dataset</h4>" + _build_subtabs_html(
                f"{tab_token}-{_sanitize_token(metric_name)}-topo-by-dataset",
                sample_tabs,
                level=3,
            ))

    chunks.append("</div>")
    return "".join(chunks)


def _build_multi_metric_details_section(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    amplitude_unit: str,
    is_combined: bool,
    tab_token: str,
) -> str:
    df_all = _tab_dataframe(bundles, tab_name)
    if df_all.empty:
        return "<section><h2>QA metrics details</h2><p>No run-level summaries are available.</p></section>"

    def _std_epoch_map(acc: ChTypeAccumulator, field: str) -> Dict[str, List[float]]:
        return _epoch_values_from_profiles(acc.std_window_profiles_by_condition, field=field)

    def _ptp_epoch_map(acc: ChTypeAccumulator, field: str) -> Dict[str, List[float]]:
        return _epoch_values_from_profiles(acc.ptp_window_profiles_by_condition, field=field)

    def _muscle_epoch_map(acc: ChTypeAccumulator, field: str) -> Dict[str, List[float]]:
        return _epoch_values_from_profiles(_muscle_profile_map(acc), field=field)

    std_panel = _build_metric_panel(
        bundles,
        tab_name,
        df_all,
        metric_name="STD",
        value_label=f"STD ({amplitude_unit})",
        default_variant="Median",
        run_col_by_variant={"Median": "std_median", "Mean": "std_mean", "Upper tail": "std_upper_tail"},
        channel_extractor_by_variant={
            "Median": lambda acc: acc.std_dist_by_condition,
            "Mean": lambda acc: acc.std_dist_mean_by_condition,
            "Upper tail": lambda acc: acc.std_dist_upper_by_condition,
        },
        epoch_extractor_by_variant={
            "Median": lambda acc: _std_epoch_map(acc, "q50"),
            "Mean": lambda acc: _std_epoch_map(acc, "mean"),
            "Upper tail": lambda acc: _std_epoch_map(acc, "q95"),
        },
        formula_a="Median: median_c(median_t STD[c,t]); Mean: mean_c(mean_t STD[c,t]); Upper tail: q95_c(q95_t STD[c,t]).",
        formula_b="Median: median_t STD[c,t] per channel; Mean: mean_t STD[c,t] per channel; Upper tail: q95_t STD[c,t] per channel.",
        formula_c="Median: median_c STD[c,t] per epoch; Mean: mean_c STD[c,t] per epoch; Upper tail: q95_c STD[c,t] per epoch.",
        fingerprint_spec=(
            "std_median",
            "std_upper_tail",
            f"Median channel STD per recording ({amplitude_unit})",
            f"Upper-tail channel STD per recording ({amplitude_unit})",
        ),
        heatmap_variants_getter=lambda acc: _heatmap_variants_by_condition_from_acc(
            acc.std_heatmap_runs_by_condition,
            acc.std_heatmap_by_condition,
            acc.std_heatmap_sum_by_condition,
            acc.std_heatmap_count_by_condition,
            acc.std_heatmap_upper_by_condition,
        ),
        heatmap_epoch_notes_getter=lambda acc: _epoch_consistency_notes(acc.std_heatmap_runs_by_condition),
        topomap_attr="std_topomap_by_condition",
        tab_token=tab_token,
    )

    ptp_panel = _build_metric_panel(
        bundles,
        tab_name,
        df_all,
        metric_name="PtP",
        value_label=f"PtP ({amplitude_unit})",
        default_variant="Upper tail",
        run_col_by_variant={"Median": "ptp_median", "Mean": "ptp_mean", "Upper tail": "ptp_upper_tail"},
        channel_extractor_by_variant={
            "Median": lambda acc: acc.ptp_dist_by_condition,
            "Mean": lambda acc: acc.ptp_dist_mean_by_condition,
            "Upper tail": lambda acc: acc.ptp_dist_upper_by_condition,
        },
        epoch_extractor_by_variant={
            "Median": lambda acc: _ptp_epoch_map(acc, "q50"),
            "Mean": lambda acc: _ptp_epoch_map(acc, "mean"),
            "Upper tail": lambda acc: _ptp_epoch_map(acc, "q95"),
        },
        formula_a="Upper tail (default): q95_c(q99_t PtP[c,t]); Mean: mean_c(mean_t PtP[c,t]); Median: median_c(q95_t PtP[c,t]).",
        formula_b="Upper tail (default): q99_t PtP[c,t] per channel; Mean: mean_t PtP[c,t] per channel; Median: stored central channel summary.",
        formula_c="Upper tail (default): q95_c PtP[c,t] per epoch; Mean: mean_c PtP[c,t] per epoch; Median: median_c PtP[c,t] per epoch.",
        fingerprint_spec=(
            "ptp_median",
            "ptp_upper_tail",
            f"Median channel PtP per recording ({amplitude_unit})",
            f"Upper-tail channel PtP per recording ({amplitude_unit})",
        ),
        heatmap_variants_getter=lambda acc: _heatmap_variants_by_condition_from_acc(
            acc.ptp_heatmap_runs_by_condition,
            acc.ptp_heatmap_by_condition,
            acc.ptp_heatmap_sum_by_condition,
            acc.ptp_heatmap_count_by_condition,
            acc.ptp_heatmap_upper_by_condition,
        ),
        heatmap_epoch_notes_getter=lambda acc: _epoch_consistency_notes(acc.ptp_heatmap_runs_by_condition),
        topomap_attr="ptp_topomap_by_condition",
        tab_token=tab_token,
    )

    psd_panel = _build_metric_panel(
        bundles,
        tab_name,
        df_all,
        metric_name="PSD mains ratio",
        value_label="Relative power (unitless)",
        default_variant="Mean",
        run_col_by_variant={"Median": "mains_ratio", "Mean": "mains_ratio", "Upper tail": "mains_harmonics_ratio"},
        channel_extractor_by_variant={
            "Median": lambda acc: acc.psd_ratio_by_condition,
            "Mean": lambda acc: acc.psd_ratio_by_condition,
            "Upper tail": lambda acc: acc.psd_harmonics_ratio_by_condition,
        },
        epoch_extractor_by_variant={"Median": lambda _acc: {}, "Mean": lambda _acc: {}, "Upper tail": lambda _acc: {}},
        formula_a="Mean/Median: mean_c mains_ratio[c]; Upper tail: mean_c harmonics_ratio[c].",
        formula_b="Mean/Median: mains_ratio[c] per channel; Upper tail: harmonics_ratio[c] per channel.",
        formula_c="Not available: epoch-wise PSD summaries are not stored.",
        fingerprint_spec=(
            "mains_ratio",
            "mains_harmonics_ratio",
            "Mains relative power per recording",
            "Harmonics relative power per recording",
        ),
        topomap_attr="psd_topomap_by_condition",
        tab_token=tab_token,
    )

    ecg_panel = _build_metric_panel(
        bundles,
        tab_name,
        df_all,
        metric_name="ECG correlation",
        value_label="|r| (unitless)",
        default_variant="Mean",
        run_col_by_variant={"Median": "ecg_mean_abs_corr", "Mean": "ecg_mean_abs_corr", "Upper tail": "ecg_p95_abs_corr"},
        channel_extractor_by_variant={
            "Median": lambda acc: acc.ecg_corr_by_condition,
            "Mean": lambda acc: acc.ecg_corr_by_condition,
            "Upper tail": lambda acc: acc.ecg_corr_by_condition,
        },
        epoch_extractor_by_variant={"Median": lambda _acc: {}, "Mean": lambda _acc: {}, "Upper tail": lambda _acc: {}},
        formula_a="Mean/Median: mean_c |r[c]|; Upper tail: q95_c |r[c]|.",
        formula_b="Channel values are |r[c]|; median and mean variants coincide in stored outputs.",
        formula_c="Not available: epoch-wise ECG summaries are not stored.",
        fingerprint_spec=(
            "ecg_mean_abs_corr",
            "ecg_p95_abs_corr",
            "Mean |ECG correlation| per recording",
            "Upper-tail |ECG correlation| per recording",
        ),
        topomap_attr="ecg_topomap_by_condition",
        tab_token=tab_token,
    )

    eog_panel = _build_metric_panel(
        bundles,
        tab_name,
        df_all,
        metric_name="EOG correlation",
        value_label="|r| (unitless)",
        default_variant="Mean",
        run_col_by_variant={"Median": "eog_mean_abs_corr", "Mean": "eog_mean_abs_corr", "Upper tail": "eog_p95_abs_corr"},
        channel_extractor_by_variant={
            "Median": lambda acc: acc.eog_corr_by_condition,
            "Mean": lambda acc: acc.eog_corr_by_condition,
            "Upper tail": lambda acc: acc.eog_corr_by_condition,
        },
        epoch_extractor_by_variant={"Median": lambda _acc: {}, "Mean": lambda _acc: {}, "Upper tail": lambda _acc: {}},
        formula_a="Mean/Median: mean_c |r[c]|; Upper tail: q95_c |r[c]|.",
        formula_b="Channel values are |r[c]|; median and mean variants coincide in stored outputs.",
        formula_c="Not available: epoch-wise EOG summaries are not stored.",
        fingerprint_spec=(
            "eog_mean_abs_corr",
            "eog_p95_abs_corr",
            "Mean |EOG correlation| per recording",
            "Upper-tail |EOG correlation| per recording",
        ),
        topomap_attr="eog_topomap_by_condition",
        tab_token=tab_token,
    )

    muscle_panel = _build_metric_panel(
        bundles,
        tab_name,
        df_all,
        metric_name="Muscle score",
        value_label="Muscle score (z-score)",
        default_variant="Median",
        run_col_by_variant={"Median": "muscle_median", "Mean": "muscle_mean", "Upper tail": "muscle_p95"},
        channel_extractor_by_variant={
            "Median": lambda acc: _run_values_by_condition_from_acc(acc, "muscle_median"),
            "Mean": lambda acc: _run_values_by_condition_from_acc(acc, "muscle_mean"),
            "Upper tail": lambda acc: _run_values_by_condition_from_acc(acc, "muscle_p95"),
        },
        epoch_extractor_by_variant={
            "Median": lambda acc: _muscle_epoch_map(acc, "q50"),
            "Mean": lambda acc: _muscle_epoch_map(acc, "mean"),
            "Upper tail": lambda acc: _muscle_epoch_map(acc, "q95"),
        },
        formula_a="Median: median_t score[t] per run; Mean: mean_t score[t] per run; Upper tail: q95_t score[t] per run.",
        formula_b="Stored run-level summaries are reused for this panel (single channel-agnostic score sequence per run).",
        formula_c="Epoch values are score[t] per run; variants summarize pooled epoch-value distributions.",
        fingerprint_spec=(
            "muscle_median",
            "muscle_p95",
            "Median muscle score per recording",
            "Upper-tail muscle score per recording",
        ),
        tab_token=tab_token,
    )

    metric_tabs = _build_subtabs_html(
        f"multi-metric-details-{tab_token}",
        [
            ("STD", std_panel),
            ("PtP", ptp_panel),
            ("PSD", psd_panel),
            ("ECG", ecg_panel),
            ("EOG", eog_panel),
            ("Muscle", muscle_panel),
        ],
        level=2,
    )

    return (
        "<section>"
        "<h2>QA metrics details</h2>"
        "<p>Distribution and fingerprint plots merge all datasets in shared metric views. "
        "Heatmaps and topographic maps are organized in dataset-specific subtabs.</p>"
        + metric_tabs
        + "</section>"
    )


def _build_multi_cumulative_section(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    amplitude_unit: str,
) -> str:
    std_vals = _collect_labeled_values_from_bundles(bundles, tab_name, lambda acc: acc.std_dist_by_condition)
    ptp_vals = _collect_labeled_values_from_bundles(bundles, tab_name, lambda acc: acc.ptp_dist_by_condition)
    psd_vals = _collect_labeled_values_from_bundles(bundles, tab_name, lambda acc: acc.psd_ratio_by_condition)
    ecg_vals = _collect_labeled_values_from_bundles(bundles, tab_name, lambda acc: acc.ecg_corr_by_condition)
    eog_vals = _collect_labeled_values_from_bundles(bundles, tab_name, lambda acc: acc.eog_corr_by_condition)
    muscle_vals = _collect_labeled_values_from_bundles(bundles, tab_name, lambda acc: acc.muscle_scalar_by_condition)

    std_ecdf = _make_ecdf_figure(std_vals, "STD ECDF (all datasets/tasks)", f"STD ({amplitude_unit})")
    ptp_ecdf = _make_ecdf_figure(ptp_vals, "PtP ECDF (all datasets/tasks)", f"PtP ({amplitude_unit})")
    psd_ecdf = _make_ecdf_figure(psd_vals, "Mains relative power ECDF (all datasets/tasks)", "Mains ratio")
    ecg_ecdf = _make_ecdf_figure(ecg_vals, "ECG |r| ECDF (all datasets/tasks)", "|r|")
    eog_ecdf = _make_ecdf_figure(eog_vals, "EOG |r| ECDF (all datasets/tasks)", "|r|")
    muscle_ecdf = _make_ecdf_figure(muscle_vals, "Muscle score ECDF (all datasets/tasks)", "Muscle score")

    return (
        "<section>"
        "<h2>Cummulative distributions</h2>"
        "<p>Each curve corresponds to one '<dataset> | <task>' group plus '<dataset> | all tasks'.</p>"
        + _figure_block(std_ecdf, "Cumulative distribution of channel-level STD summaries across all datasets/tasks.")
        + _figure_block(ptp_ecdf, "Cumulative distribution of channel-level PtP summaries across all datasets/tasks.")
        + _figure_block(psd_ecdf, "Cumulative distribution of mains relative power across all datasets/tasks.")
        + _figure_block(ecg_ecdf, "Cumulative distribution of ECG correlation magnitudes across all datasets/tasks.")
        + _figure_block(eog_ecdf, "Cumulative distribution of EOG correlation magnitudes across all datasets/tasks.")
        + _figure_block(muscle_ecdf, "Cumulative distribution of muscle-score summaries across all datasets/tasks.")
        + "</section>"
    )


def _summary_distribution_cards_html(
    df: pd.DataFrame,
    *,
    amplitude_unit: str,
    summary_group_id: str,
    channels_by_metric: Dict[str, str],
) -> str:
    metric_defs = [
        ("STD", "std_mean", f"STD ({amplitude_unit})", "#2A6FBB"),
        ("PtP", "ptp_mean", f"PtP ({amplitude_unit})", "#2B9348"),
        ("PSD", "mains_ratio", "Mains relative power", "#E07A00"),
        ("ECG", "ecg_mean_abs_corr", "ECG |r| mean", "#C23B22"),
        ("EOG", "eog_mean_abs_corr", "EOG |r| mean", "#7B5CC4"),
        ("Muscle", "muscle_mean", "Muscle mean score", "#0B8F8C"),
    ]

    def _cards_for(defs: Sequence[Tuple[str, str, str, str]]) -> str:
        cards: List[str] = []
        for metric_name, col, y_label, color in defs:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) if col in df.columns else np.array([])
            n_rec = int(np.isfinite(vals).sum())
            n_ch = channels_by_metric.get(metric_name, "n/a")
            fig = _plot_summary_distribution_recordings(
                df,
                col,
                title=metric_name,
                y_label=y_label,
                color=color,
                enable_style_controls=False,
                show_counts_annotation=False,
            )
            cards.append(
                "<div class='summary-dist-card'>"
                + f"<div class='summary-card-meta'>N recordings={n_rec}; N channels={n_ch}</div>"
                + "<div class='fig summary-strip-fig'>"
                + _figure_to_div(fig, include_axis_size_control=False, include_plot_controls=False)
                + "</div>"
                + "</div>"
            )
        return f"<div class='summary-dist-grid' data-summary-group='{summary_group_id}'>" + "".join(cards) + "</div>"

    env_defs = metric_defs[:3]
    phys_defs = metric_defs[3:]
    return (
        "<div class='summary-category-block'>"
        "<h3>Environmental/Hardware noise</h3>"
        + _cards_for(env_defs)
        + "</div>"
        "<div class='summary-category-block'>"
        "<h3>Physiological noise</h3>"
        + _cards_for(phys_defs)
        + "</div>"
        + _summary_shared_controls_panel_html(summary_group_id)
        + (
            "<div class='summary-shared-note'><strong>How to interpret:</strong> "
            "Each panel is one metric. One dot is one recording summary (subject x task x run); hover shows recording entities. "
            "Violin + box represent pooled recording distribution. "
            "Units: STD/PtP use amplitude units shown on each y-axis, PSD is mains relative power, ECG/EOG are |r| magnitudes, and Muscle is score."
            "</div>"
        )
    )


def _build_multi_summary_distributions_section(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    *,
    amplitude_unit: str,
    is_combined: bool,
    tab_token: str,
) -> str:
    """Summary distributions for multi-sample QA (pooled + per-dataset subtabs)."""
    df_all = _tab_dataframe(bundles, tab_name)
    if df_all.empty:
        return "<section><h2>Summary distributions</h2><p>No run-level summaries are available.</p></section>"

    dist_tabs: List[Tuple[str, str]] = []
    base_group_token = f"multi-summary-dist-{tab_token}"

    def _ch_count_map_for_acc(acc: ChTypeAccumulator) -> Dict[str, int]:
        def _finite_count(values_by_condition: Dict[str, List[float]]) -> int:
            total = 0
            for vals in values_by_condition.values():
                arr = np.asarray(vals, dtype=float).reshape(-1)
                total += int(np.isfinite(arr).sum())
            return int(total)

        return {
            "STD": _finite_count(acc.std_dist_by_condition),
            "PtP": _finite_count(acc.ptp_dist_by_condition),
            "PSD": _finite_count(acc.psd_ratio_by_condition),
            "ECG": _finite_count(acc.ecg_corr_by_condition),
            "EOG": _finite_count(acc.eog_corr_by_condition),
            "Muscle": _finite_count(acc.muscle_scalar_by_condition),
        }

    pooled_counts_raw: Dict[str, List[int]] = defaultdict(list)
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None:
            continue
        for metric, n_ch in _ch_count_map_for_acc(acc).items():
            if int(n_ch) > 0:
                pooled_counts_raw[metric].append(int(n_ch))

    pooled_channels_by_metric: Dict[str, str] = {}
    for metric in ["STD", "PtP", "PSD", "ECG", "EOG", "Muscle"]:
        vals = pooled_counts_raw.get(metric, [])
        if not vals:
            pooled_channels_by_metric[metric] = "n/a"
        else:
            pooled_channels_by_metric[metric] = str(int(np.sum(vals)))

    dist_tabs.append(
        (
            "Pooled sample",
            _summary_distribution_cards_html(
                df_all,
                amplitude_unit=amplitude_unit,
                summary_group_id=f"{base_group_token}-pooled",
                channels_by_metric=pooled_channels_by_metric,
            ),
        )
    )
    for bundle in bundles:
        dff = df_all.loc[df_all["sample_id"].astype(str) == bundle.sample_id].copy()
        if dff.empty:
            continue
        acc = bundle.tab_accumulators.get(tab_name)
        channel_map_int = _ch_count_map_for_acc(acc) if acc is not None else {}
        channel_map = {k: (str(int(v)) if int(v) > 0 else "n/a") for k, v in channel_map_int.items()}
        dist_tabs.append(
            (
                bundle.sample_id,
                _summary_distribution_cards_html(
                    dff,
                    amplitude_unit=amplitude_unit,
                    summary_group_id=f"{base_group_token}-{_sanitize_token(bundle.sample_id)}",
                    channels_by_metric=channel_map,
                ),
            )
        )

    topo_specs = [
        ("STD", "std_topomap_by_condition", "std_topomap_count_by_condition", f"STD ({amplitude_unit})"),
        ("PtP", "ptp_topomap_by_condition", "ptp_topomap_count_by_condition", f"PtP ({amplitude_unit})"),
        ("PSD", "psd_topomap_by_condition", "psd_topomap_count_by_condition", "Mains relative power"),
        ("ECG", "ecg_topomap_by_condition", "ecg_topomap_count_by_condition", "ECG |r| mean"),
        ("EOG", "eog_topomap_by_condition", "eog_topomap_count_by_condition", "EOG |r| mean"),
        ("Muscle", "", "", "Muscle mean score"),
    ]
    sample_topo_tabs: List[Tuple[str, str]] = []
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None:
            continue
        metric_tabs: List[Tuple[str, str]] = []
        for idx, (metric_name, payload_attr, count_attr, color_title) in enumerate(topo_specs):
            if not payload_attr:
                metric_tabs.append((metric_name, "<p>Topographic maps are not available for this metric in stored outputs.</p>"))
                continue
            payloads = getattr(acc, payload_attr, {})
            counts = getattr(acc, count_attr, {}) if count_attr else None
            pooled = _pooled_topomap_payload(payloads, counts)
            if pooled is None:
                metric_tabs.append((metric_name, "<p>Topographic maps are not available for this metric in stored outputs.</p>"))
                continue
            fig2d = plot_topomap_if_available(
                pooled,
                title=f"{metric_name} pooled channel footprint ({bundle.sample_id})",
                color_title=color_title,
            )
            fig3d = plot_topomap_3d_if_available(
                pooled,
                title=f"{metric_name} pooled channel footprint (3D) ({bundle.sample_id})",
                color_title=color_title,
            )
            views: List[Tuple[str, str]] = []
            if fig2d is not None:
                views.append(("2D", _figure_block(fig2d, "Each channel value is pooled across all recordings and tasks for this dataset.")))
            if fig3d is not None:
                views.append(("3D", _figure_block(fig3d, "3D view of the same pooled channel values for this dataset.")))
            metric_tabs.append(
                (
                    metric_name,
                    _build_subtabs_html(
                        f"multi-summary-topo-{tab_token}-{bundle.sample_id}-{re.sub(r'[^a-z0-9]+', '-', metric_name.lower())}-{idx}",
                        views,
                        level=4,
                    ) if views else "<p>Topographic maps are not available for this metric.</p>",
                )
            )
        sample_topo_tabs.append(
            (
                bundle.sample_id,
                _build_subtabs_html(
                    f"multi-summary-topo-metric-{tab_token}-{bundle.sample_id}",
                    metric_tabs,
                    level=3,
                ),
            )
        )

    topo_html = _build_subtabs_html(f"multi-summary-topo-samples-{tab_token}", sample_topo_tabs, level=2) if sample_topo_tabs else "<p>Topographic maps are not available.</p>"

    return (
        "<section>"
        "<h2>Summary distributions</h2>"
        "<p>Task-agnostic compact view. One dot equals one recording summary; hover contains recording entities.</p>"
        + _build_subtabs_html(f"multi-summary-dist-{tab_token}", dist_tabs, level=2)
        + "<h3>Pooled channel topomaps by dataset</h3>"
        + topo_html
        + "</section>"
    )


def _build_dataset_subtab_section(
    bundles: Sequence[SampleBundle],
    tab_name: str,
    *,
    section_title: str,
    section_intro: str,
    section_builder: Callable[[ChTypeAccumulator, str, bool], str],
    amplitude_unit: str,
    is_combined: bool,
    tab_token: str,
) -> str:
    sample_tabs: List[Tuple[str, str]] = []
    for bundle in bundles:
        acc = bundle.tab_accumulators.get(tab_name)
        if acc is None or acc.run_count == 0:
            continue
        section_html = section_builder(acc, amplitude_unit, is_combined)
        sample_tabs.append((bundle.sample_id, _strip_outer_section(section_html)))

    if not sample_tabs:
        return f"<section><h2>{section_title}</h2><p>No dataset content is available for this tab.</p></section>"

    return (
        "<section>"
        f"<h2>{section_title}</h2>"
        f"<p>{section_intro}</p>"
        + _build_subtabs_html(
            f"multi-{tab_token}-{_sanitize_token(section_title)}-datasets",
            sample_tabs,
            level=2,
        )
        + "</section>"
    )


def _build_tab_content(
    bundles: Sequence[SampleBundle],
    tab_name: str,
) -> str:
    is_combined = tab_name == "Combined (mag+grad)"
    if is_combined:
        amplitude_unit = "mixed pT-based MEG units (all channels)"
        tab_token = "combined"
    elif tab_name == "MAG":
        amplitude_unit = "picoTesla (pT)"
        tab_token = "mag"
    elif tab_name == "EEG":
        amplitude_unit = "microVolts (µV)"
        tab_token = "eeg"
    else:
        amplitude_unit = "picoTesla/m (pT/m)"
        tab_token = "grad"

    cohort_section = _build_dataset_subtab_section(
        bundles,
        tab_name,
        section_title="Cohort QA overview",
        section_intro="Same cohort QA view as the single-dataset report, shown per dataset.",
        section_builder=lambda acc, unit, combined: _build_cohort_overview_section(acc, amplitude_unit=unit, is_combined=combined),
        amplitude_unit=amplitude_unit,
        is_combined=is_combined,
        tab_token=tab_token,
    )

    tasks_section = _build_dataset_subtab_section(
        bundles,
        tab_name,
        section_title="QA metrics across tasks",
        section_intro="Same task profiles as the single-dataset report, shown per dataset.",
        section_builder=lambda acc, unit, combined: _build_condition_effect_section(acc, amplitude_unit=unit, is_combined=combined),
        amplitude_unit=amplitude_unit,
        is_combined=is_combined,
        tab_token=tab_token,
    )

    details_section = _build_multi_metric_details_section(
        bundles,
        tab_name,
        amplitude_unit=amplitude_unit,
        is_combined=is_combined,
        tab_token=tab_token,
    )

    cumulative_section = _build_multi_cumulative_section(
        bundles,
        tab_name,
        amplitude_unit=amplitude_unit,
    )

    sections = [
        (
            "Summary distributions",
            _build_multi_summary_distributions_section(
                bundles,
                tab_name,
                amplitude_unit=amplitude_unit,
                is_combined=is_combined,
                tab_token=tab_token,
            ),
        ),
        ("Cohort QA overview", cohort_section),
        ("QA metrics across tasks", tasks_section),
        ("QA metrics details", details_section),
        ("Cummulative distributions", cumulative_section),
    ]
    return _build_subtabs_html(f"multi-main-{tab_token}", sections, level=1)


def _build_multi_sample_report_html(
    bundles: Sequence[SampleBundle],
    tab_order: Sequence[str],
) -> str:
    _reset_lazy_figure_store()
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = getattr(meg_qc, "__version__", "unknown")
    sample_names = ", ".join(bundle.sample_id for bundle in bundles)

    tab_buttons = []
    tab_divs = []
    for idx, tab in enumerate(tab_order):
        tab_id = f"tab-{idx}"
        active_class = " active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn{active_class}' data-target='{tab_id}'>{tab}</button>")
        content = _build_tab_content(bundles, tab)
        tab_divs.append(f"<div id='{tab_id}' class='tab-content{active_class}'>{content}</div>")
    lazy_payload_scripts = _lazy_payload_script_tags_html()

    sample_rows = "".join(
        "<li>"
        + f"<strong>{bundle.sample_id}</strong>: dataset=<code>{bundle.dataset_path}</code>, "
        + f"derivatives=<code>{bundle.derivatives_root}</code>"
        + "</li>"
        for bundle in bundles
    )

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>QA multi-sample report</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{
      font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;
      margin: 0;
      color: #1a1a1a;
      background: linear-gradient(135deg, #f6fbff, #edf6ff 45%, #f9fcff);
      transition: background 0.25s ease, color 0.25s ease;
    }}
    main {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; letter-spacing: 0.2px; }}
    h2 {{ margin-top: 24px; font-size: 22px; }}
    h3 {{ margin-top: 12px; font-size: 18px; }}
    h4 {{ margin-top: 14px; font-size: 15px; color: #284b63; }}
    .report-header {{
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    section {{
      background: rgba(255, 255, 255, 0.90);
      border: 1px solid #dce9f7;
      border-radius: 14px;
      padding: 16px 16px 10px;
      margin-top: 16px;
      box-shadow: 0 6px 24px rgba(5, 45, 79, 0.06);
    }}
    .metric-block {{
      border-top: 1px solid #e7eef8;
      margin-top: 14px;
      padding-top: 12px;
    }}
    .summary-category-block {{
      margin-top: 10px;
    }}
    .summary-category-block h3 {{
      margin: 0 0 8px;
      font-size: 18px;
      color: #1f3f63;
    }}
    .summary-dist-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(300px, 1fr));
      gap: 12px;
      margin-top: 6px;
    }}
    .summary-dist-card {{
      border: 1px solid #d9e7f8;
      border-radius: 10px;
      padding: 8px;
      background: #fbfdff;
      min-width: 0;
    }}
    .summary-card-meta {{
      font-size: 12px;
      font-weight: 700;
      color: #244466;
      background: #eef6ff;
      border: 1px solid #d3e6fb;
      border-radius: 7px;
      padding: 5px 8px;
      margin-bottom: 6px;
    }}
    .summary-shared-controls {{
      margin-top: 10px;
      padding: 10px 12px 12px;
      border: 1px solid #c7dbf2;
      border-radius: 10px;
      background: #f4f9ff;
    }}
    .summary-shared-note {{
      margin: 8px 2px 12px;
      font-size: 13px;
      line-height: 1.45;
      color: #2a425f;
    }}
    ul {{ margin: 6px 0 0 16px; }}
    pre {{
      white-space: pre-wrap;
      background: #f4f8fd;
      border: 1px solid #d8e4f3;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d5e3f3;
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef5fd; }}
    .fig {{
      border-top: 1px solid #e7eef8;
      margin-top: 10px;
      padding-top: 10px;
      overflow: visible;
    }}
    .fig-title {{
      font-size: 16px;
      font-weight: 700;
      color: #1a3a5c;
      text-align: center;
      padding: 10px 8px 2px;
      line-height: 1.3;
    }}
    .fig .js-plotly-plot {{
      width: 100% !important;
    }}
    .lazy-plot-wrap {{
      width: 100%;
    }}
    .plot-controls {{
      margin-top: 10px;
      padding: 10px 12px 12px;
      border: 1px solid #c7dbf2;
      border-radius: 10px;
      background: #f4f9ff;
      display: none;
    }}
    .plot-controls.active {{
      display: block;
    }}
    .plot-control-group {{
      margin-bottom: 10px;
    }}
    .plot-control-group:last-child {{
      margin-bottom: 0;
    }}
    .plot-control-title {{
      font-size: 12px;
      font-weight: 700;
      color: #21486f;
      margin-bottom: 6px;
      letter-spacing: 0.1px;
    }}
    .plot-control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }}
    .plot-control-btn {{
      border: 1px solid #78a9df;
      border-radius: 7px;
      background: #eaf3ff;
      color: #1f3f61;
      font-size: 12px;
      font-weight: 600;
      padding: 4px 10px;
      cursor: pointer;
    }}
    .plot-control-btn.active {{
      background: #d3e8ff;
      border-color: #4f90d8;
      color: #0f3256;
      box-shadow: inset 0 0 0 1px rgba(79,144,216,0.2);
    }}
    .fig-note {{
      margin: 6px 2px 14px;
      font-size: 13px;
      line-height: 1.45;
      color: #2a425f;
    }}
    .tab-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .tab-btn {{
      border: 1px solid #99bbe5;
      border-radius: 10px;
      background: #e7f1ff;
      padding: 8px 14px;
      cursor: pointer;
      font-size: 14px;
      color: #1f3b63;
      font-weight: 600;
    }}
    .tab-btn.active {{
      background: #1d4ed8;
      border-color: #1d4ed8;
      color: #ffffff;
      font-weight: 700;
    }}
    .tab-content {{
      display: none;
      margin-top: 8px;
    }}
    .tab-content.active {{
      display: block;
    }}
    .subtab-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 10px;
    }}
    .subtab-group {{
      border: 1px solid #d8e6f7;
      border-radius: 12px;
      padding: 10px 10px 8px;
      margin-top: 10px;
    }}
    .subtab-group.level-1 {{
      background: #dbeafe;
      border-color: #7fb0ea;
    }}
    .subtab-group.level-2 {{
      background: #ecf5ff;
      border-color: #a8c8ee;
    }}
    .subtab-group.level-3 {{
      background: #f5f9ff;
      border-color: #c1d8f2;
    }}
    .subtab-group.level-4 {{
      background: #ffffff;
      border-color: #d9e6f7;
    }}
    .subtab-group.level-1 > .subtab-row .subtab-btn {{
      background: #cfe3ff;
      border-color: #79a9e4;
      font-weight: 600;
    }}
    .subtab-group.level-2 > .subtab-row .subtab-btn {{
      background: #e2efff;
      border-color: #9fc2e8;
    }}
    .subtab-group.level-3 > .subtab-row .subtab-btn {{
      background: #edf5ff;
      border-color: #b2cfea;
    }}
    .subtab-btn {{
      border: 1px solid #9fc2e8;
      border-radius: 9px;
      background: #eaf3ff;
      padding: 6px 10px;
      cursor: pointer;
      font-size: 13px;
      color: #1f3f61;
      font-weight: 600;
    }}
    .subtab-btn.active {{
      background: #cfe3ff;
      border-color: #79a9e4;
      color: #16a34a;
      font-weight: 700;
    }}
    .subtab-content {{
      display: none;
    }}
    .subtab-content.active {{
      display: block;
    }}
    .fig-switch {{
      display: inline-flex;
      gap: 6px;
      margin: 6px 0 10px;
    }}
    .fig-switch-btn {{
      border: 1px solid #9fc2e8;
      border-radius: 8px;
      background: #eaf3ff;
      padding: 5px 10px;
      font-size: 12px;
      color: #1f3f61;
      cursor: pointer;
    }}
    .fig-switch-btn.active {{
      background: #d9ebff;
      border-color: #8db5dd;
      color: #16a34a;
      font-weight: 600;
    }}
    .report-tools {{
      display: flex;
      gap: 8px;
      margin-top: 8px;
      flex-wrap: wrap;
    }}
    .tool-btn {{
      border: 1px solid #7da9dd;
      border-radius: 9px;
      background: #dbeafe;
      padding: 7px 12px;
      font-size: 13px;
      color: #143a63;
      font-weight: 700;
      cursor: pointer;
    }}
    .tool-btn.active {{
      background: #1d4ed8;
      border-color: #1d4ed8;
      color: #ffffff;
    }}
    .fig-view {{
      display: none;
    }}
    .fig-view.active {{
      display: block;
    }}
    .loading-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(241, 245, 249, 0.95);
      backdrop-filter: blur(2px);
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: opacity 220ms ease;
    }}
    .loading-overlay.hidden {{
      opacity: 0;
      pointer-events: none;
    }}
    .loading-card {{
      min-width: 280px;
      max-width: 420px;
      padding: 18px 20px;
      border-radius: 14px;
      border: 1px solid #93c5fd;
      background: #ffffff;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.14);
      text-align: center;
    }}
    .loading-spinner {{
      width: 34px;
      height: 34px;
      margin: 0 auto 10px;
      border: 4px solid #bfdbfe;
      border-top-color: #1d4ed8;
      border-radius: 50%;
      animation: spin 0.9s linear infinite;
    }}
    .loading-title {{
      font-weight: 700;
      color: #1e3a5f;
      margin-bottom: 4px;
    }}
    .loading-subtitle {{
      color: #334e68;
      font-size: 13px;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <div id=\"report-loading-overlay\" class=\"loading-overlay\">
    <div class=\"loading-card\">
      <div class=\"loading-spinner\"></div>
      <div class=\"loading-title\">Loading QA report</div>
      <div class=\"loading-subtitle\">Rendering visible figures...</div>
    </div>
  </div>
  <main>
    <section>
      <div class=\"report-header\">
        <div>
          <h1>QA multi-sample report</h1>
          <p><strong>Generated:</strong> {generated}</p>
          <p><strong>MEGqc version:</strong> {version}</p>
          <p><strong>Datasets:</strong> {sample_names}</p>
          <p><strong>Important:</strong> This report mirrors the group QA structure while comparing multiple datasets within each channel-type tab.</p>
        </div>
      </div>
      <h3>Dataset paths</h3>
      <ul>{sample_rows}</ul>
      <div class=\"report-tools\">
        <button id=\"grid-toggle-btn\" class=\"tool-btn active\" type=\"button\">Hide grids</button>
      </div>
      <div class=\"tab-row\">
        {"".join(tab_buttons)}
      </div>
      {"".join(tab_divs)}
    </section>
  </main>
  {lazy_payload_scripts}
  <script>
    (function() {{
      const buttons = Array.from(document.querySelectorAll('.tab-btn'));
      const tabs = Array.from(document.querySelectorAll('.tab-content'));
      const gridToggleBtn = document.getElementById('grid-toggle-btn');
      const loadingOverlay = document.getElementById('report-loading-overlay');
      const lazyPayloadCache = {{}};
      let gridsVisible = true;

      function getPayloadFromScript(payloadId) {{
        if (!payloadId) {{
          return null;
        }}
        if (lazyPayloadCache[payloadId]) {{
          return lazyPayloadCache[payloadId];
        }}
        const payloadEl = document.getElementById(payloadId);
        if (!payloadEl || !payloadEl.textContent) {{
          return null;
        }}
        try {{
          const payload = JSON.parse(payloadEl.textContent);
          lazyPayloadCache[payloadId] = payload;
          payloadEl.textContent = '';
          if (payloadEl.parentNode) {{
            payloadEl.parentNode.removeChild(payloadEl);
          }}
          return payload;
        }} catch (err) {{
          return null;
        }}
      }}

      function hideLoadingOverlay() {{
        if (!loadingOverlay || loadingOverlay.dataset.hidden === '1') {{
          return;
        }}
        loadingOverlay.dataset.hidden = '1';
        loadingOverlay.classList.add('hidden');
        window.setTimeout(() => {{
          if (loadingOverlay && loadingOverlay.parentNode) {{
            loadingOverlay.parentNode.removeChild(loadingOverlay);
          }}
        }}, 260);
      }}

      function runPlotlyControlAction(plotEl, control) {{
        if (typeof Plotly === 'undefined' || !plotEl || !control) {{
          return;
        }}
        const method = String(control.method || 'restyle').toLowerCase();
        const args = Array.isArray(control.args) ? control.args : [];
        try {{
          if (method === 'relayout') {{
            Plotly.relayout(plotEl, ...(args || []));
          }} else if (method === 'update') {{
            Plotly.update(plotEl, ...(args || []));
          }} else {{
            Plotly.restyle(plotEl, ...(args || []));
          }}
        }} catch (err) {{
          // no-op
        }}
      }}

      function renderExternalControls(plotEl, controls) {{
        const wrap = plotEl ? plotEl.closest('.lazy-plot-wrap') : null;
        const panel = wrap ? wrap.querySelector('.plot-controls') : null;
        if (!panel) {{
          return;
        }}
        panel.innerHTML = '';
        const groups = Array.isArray(controls) ? controls : [];
        if (groups.length === 0) {{
          panel.classList.remove('active');
          return;
        }}
        panel.classList.add('active');
        groups.forEach((group) => {{
          const title = String((group && group.title) || 'Control');
          const buttons = Array.isArray(group && group.buttons) ? group.buttons : [];
          if (buttons.length === 0) {{
            return;
          }}
          const gEl = document.createElement('div');
          gEl.className = 'plot-control-group';

          const tEl = document.createElement('div');
          tEl.className = 'plot-control-title';
          tEl.textContent = title;
          gEl.appendChild(tEl);

          const row = document.createElement('div');
          row.className = 'plot-control-row';
          const activeIdx = Number.isFinite(Number(group.active)) ? Number(group.active) : 0;
          buttons.forEach((btn, idx) => {{
            const bEl = document.createElement('button');
            bEl.type = 'button';
            bEl.className = 'plot-control-btn' + (idx === activeIdx ? ' active' : '');
            bEl.textContent = String((btn && btn.label) || String(idx + 1));
            bEl.addEventListener('click', () => {{
              Array.from(row.querySelectorAll('.plot-control-btn')).forEach((x) => x.classList.remove('active'));
              bEl.classList.add('active');
              runPlotlyControlAction(plotEl, btn || {{}});
            }});
            row.appendChild(bEl);
          }});
          gEl.appendChild(row);
          panel.appendChild(gEl);
        }});
      }}

      const summaryStyleState = {{}};
      const summaryDotMap = {{1: 3.0, 2: 4.0, 3: 5.5, 4: 7.0, 5: 8.5, 6: 10.0, 7: 12.0, 8: 14.0}};
      const summaryDispMap = {{1: 0.00, 2: 0.02, 3: 0.04, 4: 0.06, 5: 0.08, 6: 0.10, 7: 0.12, 8: 0.14, 9: 0.16, 10: 0.18, 11: 0.20, 12: 0.22}};

      function getSummaryGrids(groupId) {{
        if (!groupId) return [];
        return Array.from(document.querySelectorAll(`.summary-dist-grid[data-summary-group="${{groupId}}"]`));
      }}

      function getSummaryPlots(groupId) {{
        const grids = getSummaryGrids(groupId);
        if (grids.length === 0) return [];
        const plots = [];
        grids.forEach((grid) => {{
          plots.push(...Array.from(grid.querySelectorAll('.js-plotly-plot')));
        }});
        return plots;
      }}

      function applySummaryLineLevel(plotEl, level) {{
        const lw = Number(level);
        const lineWidth = (plotEl.data || []).map((tr) => {{
          const t = String((tr && tr.type) || '');
          return (t === 'scatter' || t === 'scattergl' || t === 'violin' || t === 'box') ? lw : null;
        }});
        const markerLineWidth = (plotEl.data || []).map((tr) => {{
          const t = String((tr && tr.type) || '');
          const mode = String((tr && tr.mode) || '');
          if (t === 'histogram') return Math.max(0.5, lw * 0.6);
          if ((t === 'scatter' || t === 'scattergl') && mode.includes('markers')) return Math.max(0.25, lw * 0.2);
          return null;
        }});
        try {{
          Plotly.restyle(plotEl, {{'line.width': lineWidth, 'marker.line.width': markerLineWidth}});
        }} catch (err) {{}}
      }}

      function applySummaryDotLevel(plotEl, level) {{
        const size = Number(summaryDotMap[level] || 8.0);
        const markerSize = (plotEl.data || []).map((tr) => {{
          const t = String((tr && tr.type) || '');
          const mode = String((tr && tr.mode) || '');
          return ((t === 'scatter' || t === 'scattergl') && mode.includes('markers')) ? size : null;
        }});
        try {{
          Plotly.restyle(plotEl, {{'marker.size': markerSize}});
        }} catch (err) {{}}
      }}

      function _summaryScatterIdx(plotEl) {{
        const out = [];
        (plotEl.data || []).forEach((tr, idx) => {{
          const t = String((tr && tr.type) || '');
          const mode = String((tr && tr.mode) || '');
          if ((t === 'scatter' || t === 'scattergl') && mode.includes('markers')) out.push(idx);
        }});
        return out;
      }}

      function applySummaryDotVisibility(plotEl, visibleLevel) {{
        const visible = Number(visibleLevel) > 0;
        const traceVisible = (plotEl.data || []).map((tr) => {{
          const t = String((tr && tr.type) || '');
          const mode = String((tr && tr.mode) || '');
          return ((t === 'scatter' || t === 'scattergl') && mode.includes('markers')) ? visible : null;
        }});
        try {{
          Plotly.restyle(plotEl, {{'visible': traceVisible}});
        }} catch (err) {{}}
      }}

      function applySummaryDisplacement(plotEl, level) {{
        const shift = Number(summaryDispMap[level] ?? 0.0);
        if (!plotEl.__summaryBaseX) plotEl.__summaryBaseX = {{}};
        const idxs = _summaryScatterIdx(plotEl);
        const xUpdate = [];
        const traceIdx = [];
        idxs.forEach((i) => {{
          if (!plotEl.__summaryBaseX.hasOwnProperty(i)) {{
            const base = Array.isArray(plotEl.data[i].x) ? plotEl.data[i].x.slice() : [];
            plotEl.__summaryBaseX[i] = base;
          }}
          const base = plotEl.__summaryBaseX[i] || [];
          xUpdate.push(base.map((v) => Number(v) + shift));
          traceIdx.push(i);
        }});
        if (traceIdx.length === 0) return;
        try {{
          Plotly.restyle(plotEl, {{'x': xUpdate}}, traceIdx);
        }} catch (err) {{}}
      }}

      function applySummaryAxisLevel(plotEl, level) {{
        const tickMap = {{1: 9, 2: 10, 3: 12, 4: 14, 5: 16, 6: 18, 7: 20, 8: 22}};
        const tick = Number(tickMap[level] || 14);
        const title = tick + 2;
        const layout = plotEl.layout || {{}};
        const axisKeys = Object.keys(layout).filter((k) => k.startsWith('xaxis') || k.startsWith('yaxis'));
        const upd = {{}};
        if (axisKeys.length === 0) {{
          upd['xaxis.tickfont.size'] = tick;
          upd['xaxis.title.font.size'] = title;
          upd['yaxis.tickfont.size'] = tick;
          upd['yaxis.title.font.size'] = title;
        }} else {{
          axisKeys.forEach((k) => {{
            upd[`${{k}}.tickfont.size`] = tick;
            upd[`${{k}}.title.font.size`] = title;
          }});
        }}
        try {{
          Plotly.relayout(plotEl, upd);
        }} catch (err) {{}}
      }}

      function applySummaryState(groupId) {{
        const state = summaryStyleState[groupId] || {{line: 4, dot: 4, axis: 4, dotvis: 1, disp: 1}};
        const grids = getSummaryGrids(groupId);
        const run = () => {{
          const plots = getSummaryPlots(groupId);
          plots.forEach((plotEl) => {{
            applySummaryLineLevel(plotEl, state.line);
            applySummaryDotLevel(plotEl, state.dot);
            applySummaryDotVisibility(plotEl, state.dotvis);
            applySummaryDisplacement(plotEl, state.disp);
            applySummaryAxisLevel(plotEl, state.axis);
          }});
        }};
        if (grids.length > 0) {{
          Promise.all(grids.map((grid) => renderLazyInScope(grid))).then(run);
        }} else {{
          run();
        }}
      }}

      function bindSummarySharedControls(scopeRoot) {{
        const scope = scopeRoot || document;
        const panels = Array.from(scope.querySelectorAll('.summary-shared-controls'));
        panels.forEach((panel) => {{
          if (panel.dataset.bound === '1') return;
          panel.dataset.bound = '1';
          const groupId = panel.dataset.summaryGroup;
          if (!groupId) return;
          if (!summaryStyleState[groupId]) {{
            summaryStyleState[groupId] = {{line: 4, dot: 4, axis: 4, dotvis: 1, disp: 1}};
          }}
          Array.from(panel.querySelectorAll('.plot-control-btn[data-summary-kind]')).forEach((btn) => {{
            btn.addEventListener('click', () => {{
              const kind = String(btn.dataset.summaryKind || '');
              const level = Number(btn.dataset.level || 4);
              if (!(kind in summaryStyleState[groupId])) return;
              summaryStyleState[groupId][kind] = level;
              const row = btn.parentElement;
              if (row) {{
                Array.from(row.querySelectorAll('.plot-control-btn')).forEach((b) => b.classList.remove('active'));
              }}
              btn.classList.add('active');
              applySummaryState(groupId);
            }});
          }});
          applySummaryState(groupId);
        }});
      }}

      function renderLazyInScope(scopeRoot) {{
        if (typeof Plotly === 'undefined') {{
          return Promise.resolve();
        }}
        const scope = scopeRoot || document;
        const placeholders = Array.from(scope.querySelectorAll('.js-lazy-plot'));
        const renderPromises = [];
        placeholders.forEach((el) => {{
          if (el.dataset.rendered === '1') {{
            return;
          }}
          if (el.offsetParent === null) {{
            return;
          }}
          const payloadId = el.dataset.payloadId;
          const payload = getPayloadFromScript(payloadId);
          if (!payload || !payload.figure) {{
            return;
          }}
          try {{
            const renderResult = Plotly.newPlot(el, payload.figure.data || [], payload.figure.layout || {{}}, payload.config || {{responsive: true, displaylogo: false}});
            const postRender = (renderResult && typeof renderResult.then === 'function')
              ? renderResult
              : Promise.resolve(renderResult);
            const withFrames = postRender.then(() => {{
              const frames = (payload.figure && payload.figure.frames) ? payload.figure.frames : [];
              if (frames && frames.length > 0 && typeof Plotly.addFrames === 'function') {{
                return Plotly.addFrames(el, frames).catch(() => undefined);
              }}
              return undefined;
            }}).then(() => {{
              renderExternalControls(el, payload.controls || []);
            }});
            el.dataset.rendered = '1';
            renderPromises.push(withFrames.catch(() => undefined));
          }} catch (err) {{
            // no-op
          }}
        }});
        return renderPromises.length > 0 ? Promise.all(renderPromises).then(() => undefined) : Promise.resolve();
      }}

      function teardownLazyInScope(scopeRoot) {{
        if (typeof Plotly === 'undefined') return;
        const scope = scopeRoot || document;
        const placeholders = Array.from(scope.querySelectorAll('.js-lazy-plot[data-rendered="1"]'));
        placeholders.forEach((el) => {{
          try {{
            Plotly.purge(el);
          }} catch (err) {{
            // no-op
          }}
          el.innerHTML = '';
          delete el.dataset.rendered;
          const wrap = el.closest('.lazy-plot-wrap');
          const panel = wrap ? wrap.querySelector('.plot-controls') : null;
          if (panel) {{
            panel.innerHTML = '';
            panel.classList.remove('active');
          }}
        }});
      }}

      function applyGridToPlot(plotEl, show) {{
        if (typeof Plotly === 'undefined' || !plotEl) {{
          return;
        }}
        const layout = plotEl.layout || {{}};
        const axisKeys = Object.keys(layout).filter((k) => k.startsWith('xaxis') || k.startsWith('yaxis'));
        const relayoutUpdate = {{}};
        if (axisKeys.length === 0) {{
          relayoutUpdate['xaxis.showgrid'] = show;
          relayoutUpdate['yaxis.showgrid'] = show;
        }} else {{
          axisKeys.forEach((k) => {{
            relayoutUpdate[`${{k}}.showgrid`] = show;
          }});
        }}
        try {{
          Plotly.relayout(plotEl, relayoutUpdate);
        }} catch (err) {{
          // no-op
        }}
      }}

      function applyGridState(show, scopeRoot) {{
        const scope = scopeRoot || document;
        const plots = Array.from(scope.querySelectorAll('.js-plotly-plot'));
        plots.forEach((plotEl) => applyGridToPlot(plotEl, show));
      }}

      function updateGridButtonState() {{
        if (!gridToggleBtn) return;
        gridToggleBtn.textContent = gridsVisible ? 'Hide grids' : 'Show grids';
        gridToggleBtn.classList.toggle('active', gridsVisible);
      }}

      function resizePlots(targetId) {{
        if (typeof Plotly === 'undefined') {{
          return Promise.resolve();
        }}
        const root = document.getElementById(targetId);
        if (!root) {{
          return Promise.resolve();
        }}
        return renderLazyInScope(root).then(() => {{
          const plots = Array.from(root.querySelectorAll('.js-plotly-plot'));
          plots.forEach((plotEl) => {{
            try {{
              Plotly.Plots.resize(plotEl);
            }} catch (err) {{
              // no-op
            }}
          }});
          if (!gridsVisible) {{
            applyGridState(false, root);
          }}
          bindSummarySharedControls(root);
        }});
      }}

      function activate(targetId) {{
        const activePrev = tabs.find(t => t.classList.contains('active'));
        if (activePrev && activePrev.id !== targetId) {{
          teardownLazyInScope(activePrev);
        }}
        tabs.forEach(t => t.classList.toggle('active', t.id === targetId));
        buttons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        window.requestAnimationFrame(() => {{
          resizePlots(targetId).then(() => {{
            window.setTimeout(() => resizePlots(targetId), 120);
          }});
        }});
      }}

      buttons.forEach(btn => {{
        btn.addEventListener('click', () => activate(btn.dataset.target));
      }});
      if (buttons.length > 0) {{
        activate(buttons[0].dataset.target);
      }}

      if (gridToggleBtn) {{
        updateGridButtonState();
        gridToggleBtn.addEventListener('click', () => {{
          gridsVisible = !gridsVisible;
          applyGridState(gridsVisible, document);
          updateGridButtonState();
        }});
      }}

      function activateSubtab(groupId, targetId) {{
        const subButtons = Array.from(document.querySelectorAll(`.subtab-btn[data-tab-group="${{groupId}}"]`));
        const subPanels = Array.from(document.querySelectorAll(`.subtab-content[data-tab-group="${{groupId}}"]`));
        const prevPanel = subPanels.find((p) => p.classList.contains('active'));
        if (prevPanel && prevPanel.id !== targetId) {{
          teardownLazyInScope(prevPanel);
        }}
        subPanels.forEach(p => p.classList.toggle('active', p.id === targetId));
        subButtons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        const activePanel = document.getElementById(targetId);
        if (activePanel) {{
          renderLazyInScope(activePanel);
        }}
        const activeTop = tabs.find(t => t.classList.contains('active'));
        if (activeTop) {{
          window.requestAnimationFrame(() => resizePlots(activeTop.id));
        }}
      }}

      const subButtonsAll = Array.from(document.querySelectorAll('.subtab-btn'));
      const seenGroups = new Set();
      subButtonsAll.forEach(btn => {{
        const groupId = btn.dataset.tabGroup;
        if (!groupId) {{
          return;
        }}
        btn.addEventListener('click', () => activateSubtab(groupId, btn.dataset.target));
        seenGroups.add(groupId);
      }});
      seenGroups.forEach(groupId => {{
        const firstBtn = document.querySelector(`.subtab-btn[data-tab-group="${{groupId}}"]`);
        if (firstBtn) {{
          activateSubtab(groupId, firstBtn.dataset.target);
        }}
      }});
      bindSummarySharedControls(document);

      function activateFigVariant(toggleId, targetId) {{
        const root = document.querySelector(`.fig-switch[data-fig-toggle="${{toggleId}}"]`);
        if (!root) return;
        const btns = Array.from(root.querySelectorAll('.fig-switch-btn'));
        btns.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
        const raw = document.getElementById(`${{toggleId}}-raw`);
        const norm = document.getElementById(`${{toggleId}}-norm`);
        if (raw) raw.classList.toggle('active', `${{toggleId}}-raw` === targetId);
        if (norm) norm.classList.toggle('active', `${{toggleId}}-norm` === targetId);
        const activeView = document.getElementById(targetId);
        if (activeView) {{
          renderLazyInScope(activeView);
        }}
        const activeTop = tabs.find(t => t.classList.contains('active'));
        if (activeTop) {{
          window.requestAnimationFrame(() => resizePlots(activeTop.id));
        }}
      }}

      Array.from(document.querySelectorAll('.fig-switch')).forEach((sw) => {{
        const toggleId = sw.dataset.figToggle;
        Array.from(sw.querySelectorAll('.fig-switch-btn')).forEach((btn) => {{
          btn.addEventListener('click', () => activateFigVariant(toggleId, btn.dataset.target));
        }});
        const first = sw.querySelector('.fig-switch-btn');
        if (first) activateFigVariant(toggleId, first.dataset.target);
      }});

      window.addEventListener('resize', () => {{
        const active = tabs.find(t => t.classList.contains('active'));
        if (active) {{
          resizePlots(active.id);
        }}
      }});
      window.requestAnimationFrame(() => {{
        const active = tabs.find(t => t.classList.contains('active'));
        const renderPromise = active ? resizePlots(active.id) : Promise.resolve();
        renderPromise.then(() => {{
          window.setTimeout(hideLoadingOverlay, 120);
        }}).catch(() => {{
          hideLoadingOverlay();
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def make_multi_sample_group_plots_meg_qc(
    dataset_paths: Sequence[str],
    derivatives_bases: Optional[Sequence[Optional[str]]] = None,
    output_report_path: Optional[str] = None,
    n_jobs: int = 1,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Build one HTML report comparing multiple MEGqc datasets.

    Parameters
    ----------
    dataset_paths
        Paths to two or more BIDS datasets.
    derivatives_bases
        Optional list (same length as ``dataset_paths``) of external derivatives
        parent folders. Use ``None`` for default in-dataset derivatives.
    output_report_path
        Optional explicit output path for the final HTML report.
    n_jobs
        Number of parallel workers used for subject-level run loading within
        each dataset. Use ``1`` for sequential mode, or ``-1`` for all cores.
    analysis_mode
        Analysis root selection mode (``legacy``, ``new``, ``reuse``, ``latest``).
    analysis_id
        Profile ID used with ``analysis_mode='reuse'``.

    Returns
    -------
    dict
        Mapping ``{"report": Path(...)}`` for the generated report.
    """
    if not dataset_paths:
        print("___MEGqc___: Multi-sample QA: no datasets were provided.")
        return {}
    if len(dataset_paths) < 2:
        print("___MEGqc___: Multi-sample QA: provide at least two datasets for comparison.")
        return {}

    if derivatives_bases is None:
        derivatives_bases = [None] * len(dataset_paths)
    if len(derivatives_bases) != len(dataset_paths):
        raise ValueError("derivatives_bases must be None or have same length as dataset_paths")

    bundles: List[SampleBundle] = []
    for ds_path, der_base in zip(dataset_paths, derivatives_bases):
        bundle = _collect_sample_bundle(
            ds_path,
            der_base,
            n_jobs=n_jobs,
            analysis_mode=analysis_mode,
            analysis_id=analysis_id,
        )
        if bundle is not None:
            bundles.append(bundle)

    if len(bundles) < 2:
        print("___MEGqc___: Multi-sample QA: fewer than two datasets had usable derivatives.")
        return {}

    # ── Classify bundles by modality ──────────────────────────────────
    meg_tabs_all = ["Combined (mag+grad)", "MAG", "GRAD"]
    eeg_tabs_all = ["EEG"]

    def _bundle_has_modality(b, tab_name):
        acc = b.tab_accumulators.get(tab_name)
        return acc is not None and acc.run_count > 0

    meg_bundles = [b for b in bundles if any(_bundle_has_modality(b, t) for t in meg_tabs_all)]
    eeg_bundles = [b for b in bundles if _bundle_has_modality(b, "EEG")]

    # Determine primary reports dir and naming
    primary_reports_dir = bundles[0].reports_dir
    ids = [_sanitize_token(b.sample_id) for b in bundles]
    name_suffix = "_vs_".join(ids[:4])
    if len(ids) > 4:
        name_suffix = f"{name_suffix}_and_{len(ids) - 4}_more"

    result = {}

    # ── MEG multi-sample report ───────────────────────────────────────
    if len(meg_bundles) >= 2:
        # Include MEG tabs that any bundle provides — tabs with missing data in
        # some bundles will show a "no data" placeholder rather than disappearing.
        meg_shared = [t for t in meg_tabs_all
                      if any(_bundle_has_modality(b, t) for b in meg_bundles)]
        if meg_shared:
            meg_html = _build_multi_sample_report_html(meg_bundles, meg_shared)
            meg_dir = primary_reports_dir / "meg"
            meg_dir.mkdir(parents=True, exist_ok=True)
            if output_report_path is not None:
                meg_path = Path(output_report_path).parent / "meg" / (Path(output_report_path).stem + "_meg.html")
            else:
                meg_path = meg_dir / f"QA_multi_sample_report_{name_suffix}_meg.html"
            meg_path.parent.mkdir(parents=True, exist_ok=True)
            meg_path.write_text(meg_html, encoding="utf-8")
            result["meg_report"] = meg_path
            print(f"___MEGqc___: Multi-sample MEG QA report created: {meg_path}")
            meg_ids = [b.sample_id for b in meg_bundles]
            excluded = [b.sample_id for b in bundles if b not in meg_bundles]
            if excluded:
                print(f"___MEGqc___:   MEG comparison includes [{', '.join(meg_ids)}], excludes [{', '.join(excluded)}] (no MEG data)")
        else:
            print("___MEGqc___: Multi-sample QA: MEG datasets found but no shared MEG tabs across them.")
    elif len(meg_bundles) == 1:
        print(f"___MEGqc___: Multi-sample QA: only one MEG dataset ({meg_bundles[0].sample_id}), skipping MEG comparison.")

    # ── EEG multi-sample report ───────────────────────────────────────
    if len(eeg_bundles) >= 2:
        eeg_html = _build_multi_sample_report_html(eeg_bundles, eeg_tabs_all)
        eeg_dir = primary_reports_dir / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        if output_report_path is not None:
            eeg_path = Path(output_report_path).parent / "eeg" / (Path(output_report_path).stem + "_eeg.html")
        else:
            eeg_path = eeg_dir / f"QA_multi_sample_report_{name_suffix}_eeg.html"
        eeg_path.parent.mkdir(parents=True, exist_ok=True)
        eeg_path.write_text(eeg_html, encoding="utf-8")
        result["eeg_report"] = eeg_path
        print(f"___MEGqc___: Multi-sample EEG QA report created: {eeg_path}")
        eeg_ids = [b.sample_id for b in eeg_bundles]
        excluded = [b.sample_id for b in bundles if b not in eeg_bundles]
        if excluded:
            print(f"___MEGqc___:   EEG comparison includes [{', '.join(eeg_ids)}], excludes [{', '.join(excluded)}] (no EEG data)")
    elif len(eeg_bundles) == 1:
        print(f"___MEGqc___: Multi-sample QA: only one EEG dataset ({eeg_bundles[0].sample_id}), skipping EEG comparison.")

    # ── No combined agnostic report — only per-modality reports ──────
    if not result:
        print("___MEGqc___: Multi-sample QA: could not produce any comparison reports.")

    return result
