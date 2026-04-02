"""Dataset-level QC plotting from Global Quality Index summary tables.

This module builds an interactive HTML QC report from per-modality GQI TSVs:
``summary_reports/group_metrics/meg/Global_Quality_Index_attempt_*_meg.tsv``
``summary_reports/group_metrics/eeg/Global_Quality_Index_attempt_*_eeg.tsv``
produced by the GQI calculation pipeline.  Legacy combined TSVs at the root
``group_metrics/`` level are also supported for backward compatibility.

Public entrypoint
-----------------
``make_group_qc_plots_meg_qc(dataset_path, input_tsv=None, output_html=None,\n                              attempt=None, derivatives_base=None)``
"""

from __future__ import annotations

import configparser
import datetime as dt
import html
import json
import os
import re
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

import meg_qc
from meg_qc.calculation.meg_qc_pipeline import resolve_analysis_root


METRIC_ORDER = ("GQI", "STD", "PtP", "PSD", "ECG", "EOG", "Muscle")
TOP_TABS = ("Combined (mag+grad)", "MAG", "GRAD", "EEG")

def _adaptive_legend_layout(n_entries: int, groupclick: str = "") -> dict:
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
    d: dict = {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0.0,
        "font": {"size": 11},
        "itemsizing": "constant",
        "tracegroupgap": 4,
    }
    if groupclick:
        d["groupclick"] = groupclick
    return {
        "legend": d,
        "margin": {"l": 55, "r": 30, "t": margin_t, "b": 60},
        "height": max(620, margin_t + 60 + 400),
    }


_LAZY_PLOT_COUNTER = count(1)
_LAZY_PAYLOAD_COUNTER = count(1)
_LAZY_PLOT_PAYLOADS: Dict[str, str] = {}


@dataclass(frozen=True)
class ComponentSpec:
    """One plottable scalar component inside a QC metric tab."""

    key: str
    label: str
    unit: str
    mag_col: Optional[str] = None
    grad_col: Optional[str] = None
    eeg_col: Optional[str] = None
    global_col: Optional[str] = None


@dataclass
class QCDatasetBundle:
    """Resolved and loaded inputs for one dataset QC report source."""

    dataset_path: str
    dataset_name: str
    derivatives_root: str
    reports_dir: Path
    tsv_path: Path
    cfg_path: Optional[Path]
    attempt: Optional[int]
    df: pd.DataFrame
    general_settings_snapshot: str
    gqi_settings_snapshot: str


COMPONENTS: Dict[str, List[ComponentSpec]] = {
    "GQI": [
        ComponentSpec("gqi", "Global Quality Index", "% quality", global_col="GQI"),
        ComponentSpec("pen_ch", "Variability penalty", "penalty units", global_col="GQI_penalty_ch"),
        ComponentSpec("pen_corr", "Correlational penalty", "penalty units", global_col="GQI_penalty_corr"),
        ComponentSpec("pen_mus", "Muscle penalty", "penalty units", global_col="GQI_penalty_mus"),
        ComponentSpec("pen_psd", "PSD penalty", "penalty units", global_col="GQI_penalty_psd"),
        ComponentSpec("gqi_std", "GQI STD component", "%", global_col="GQI_std_pct"),
        ComponentSpec("gqi_ptp", "GQI PtP component", "%", global_col="GQI_ptp_pct"),
        ComponentSpec("gqi_ecg", "GQI ECG component", "%", global_col="GQI_ecg_pct"),
        ComponentSpec("gqi_eog", "GQI EOG component", "%", global_col="GQI_eog_pct"),
        ComponentSpec("gqi_mus", "GQI muscle component", "%", global_col="GQI_muscle_pct"),
        ComponentSpec("gqi_psd", "GQI PSD-noise component", "%", global_col="GQI_psd_noise_pct"),
    ],
    "STD": [
        ComponentSpec(
            "std_ts_noisy",
            "Time-series noisy channels",
            "% channels",
            mag_col="STD_ts_noisy_channels_mag_percentage",
            grad_col="STD_ts_noisy_channels_grad_percentage",
            eeg_col="STD_ts_noisy_channels_eeg_percentage",
        ),
        ComponentSpec(
            "std_ts_flat",
            "Time-series flat channels",
            "% channels",
            mag_col="STD_ts_flat_channels_mag_percentage",
            grad_col="STD_ts_flat_channels_grad_percentage",
            eeg_col="STD_ts_flat_channels_eeg_percentage",
        ),
        ComponentSpec(
            "std_ep_noisy",
            "Noisy epochs",
            "% epochs",
            mag_col="STD_ep_mag_noisy_percentage",
            grad_col="STD_ep_grad_noisy_percentage",
            eeg_col="STD_ep_eeg_noisy_percentage",
        ),
        ComponentSpec(
            "std_ep_flat",
            "Flat epochs",
            "% epochs",
            mag_col="STD_ep_mag_flat_percentage",
            grad_col="STD_ep_grad_flat_percentage",
            eeg_col="STD_ep_eeg_flat_percentage",
        ),
    ],
    "PtP": [
        ComponentSpec(
            "ptp_ts_noisy",
            "Time-series noisy channels",
            "% channels",
            mag_col="PTP_ts_noisy_channels_mag_percentage",
            grad_col="PTP_ts_noisy_channels_grad_percentage",
            eeg_col="PTP_ts_noisy_channels_eeg_percentage",
        ),
        ComponentSpec(
            "ptp_ts_flat",
            "Time-series flat channels",
            "% channels",
            mag_col="PTP_ts_flat_channels_mag_percentage",
            grad_col="PTP_ts_flat_channels_grad_percentage",
            eeg_col="PTP_ts_flat_channels_eeg_percentage",
        ),
        ComponentSpec(
            "ptp_ep_noisy",
            "Noisy epochs",
            "% epochs",
            mag_col="PTP_ep_mag_noisy_percentage",
            grad_col="PTP_ep_grad_noisy_percentage",
            eeg_col="PTP_ep_eeg_noisy_percentage",
        ),
        ComponentSpec(
            "ptp_ep_flat",
            "Flat epochs",
            "% epochs",
            mag_col="PTP_ep_mag_flat_percentage",
            grad_col="PTP_ep_grad_flat_percentage",
            eeg_col="PTP_ep_eeg_flat_percentage",
        ),
    ],
    "PSD": [
        ComponentSpec(
            "psd_noise",
            "PSD noise burden",
            "% relative power",
            mag_col="PSD_noise_mag_percentage",
            grad_col="PSD_noise_grad_percentage",
            eeg_col="PSD_noise_eeg_percentage",
        ),
    ],
    "ECG": [
        ComponentSpec(
            "ecg_high_corr",
            "High-correlation channels",
            "% channels",
            mag_col="ECG_mag_high_corr_percentage",
            grad_col="ECG_grad_high_corr_percentage",
            eeg_col="ECG_eeg_high_corr_percentage",
        ),
    ],
    "EOG": [
        ComponentSpec(
            "eog_high_corr",
            "High-correlation channels",
            "% channels",
            mag_col="EOG_mag_high_corr_percentage",
            grad_col="EOG_grad_high_corr_percentage",
            eeg_col="EOG_eeg_high_corr_percentage",
        ),
    ],
    "Muscle": [
        ComponentSpec("muscle_events", "Muscle events", "event count", global_col="Muscle_events_num"),
        ComponentSpec("muscle_rate", "Muscle event rate", "% of events", global_col="__muscle_event_rate_pct"),
        ComponentSpec("gqi_muscle_component", "GQI muscle component", "%", global_col="GQI_muscle_pct"),
    ],
}


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------


def _attempt_from_name(path: Path) -> Optional[int]:
    match = re.search(r"attempt_(\d+)", path.name)
    return int(match.group(1)) if match else None


def _latest_attempt_tsv(group_metrics_dir: Path) -> Optional[Path]:
    """Find the latest GQI attempt TSV, scanning per-modality subdirs first.

    Looks in ``group_metrics/meg/`` and ``group_metrics/eeg/`` for
    per-modality files (``Global_Quality_Index_attempt_*_{meg,eeg}.tsv``).
    Falls back to legacy combined files in ``group_metrics/`` for backward
    compatibility with older analysis runs.
    """
    files: list = []
    # Per-modality subdirectories (new layout)
    for subdir in ("meg", "eeg"):
        d = group_metrics_dir / subdir
        if d.is_dir():
            files.extend(d.glob("Global_Quality_Index_attempt_*.tsv"))
    # Fallback: legacy combined files in root group_metrics/
    if not files:
        files = list(group_metrics_dir.glob("Global_Quality_Index_attempt_*.tsv"))
    if not files:
        return None
    files.sort(key=lambda p: (_attempt_from_name(p) or -1, p.stat().st_mtime), reverse=True)
    return files[0]


def _all_attempt_tsvs(group_metrics_dir: Path, attempt: Optional[int] = None) -> List[Path]:
    """Return all GQI TSVs for an attempt (or latest), from per-modality subdirs.

    Falls back to legacy combined files in root group_metrics/ for backward
    compatibility.
    """
    files: list = []
    # Per-modality subdirectories (new layout)
    for subdir in ("meg", "eeg"):
        d = group_metrics_dir / subdir
        if d.is_dir():
            files.extend(d.glob("Global_Quality_Index_attempt_*.tsv"))
    # Fallback: legacy combined files in root group_metrics/
    if not files:
        files = list(group_metrics_dir.glob("Global_Quality_Index_attempt_*.tsv"))
    if not files:
        return []

    if attempt is not None:
        files = [f for f in files if _attempt_from_name(f) == attempt]
    else:
        # Keep only files from the latest attempt number
        latest = max(_attempt_from_name(f) or 0 for f in files)
        files = [f for f in files if (_attempt_from_name(f) or 0) == latest]

    return sorted(files, key=lambda p: p.name)


def _resolve_input_paths(
    dataset_path: str,
    derivatives_base: Optional[str],
    input_tsv: Optional[str],
    attempt: Optional[int],
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> Tuple[str, str, Path, Optional[Path], Optional[int], Path]:
    """Resolve derivatives root, selected TSV, matching config, and reports dir."""
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

    # Reports always go to the resolved (external) path.
    reports_dir = Path(megqc_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_root = Path(megqc_root) / "summary_reports"
    group_metrics_dir = summary_root / "group_metrics"
    config_dir = summary_root / "config"

    # Scenario C: if GQI data lives in the original BIDS dataset (calc ran without
    # external path), fall back to reading from there while still writing to the
    # external reports_dir.
    if not group_metrics_dir.exists() and derivatives_base is not None:
        original_megqc_root = os.path.join(
            dataset_path, "derivatives", "Meg_QC", *analysis_segments
        )
        original_summary_root = Path(original_megqc_root) / "summary_reports"
        original_group_metrics_dir = original_summary_root / "group_metrics"
        if original_group_metrics_dir.exists():
            megqc_root = original_megqc_root
            summary_root = original_summary_root
            group_metrics_dir = original_group_metrics_dir
            config_dir = original_summary_root / "config"
            print(
                "___MEGqc___: QC group report: Scenario C — "
                f"reading GQI from original dataset: {group_metrics_dir}"
            )

    if input_tsv:
        tsv_path = Path(input_tsv)
        if not tsv_path.exists():
            raise FileNotFoundError(f"Input TSV does not exist: {tsv_path}")
    else:
        if attempt is not None:
            # Look in per-modality subdirs first, then fall back to root
            tsv_candidates = _all_attempt_tsvs(group_metrics_dir, attempt=attempt)
            if not tsv_candidates:
                # Legacy fallback: try the combined file at root level
                legacy = group_metrics_dir / f"Global_Quality_Index_attempt_{attempt}.tsv"
                if legacy.exists():
                    tsv_candidates = [legacy]
            if not tsv_candidates:
                raise FileNotFoundError(
                    f"Requested attempt {attempt} not found in {group_metrics_dir}"
                )
            tsv_path = tsv_candidates[0]
        else:
            tsv_path = _latest_attempt_tsv(group_metrics_dir)
            if tsv_path is None:
                raise FileNotFoundError(
                    f"No Global_Quality_Index_attempt_*.tsv found in {group_metrics_dir}"
                )

    resolved_attempt = _attempt_from_name(tsv_path)
    cfg_path = None
    if resolved_attempt is not None:
        candidate = config_dir / f"global_quality_index_{resolved_attempt}.ini"
        if candidate.exists():
            cfg_path = candidate

    return derivatives_root, megqc_root, tsv_path, cfg_path, resolved_attempt, reports_dir


def _load_one_dataset_bundle(
    dataset_path: str,
    *,
    derivatives_base: Optional[str],
    input_tsv: Optional[str],
    attempt: Optional[int],
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> QCDatasetBundle:
    (
        derivatives_root,
        megqc_root,
        tsv_path,
        cfg_path,
        resolved_attempt,
        reports_dir,
    ) = _resolve_input_paths(
        dataset_path=dataset_path,
        derivatives_base=derivatives_base,
        input_tsv=input_tsv,
        attempt=attempt,
        analysis_mode=analysis_mode,
        analysis_id=analysis_id,
    )

    # Load all per-modality TSVs for this attempt and concatenate them.
    # _resolve_input_paths returns one representative tsv_path; we also
    # look for sibling modality TSVs from the same attempt.
    summary_root = Path(megqc_root) / "summary_reports"
    group_metrics_dir = summary_root / "group_metrics"
    all_tsvs = _all_attempt_tsvs(group_metrics_dir, attempt=resolved_attempt)
    if not all_tsvs:
        # Fallback: use the single TSV resolved above (legacy combined file)
        all_tsvs = [tsv_path]

    dfs = []
    for tp in all_tsvs:
        df_part = pd.read_csv(tp, sep="\t", low_memory=False)
        # Tag modality from filename or parent directory
        if "modality" not in df_part.columns:
            tname = tp.name.lower()
            tparent = tp.parent.name.lower()
            if "_eeg" in tname or tparent == "eeg":
                df_part["modality"] = "eeg"
            elif "_meg" in tname or tparent == "meg":
                df_part["modality"] = "meg"
            else:
                df_part["modality"] = "unknown"
        dfs.append(df_part)

    df_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    df = _prepare_table(df_raw)
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    df["dataset"] = dataset_name
    df["dataset_path"] = dataset_path

    return QCDatasetBundle(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        derivatives_root=derivatives_root,
        reports_dir=reports_dir,
        tsv_path=tsv_path,
        cfg_path=cfg_path,
        attempt=resolved_attempt,
        df=df,
        general_settings_snapshot=_general_settings_snapshot(megqc_root),
        gqi_settings_snapshot=_config_snapshot(cfg_path),
    )


# ---------------------------------------------------------------------------
# Data adapters
# ---------------------------------------------------------------------------


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "subject" not in out.columns:
        out["subject"] = "n/a"
    if "task" not in out.columns:
        out["task"] = "unknown"

    out["subject"] = out["subject"].fillna("n/a").astype(str)
    out["task"] = out["task"].fillna("unknown").astype(str)
    out["task_label"] = out["task"].astype(str)  # Clean label: just the task name (e.g. 'rest', not 'task=rest')

    # Build recording_label: sub + task [+ run when present] — uniquely identifies a BIDS recording
    if "run" in out.columns:
        out["run"] = out["run"].fillna("n/a").astype(str)
        out["recording_label"] = out.apply(
            lambda r: (
                r["subject"] + "|" + r["task_label"]
                + (f"|run-{r['run']}" if r["run"] not in ("n/a", "nan", "") else "")
            ),
            axis=1,
        )
    else:
        out["recording_label"] = out["subject"] + "|" + out["task_label"]

    if {"Muscle_events_num", "Muscle_events_total"}.issubset(out.columns):
        num = _coerce_numeric(out["Muscle_events_num"])
        den = _coerce_numeric(out["Muscle_events_total"])
        out["__muscle_event_rate_pct"] = np.where(
            np.isfinite(num) & np.isfinite(den) & (den > 0),
            100.0 * num / den,
            np.nan,
        )
    else:
        out["__muscle_event_rate_pct"] = np.nan

    return out


def _series_for_component(df: pd.DataFrame, spec: ComponentSpec, view: str) -> pd.Series:
    """Return one numeric series for a metric component and channel-type view."""

    def _from_col(col: Optional[str]) -> pd.Series:
        if col is None or col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return _coerce_numeric(df[col])

    if view == "mag":
        s = _from_col(spec.mag_col)
        if s.notna().any():
            return s
        if spec.global_col:
            return _from_col(spec.global_col)
        return s

    if view == "grad":
        s = _from_col(spec.grad_col)
        if s.notna().any():
            return s
        if spec.global_col:
            return _from_col(spec.global_col)
        return s

    if view == "eeg":
        s = _from_col(spec.eeg_col)
        if s.notna().any():
            return s
        if spec.global_col:
            return _from_col(spec.global_col)
        return s

    # combined: average mag/grad if possible, else fallback.
    sm = _from_col(spec.mag_col)
    sg = _from_col(spec.grad_col)
    if sm.notna().any() and sg.notna().any():
        both = pd.concat([sm, sg], axis=1)
        return both.mean(axis=1, skipna=True)
    if sm.notna().any():
        return sm
    if sg.notna().any():
        return sg
    if spec.global_col:
        return _from_col(spec.global_col)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _component_frame(df: pd.DataFrame, spec: ComponentSpec, view: str) -> pd.DataFrame:
    vals = _series_for_component(df, spec, view)
    base_cols = ["subject", "task", "task_label", "recording_label"]
    if "dataset" in df.columns:
        base_cols.append("dataset")
    out = df[base_cols].copy()
    out["value"] = vals
    out = out.loc[np.isfinite(out["value"])].copy()
    return out


def _summary_component_frame(df: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    base_cols = ["subject", "task", "task_label", "recording_label"]
    if "dataset" in df.columns:
        base_cols.append("dataset")
    out = df[base_cols].copy()
    out["value"] = pd.to_numeric(values, errors="coerce")
    out = out.loc[np.isfinite(out["value"])].copy()
    return out


def _with_all_tasks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    all_rows = out.copy()
    all_rows["task_label"] = "all tasks"
    return pd.concat([all_rows, out], ignore_index=True)


def _with_all_tasks_by_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Expanded labels for multi-dataset distributions.

    Includes:
    - per-dataset all-tasks distributions
    - per-dataset per-task distributions
    - pooled all-datasets all-tasks distribution
    """
    if df.empty:
        return df
    out = df.copy()
    out["dataset"] = out.get("dataset", "dataset").astype(str)
    out["task_label"] = out.get("task_label", "unknown").astype(str)

    per_task = out.copy()
    per_task["task_label"] = per_task["dataset"] + " | " + per_task["task_label"]
    per_task["__group_kind"] = "dataset_task"
    per_task["__group_dataset"] = per_task["dataset"].astype(str)
    per_task["__group_task"] = per_task["task"].astype(str)

    per_dataset_all = out.copy()
    per_dataset_all["task_label"] = per_dataset_all["dataset"] + " | all tasks"
    per_dataset_all["__group_kind"] = "dataset_all"
    per_dataset_all["__group_dataset"] = per_dataset_all["dataset"].astype(str)
    per_dataset_all["__group_task"] = "all tasks"

    pooled = out.copy()
    pooled["task_label"] = "all datasets | all tasks"
    pooled["__group_kind"] = "pooled"
    pooled["__group_dataset"] = "all datasets"
    pooled["__group_task"] = "all tasks"
    return pd.concat([pooled, per_dataset_all, per_task], ignore_index=True)


def _has_channel_type_data(df: pd.DataFrame, view: str) -> bool:
    """True when a channel-type view has at least one finite metric column."""
    if view not in {"mag", "grad", "eeg"}:
        return True
    cols: List[str] = []
    for specs in COMPONENTS.values():
        for spec in specs:
            if view == "mag":
                col = spec.mag_col
            elif view == "grad":
                col = spec.grad_col
            elif view == "eeg":
                col = spec.eeg_col
            else:
                col = None
            if col:
                cols.append(col)
    for col in cols:
        if col in df.columns and np.isfinite(_coerce_numeric(df[col])).any():
            return True
    return False


def _has_modality(df: pd.DataFrame, modality: str) -> bool:
    """True when the dataframe contains rows belonging to *modality*.

    Uses the ``modality`` column (tagged from TSV filename / parent directory)
    as the authoritative source so that a MEG file which happens to contain a
    few EEG channels is *not* mistaken for an EEG recording.

    Falls back to :func:`_has_channel_type_data` only for legacy combined TSVs
    that have no ``modality`` column or where every row is ``"unknown"``.
    """
    if "modality" in df.columns:
        known = df.loc[df["modality"].isin({"meg", "eeg", "ieeg"})]
        if not known.empty:
            return bool(known["modality"].eq(modality).any())
    # Legacy / unknown fallback
    if modality == "meg":
        return _has_channel_type_data(df, "mag") or _has_channel_type_data(df, "grad")
    if modality == "eeg":
        return _has_channel_type_data(df, "eeg")
    return False


def _filter_by_modality(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """Return only rows belonging to *modality*, or the full df for legacy data.

    When the ``modality`` column exists and has at least one non-unknown entry
    the df is filtered to the requested modality.  This prevents MEG rows from
    polluting an EEG report (and vice-versa) even when both modalities are
    present in a concatenated dataframe.
    """
    if "modality" in df.columns:
        known = df.loc[df["modality"].isin({"meg", "eeg", "ieeg"})]
        if not known.empty:
            filtered = df.loc[df["modality"] == modality]
            return filtered if not filtered.empty else df
    return df


def _group_order(df: pd.DataFrame) -> List[str]:
    if "__group_kind" in df.columns:
        keys = (
            df[["task_label", "__group_kind", "__group_dataset", "__group_task"]]
            .drop_duplicates()
            .copy()
        )
        rank = {"pooled": 0, "dataset_all": 1, "dataset_task": 2}
        keys["__rank"] = keys["__group_kind"].map(rank).fillna(99).astype(int)
        keys = keys.sort_values(
            by=["__rank", "__group_dataset", "__group_task", "task_label"],
            kind="stable",
        )
        return keys["task_label"].astype(str).tolist()

    groups = sorted(set(df["task_label"].astype(str).tolist()))
    if "all tasks" in groups:
        groups.remove("all tasks")
        return ["all tasks", *groups]
    return groups


def _palette(labels: Sequence[str]) -> Dict[str, str]:
    base = [
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
        "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#2CA02C", "#D62728",
    ]
    return {label: base[i % len(base)] for i, label in enumerate(labels)}


def _summary_distribution_components(df: pd.DataFrame, view: str) -> List[Tuple[str, str, pd.DataFrame]]:
    """Return summary component dataframes (one recording = one row)."""
    gqi = _summary_component_frame(df, _series_for_component(df, COMPONENTS["GQI"][0], view))
    std_noisy = _series_for_component(df, COMPONENTS["STD"][0], view)
    std_flat = _series_for_component(df, COMPONENTS["STD"][1], view)
    std_vals = pd.Series(np.nanmean(np.column_stack([std_noisy.to_numpy(dtype=float), std_flat.to_numpy(dtype=float)]), axis=1), index=df.index)
    std = _summary_component_frame(df, std_vals)

    ptp_noisy = _series_for_component(df, COMPONENTS["PtP"][0], view)
    ptp_flat = _series_for_component(df, COMPONENTS["PtP"][1], view)
    ptp_vals = pd.Series(np.nanmean(np.column_stack([ptp_noisy.to_numpy(dtype=float), ptp_flat.to_numpy(dtype=float)]), axis=1), index=df.index)
    ptp = _summary_component_frame(df, ptp_vals)

    psd = _summary_component_frame(df, _series_for_component(df, COMPONENTS["PSD"][0], view))
    ecg = _summary_component_frame(df, _series_for_component(df, COMPONENTS["ECG"][0], view))
    eog = _summary_component_frame(df, _series_for_component(df, COMPONENTS["EOG"][0], view))
    muscle = _summary_component_frame(df, _series_for_component(df, COMPONENTS["Muscle"][0], view))

    return [
        ("GQI", "GQI (%)", gqi),
        ("STD", "STD pooled noisy/flat channels (% channels)", std),
        ("PtP", "PtP pooled noisy/flat channels (% channels)", ptp),
        ("PSD", "PSD noise burden (% relative power)", psd),
        ("ECG", "ECG high-correlation channels (% channels)", ecg),
        ("EOG", "EOG high-correlation channels (% channels)", eog),
        ("Muscle", "Muscle events (count)", muscle),
    ]


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def _gaussian_kde_line(values: np.ndarray, n_points: int = 240) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmax <= vmin:
        vmax = vmin + 1.0

    std = float(np.nanstd(vals))
    bw = 1.06 * std * (vals.size ** (-1.0 / 5.0)) if std > 0 else 0.3
    bw = max(float(bw), max(abs(vmin), abs(vmax), 1.0) * 1e-3)

    x = np.linspace(vmin, vmax, n_points)
    diff = (x[:, None] - vals[None, :]) / bw
    y = np.exp(-0.5 * diff ** 2).sum(axis=1) / (vals.size * bw * np.sqrt(2.0 * np.pi))
    return x, y


def _hex_to_rgba(color: str, alpha: float) -> str:
    c = str(color).strip()
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{float(alpha):.3f})"
    return c


def _attach_distribution_style_controls(
    fig: go.Figure,
    *,
    default_line_width: float = 2.2,
    default_marker_size: float = 8.0,
) -> None:
    """Attach numeric style controls for distribution plots."""
    if fig is None or not fig.data:
        return

    marker_level_to_size = {1: 3.0, 2: 4.0, 3: 5.5, 4: 7.0, 5: 8.5, 6: 10.0, 7: 12.0, 8: 14.0}

    has_marker_traces = False
    for tr in fig.data:
        t = getattr(tr, "type", "")
        mode = str(getattr(tr, "mode", "") or "")
        if t in {"scatter", "scattergl"} and ("markers" in mode):
            has_marker_traces = True
            break

    def _line_restyle(level: int) -> Dict[str, List[Optional[float]]]:
        line_width: List[Optional[float]] = []
        marker_line_width: List[Optional[float]] = []
        lw = float(level)
        for tr in fig.data:
            t = getattr(tr, "type", "")
            mode = str(getattr(tr, "mode", "") or "")
            if t in {"scatter", "scattergl", "violin", "box"}:
                line_width.append(lw)
            else:
                line_width.append(None)
            if t == "histogram":
                marker_line_width.append(max(0.5, lw * 0.6))
            elif t in {"scatter", "scattergl"} and ("markers" in mode):
                marker_line_width.append(max(0.25, lw * 0.2))
            else:
                marker_line_width.append(None)
        return {
            "line.width": line_width,
            "marker.line.width": marker_line_width,
        }

    def _marker_restyle(level: int) -> Dict[str, List[Optional[float]]]:
        marker_size: List[Optional[float]] = []
        size = float(marker_level_to_size.get(level, default_marker_size))
        for tr in fig.data:
            t = getattr(tr, "type", "")
            mode = str(getattr(tr, "mode", "") or "")
            if t in {"scatter", "scattergl"} and ("markers" in mode):
                marker_size.append(size)
            else:
                marker_size.append(None)
        return {"marker.size": marker_size}

    levels = list(range(1, 9))
    line_active = int(np.clip(int(round(default_line_width)), 1, 8) - 1)
    marker_active_guess = min(levels, key=lambda lv: abs(marker_level_to_size[lv] - float(default_marker_size)))
    marker_active = int(marker_active_guess - 1)

    line_buttons = [
        dict(
            label=str(level),
            method="restyle",
            args=[_line_restyle(level)],
        )
        for level in levels
    ]
    marker_buttons = []
    if has_marker_traces:
        marker_buttons = [
            dict(
                label=str(level),
                method="restyle",
                args=[_marker_restyle(level)],
            )
            for level in levels
        ]

    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    menu_names = {str(getattr(m, "name", "") or "") for m in existing}
    if "Line thickness level" not in menu_names:
        existing.append(
            dict(
                name="Line thickness level",
                type="buttons",
                direction="right",
                showactive=True,
                active=line_active,
                buttons=line_buttons,
            )
        )
    if marker_buttons and ("Dot size level" not in menu_names):
        existing.append(
            dict(
                name="Dot size level",
                type="buttons",
                direction="right",
                showactive=True,
                active=marker_active,
                buttons=marker_buttons,
            )
        )
    fig.update_layout(updatemenus=existing)


def _attach_dot_alignment_controller(
    fig: go.Figure,
    *,
    centered_trace_indices: Sequence[int],
    side_trace_indices: Sequence[int],
    default_side: bool = False,
) -> None:
    """Attach a controller to toggle subject dots between centered and side placement."""
    if fig is None or not fig.data:
        return
    centered = [int(i) for i in centered_trace_indices]
    side = [int(i) for i in side_trace_indices]
    if not centered or not side:
        return

    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    menu_names = {str(getattr(m, "name", "") or "") for m in existing}
    if "Dot placement" in menu_names:
        return

    target_indices = centered + side
    centered_visible = [True] * len(centered) + [False] * len(side)
    side_visible = [False] * len(centered) + [True] * len(side)

    existing.append(
        dict(
            name="Dot placement",
            type="buttons",
            direction="right",
            showactive=True,
            active=1 if default_side else 0,
            buttons=[
                dict(label="Centered dots", method="restyle", args=[{"visible": centered_visible}, target_indices]),
                dict(label="Side dots", method="restyle", args=[{"visible": side_visible}, target_indices]),
            ],
        )
    )
    fig.update_layout(updatemenus=existing)


def _attach_side_displacement_controller(
    fig: go.Figure,
    *,
    side_trace_indices: Sequence[int],
    side_x_by_level: Dict[int, List[np.ndarray]],
    default_level: int = 4,
) -> None:
    """Attach a controller to vary side-dot horizontal displacement (levels 1..8)."""
    if fig is None or not fig.data:
        return
    side_indices = [int(i) for i in side_trace_indices]
    if not side_indices:
        return

    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    menu_names = {str(getattr(m, "name", "") or "") for m in existing}
    if "Side displacement level" in menu_names:
        return

    levels = [lv for lv in range(1, 9) if lv in side_x_by_level and len(side_x_by_level[lv]) == len(side_indices)]
    if not levels:
        return
    active = int(np.clip(int(default_level), min(levels), max(levels))) - 1
    active = max(0, min(active, len(levels) - 1))

    existing.append(
        dict(
            name="Side displacement level",
            type="buttons",
            direction="right",
            showactive=True,
            active=active,
            buttons=[
                dict(
                    label=str(level),
                    method="restyle",
                    args=[{"x": side_x_by_level[level]}, side_indices],
                )
                for level in levels
            ],
        )
    )
    fig.update_layout(updatemenus=existing)


def _attach_axis_label_size_controller(fig: go.Figure) -> None:
    """Add a numeric axis-label/ticks-size controller for any figure with axes."""
    if fig is None:
        return
    layout_dict = fig.to_plotly_json().get("layout", {})
    axis_keys = [k for k in layout_dict.keys() if str(k).startswith("xaxis") or str(k).startswith("yaxis")]
    scene_keys = [k for k in layout_dict.keys() if re.fullmatch(r"scene\d*", str(k) or "")]
    if (not axis_keys) and (not scene_keys):
        return

    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    if any(str(getattr(m, "name", "") or "") == "Axis label/ticks size level" for m in existing):
        return

    axis_level_to_tick = {1: 9, 2: 10, 3: 12, 4: 14, 5: 16, 6: 18, 7: 20, 8: 22}

    def _axis_relayout(level: int) -> Dict[str, int]:
        tick_sz = int(axis_level_to_tick.get(level, 14))
        title_sz = tick_sz + 2
        payload: Dict[str, int] = {}
        for key in axis_keys:
            payload[f"{key}.tickfont.size"] = tick_sz
            payload[f"{key}.title.font.size"] = title_sz
        for sk in scene_keys:
            payload[f"{sk}.xaxis.tickfont.size"] = tick_sz
            payload[f"{sk}.yaxis.tickfont.size"] = tick_sz
            payload[f"{sk}.zaxis.tickfont.size"] = tick_sz
            payload[f"{sk}.xaxis.title.font.size"] = title_sz
            payload[f"{sk}.yaxis.title.font.size"] = title_sz
            payload[f"{sk}.zaxis.title.font.size"] = title_sz
        return payload

    existing.append(
        dict(
            name="Axis label/ticks size level",
            type="buttons",
            direction="right",
            showactive=True,
            active=3,
            buttons=[dict(label=str(level), method="relayout", args=[_axis_relayout(level)]) for level in range(1, 9)],
        )
    )
    fig.update_layout(updatemenus=existing)


def _infer_updatemenu_title(menu, idx: int) -> str:
    explicit = str(getattr(menu, "name", "") or "").strip()
    if explicit:
        return explicit
    buttons = list(getattr(menu, "buttons", []) or [])
    labels: List[str] = [str(getattr(btn, "label", "") or "").strip() for btn in buttons]
    low = " ".join(l.lower() for l in labels)
    if "heat:" in low:
        return "Heat summary variant"
    if "top:" in low:
        return "Top profile summary"
    if "right:" in low:
        return "Right profile summary"
    if "cap" in low:
        return "3D cap display"
    if labels and all(l in {str(i) for i in range(1, 9)} for l in labels):
        methods = {str(getattr(btn, "method", "") or "").lower() for btn in buttons}
        arg_keys: set[str] = set()
        for btn in buttons:
            args = getattr(btn, "args", None)
            if isinstance(args, (list, tuple)) and args:
                first = args[0]
                if isinstance(first, dict):
                    arg_keys.update(str(k) for k in first.keys())
        if "relayout" in methods:
            return "Axis label/ticks size level"
        if "marker.size" in arg_keys:
            return "Dot size level"
        if "line.width" in arg_keys:
            return "Line thickness level"
        return "Style level"
    return f"Figure control {idx + 1}"


def _encapsulate_updatemenus_panel(fig: go.Figure) -> None:
    """Place all Plotly controllers in a dedicated bottom panel to avoid overlaps."""
    if fig is None:
        return
    menus = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    if not menus:
        return
    menu_ys: List[float] = []
    all_have_y = True
    for m in menus:
        yv = getattr(m, "y", None)
        if yv is None:
            all_have_y = False
            break
        try:
            menu_ys.append(float(yv))
        except Exception:
            all_have_y = False
            break
    if all_have_y and menu_ys and max(menu_ys) < 0:
        return

    start_y = -0.17
    step_y = 0.16
    new_menus = []
    title_annotations = []
    for idx, menu in enumerate(menus):
        mj = menu.to_plotly_json() if hasattr(menu, "to_plotly_json") else dict(menu)
        y = start_y - idx * step_y
        mj["x"] = 0.03
        mj["y"] = y
        mj["xanchor"] = "left"
        mj["yanchor"] = "top"
        mj["direction"] = "right"
        mj["showactive"] = True if mj.get("showactive") is None else mj["showactive"]
        mj.setdefault("bgcolor", "rgba(231,241,252,0.96)")
        mj.setdefault("bordercolor", "#6EA2D5")
        mj.setdefault("borderwidth", 1)
        mj.setdefault("pad", {"r": 8, "t": 2, "l": 2, "b": 2})
        new_menus.append(mj)

        title = _infer_updatemenu_title(menu, idx)
        title_annotations.append(
            dict(
                x=0.03,
                y=y + 0.038,
                xref="paper",
                yref="paper",
                text=f"<b>{title}</b>",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font={"size": 12, "color": "#17395C"},
                align="left",
            )
        )

    anns = list(fig.layout.annotations) if fig.layout.annotations else []
    anns.extend(title_annotations)

    y_top = -0.06
    y_bottom = start_y - (len(new_menus) - 1) * step_y - 0.10
    shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    shapes.append(
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.0,
            x1=1.0,
            y0=y_bottom,
            y1=y_top,
            line={"color": "#9FC2E8", "width": 1},
            fillcolor="rgba(238,246,255,0.92)",
            layer="below",
        )
    )

    margin = fig.layout.margin or {}
    current_bottom = int(getattr(margin, "b", 60) or 60)
    needed_bottom = 140 + int(len(new_menus) * 92)
    fig.update_layout(
        updatemenus=new_menus,
        annotations=anns,
        shapes=shapes,
        margin={
            "l": margin.l if hasattr(margin, "l") and margin.l is not None else 55,
            "r": margin.r if hasattr(margin, "r") and margin.r is not None else 30,
            "t": margin.t if hasattr(margin, "t") and margin.t is not None else 120,
            "b": max(current_bottom, needed_bottom),
        },
    )


def plot_violin_with_subject_points(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    multi_dataset_mode: bool = False,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)
    labels = [label for label in groups if np.isfinite(expanded.loc[expanded["task_label"] == label, "value"]).any()]
    if not labels:
        return None
    xpos = {label: float(i) for i, label in enumerate(labels)}

    fig = go.Figure()
    for group in labels:
        g = expanded.loc[expanded["task_label"] == group].copy()
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        group_id = f"group::{group}"
        fig.add_trace(
            go.Violin(
                x=np.full(vals.size, xpos[group], dtype=float),
                y=vals,
                name=f"{group} (n={vals.size})",
                legendgroup=group_id,
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"color": colors[group], "width": 2.0},
                fillcolor=colors[group],
                opacity=0.38,
                width=0.82,
                spanmode="soft",
                hovertemplate="group=" + group + "<br>value=%{y:.3g}<extra></extra>",
            )
        )
        # Add explicit box/whisker overlay so min/max/median are clearly visible
        # in the violin view, matching the interpretability of the boxplot panel.
        fig.add_trace(
            go.Box(
                x=np.full(vals.size, xpos[group], dtype=float),
                y=vals,
                name=f"{group} box",
                legendgroup=group_id,
                showlegend=False,
                boxpoints=False,
                line={"color": colors[group], "width": 2.0},
                marker={"color": colors[group]},
                fillcolor="rgba(0,0,0,0)",
                opacity=1.0,
                whiskerwidth=0.9,
                width=0.20,
                hovertemplate="group=" + group + "<br>value=%{y:.3g}<extra></extra>",
            )
        )

    # Subject dots: one robust value per subject per task group, with independent subject colors.
    subject_rows: List[Tuple[str, float, str, str]] = []
    tmp = expanded.loc[np.isfinite(pd.to_numeric(expanded["value"], errors="coerce"))].copy()
    tmp["subject"] = tmp["subject"].fillna("n/a").astype(str)
    if "dataset" in tmp.columns:
        tmp["dataset"] = tmp["dataset"].fillna("dataset").astype(str)
    else:
        tmp["dataset"] = "dataset"
    tmp["task"] = tmp["task"].fillna("unknown").astype(str)
    tmp["recording_label"] = tmp["recording_label"].fillna("n/a").astype(str)
    for (group, dataset, subject), grp in tmp.groupby(["task_label", "dataset", "subject"], dropna=False):
        vals = pd.to_numeric(grp["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or group not in xpos:
            continue
        value = float(np.nanmedian(vals))
        runs = sorted(set(grp["recording_label"].astype(str).tolist()))
        subject_rows.append(
            (
                str(group),
                value,
                f"{dataset}|{subject}",
                f"dataset={dataset}<br>subject={subject}<br>task_group={group}<br>n_rows={len(grp)}"
                + (f"<br>records={', '.join(runs[:4])}" + ("..." if len(runs) > 4 else "") if runs else "")
                + f"<br>value={value:.3g}",
            )
        )
    points_df = pd.DataFrame(subject_rows, columns=["label", "value", "subject", "hover"])
    centered_trace_indices: List[int] = []
    side_trace_indices: List[int] = []
    side_x_by_level: Dict[int, List[np.ndarray]] = {level: [] for level in range(1, 9)}
    displacement_map = {1: 0.10, 2: 0.14, 3: 0.18, 4: 0.22, 5: 0.26, 6: 0.30, 7: 0.34, 8: 0.38}
    default_side_level = 4
    if not points_df.empty:
        points_df["subj_code"] = pd.Categorical(points_df["subject"]).codes.astype(float)
        rng = np.random.default_rng(7)
        for label in labels:
            gp = points_df.loc[points_df["label"] == label].copy()
            if gp.empty:
                continue
            x_center = np.full(gp.shape[0], xpos[label], dtype=float)
            x_center = x_center + rng.uniform(-0.08, 0.08, size=x_center.size)
            side_jitter = rng.uniform(-0.06, 0.06, size=x_center.size)
            x_side_levels = {
                level: (np.full(gp.shape[0], xpos[label] + offset, dtype=float) + side_jitter)
                for level, offset in displacement_map.items()
            }
            fig.add_trace(
                go.Scattergl(
                    x=x_center,
                    y=gp["value"].to_numpy(dtype=float),
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": gp["subj_code"].to_numpy(dtype=float),
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "line": {"width": 0.4, "color": "rgba(20,20,20,0.55)"},
                        "showscale": False,
                    },
                    customdata=np.stack([gp["hover"].to_numpy()], axis=-1),
                    hovertemplate="%{customdata[0]}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=True,
                )
            )
            centered_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scattergl(
                    x=x_side_levels[default_side_level],
                    y=gp["value"].to_numpy(dtype=float),
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": gp["subj_code"].to_numpy(dtype=float),
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "line": {"width": 0.4, "color": "rgba(20,20,20,0.55)"},
                        "showscale": False,
                    },
                    customdata=np.stack([gp["hover"].to_numpy()], axis=-1),
                    hovertemplate="%{customdata[0]}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=False,
                )
            )
            side_trace_indices.append(len(fig.data) - 1)
            for level in range(1, 9):
                side_x_by_level[level].append(x_side_levels[level])

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task",
        yaxis_title=y_label,
        **_adaptive_legend_layout(len(labels), groupclick="togglegroup"),
        violinmode="group",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.2 + (0.50 if side_trace_indices else 0.0)],
    )
    _attach_distribution_style_controls(fig, default_line_width=2.2, default_marker_size=8.0)
    for idx in centered_trace_indices:
        if 0 <= idx < len(fig.data):
            fig.data[idx].visible = True
    for idx in side_trace_indices:
        if 0 <= idx < len(fig.data):
            fig.data[idx].visible = False
    _attach_dot_alignment_controller(
        fig,
        centered_trace_indices=centered_trace_indices,
        side_trace_indices=side_trace_indices,
        default_side=False,
    )
    _attach_side_displacement_controller(
        fig,
        side_trace_indices=side_trace_indices,
        side_x_by_level=side_x_by_level,
        default_level=default_side_level,
    )
    return fig


def plot_box_with_subject_points(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    multi_dataset_mode: bool = False,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)
    labels = [label for label in groups if np.isfinite(expanded.loc[expanded["task_label"] == label, "value"]).any()]
    if not labels:
        return None
    xpos = {label: float(i) for i, label in enumerate(labels)}

    fig = go.Figure()
    for group in labels:
        g = expanded.loc[expanded["task_label"] == group].copy()
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        group_id = f"group::{group}"
        color = colors[group]
        fig.add_trace(
            go.Box(
                x=np.full(vals.size, xpos[group], dtype=float),
                y=vals,
                name=f"{group} (n={vals.size})",
                legendgroup=group_id,
                boxpoints=False,
                line={"color": color, "width": 2.0},
                marker={"color": color},
                fillcolor=_hex_to_rgba(color, 0.26),
                opacity=0.9,
                hovertemplate="group=" + group + "<br>value=%{y:.3g}<extra></extra>",
            )
        )

    subject_rows: List[Tuple[str, float, str, str]] = []
    tmp = expanded.loc[np.isfinite(pd.to_numeric(expanded["value"], errors="coerce"))].copy()
    tmp["subject"] = tmp["subject"].fillna("n/a").astype(str)
    if "dataset" in tmp.columns:
        tmp["dataset"] = tmp["dataset"].fillna("dataset").astype(str)
    else:
        tmp["dataset"] = "dataset"
    tmp["task"] = tmp["task"].fillna("unknown").astype(str)
    tmp["recording_label"] = tmp["recording_label"].fillna("n/a").astype(str)
    for (group, dataset, subject), grp in tmp.groupby(["task_label", "dataset", "subject"], dropna=False):
        vals = pd.to_numeric(grp["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0 or group not in xpos:
            continue
        value = float(np.nanmedian(vals))
        runs = sorted(set(grp["recording_label"].astype(str).tolist()))
        subject_rows.append(
            (
                str(group),
                value,
                f"{dataset}|{subject}",
                f"dataset={dataset}<br>subject={subject}<br>task_group={group}<br>n_rows={len(grp)}"
                + (f"<br>records={', '.join(runs[:4])}" + ("..." if len(runs) > 4 else "") if runs else "")
                + f"<br>value={value:.3g}",
            )
        )
    points_df = pd.DataFrame(subject_rows, columns=["label", "value", "subject", "hover"])
    centered_trace_indices: List[int] = []
    side_trace_indices: List[int] = []
    side_x_by_level: Dict[int, List[np.ndarray]] = {level: [] for level in range(1, 9)}
    displacement_map = {1: 0.10, 2: 0.14, 3: 0.18, 4: 0.22, 5: 0.26, 6: 0.30, 7: 0.34, 8: 0.38}
    default_side_level = 4
    if not points_df.empty:
        points_df["subj_code"] = pd.Categorical(points_df["subject"]).codes.astype(float)
        rng = np.random.default_rng(7)
        for label in labels:
            gp = points_df.loc[points_df["label"] == label].copy()
            if gp.empty:
                continue
            x_center = np.full(gp.shape[0], xpos[label], dtype=float)
            x_center = x_center + rng.uniform(-0.08, 0.08, size=x_center.size)
            side_jitter = rng.uniform(-0.06, 0.06, size=x_center.size)
            x_side_levels = {
                level: (np.full(gp.shape[0], xpos[label] + offset, dtype=float) + side_jitter)
                for level, offset in displacement_map.items()
            }
            fig.add_trace(
                go.Scattergl(
                    x=x_center,
                    y=gp["value"].to_numpy(dtype=float),
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": gp["subj_code"].to_numpy(dtype=float),
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "line": {"width": 0.4, "color": "rgba(20,20,20,0.55)"},
                        "showscale": False,
                    },
                    customdata=np.stack([gp["hover"].to_numpy()], axis=-1),
                    hovertemplate="%{customdata[0]}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=True,
                )
            )
            centered_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scattergl(
                    x=x_side_levels[default_side_level],
                    y=gp["value"].to_numpy(dtype=float),
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": gp["subj_code"].to_numpy(dtype=float),
                        "colorscale": "Turbo",
                        "opacity": 0.78,
                        "line": {"width": 0.4, "color": "rgba(20,20,20,0.55)"},
                        "showscale": False,
                    },
                    customdata=np.stack([gp["hover"].to_numpy()], axis=-1),
                    hovertemplate="%{customdata[0]}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=False,
                )
            )
            side_trace_indices.append(len(fig.data) - 1)
            for level in range(1, 9):
                side_x_by_level[level].append(x_side_levels[level])

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task",
        yaxis_title=y_label,
        **_adaptive_legend_layout(len(labels), groupclick="togglegroup"),
        boxmode="overlay",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.2 + (0.50 if side_trace_indices else 0.0)],
    )
    _attach_distribution_style_controls(fig, default_line_width=2.2, default_marker_size=8.0)
    for idx in centered_trace_indices:
        if 0 <= idx < len(fig.data):
            fig.data[idx].visible = True
    for idx in side_trace_indices:
        if 0 <= idx < len(fig.data):
            fig.data[idx].visible = False
    _attach_dot_alignment_controller(
        fig,
        centered_trace_indices=centered_trace_indices,
        side_trace_indices=side_trace_indices,
        default_side=False,
    )
    _attach_side_displacement_controller(
        fig,
        side_trace_indices=side_trace_indices,
        side_x_by_level=side_x_by_level,
        default_level=default_side_level,
    )
    return fig


def plot_histogram_by_task(df: pd.DataFrame, title: str, x_label: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    multi_dataset_mode = "dataset" in df.columns and (df["dataset"].astype(str).nunique() > 1)
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)

    fig = go.Figure()
    for group in groups:
        g = expanded.loc[expanded["task_label"] == group]
        vals = g["value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        group_id = f"group::{group}"
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=f"{group} (n={vals.size})",
                legendgroup=group_id,
                marker={
                    "color": _hex_to_rgba(colors[group], 0.34),
                    "line": {"width": 1.2, "color": colors[group]},
                },
                opacity=1.0,
                histnorm="probability density",
                hovertemplate="group=" + group + "<br>value=%{x:.3g}<br>density=%{y:.3g}<extra></extra>",
            )
        )
        kde = _gaussian_kde_line(vals)
        if kde is not None:
            kx, ky = kde
            fig.add_trace(
                go.Scatter(
                    x=kx,
                    y=ky,
                    mode="lines",
                    name=f"{group} density",
                    legendgroup=group_id,
                    showlegend=False,
                    line={"color": colors[group], "width": 2.8},
                    hovertemplate="group=" + group + "<br>value=%{x:.3g}<br>density=%{y:.3g}<extra></extra>",
                )
            )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title="Density",
        barmode="overlay",
        **_adaptive_legend_layout(len(groups), groupclick="togglegroup"),
    )
    _attach_distribution_style_controls(fig, default_line_width=2.8, default_marker_size=8.0)
    return fig


def plot_density_by_task(df: pd.DataFrame, title: str, x_label: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    multi_dataset_mode = "dataset" in df.columns and (df["dataset"].astype(str).nunique() > 1)
    expanded = _with_all_tasks_by_dataset(df) if multi_dataset_mode else _with_all_tasks(df)
    groups = _group_order(expanded)
    colors = _palette(groups)

    fig = go.Figure()
    for group in groups:
        vals = expanded.loc[expanded["task_label"] == group, "value"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        kde = _gaussian_kde_line(vals)
        if kde is None:
            continue
        x, y = kde
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{group} (n={vals.size})",
                legendgroup=group,
                line={"color": colors[group], "width": 2.8},
                hovertemplate="group=" + group + "<br>value=%{x:.3g}<br>density=%{y:.3g}<extra></extra>",
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title=x_label,
        yaxis_title="Kernel density",
        **_adaptive_legend_layout(len(fig.data), groupclick="togglegroup"),
    )
    _attach_distribution_style_controls(fig, default_line_width=2.8, default_marker_size=8.0)
    return fig


def _plot_summary_distribution_recordings_qc(
    df_value: pd.DataFrame,
    *,
    title: str,
    y_label: str,
    color: str = "#2A6FBB",
    enable_style_controls: bool = True,
    show_counts_annotation: bool = True,
) -> Optional[go.Figure]:
    """One pooled box+violin where each dot is one recording row."""
    if df_value.empty or "value" not in df_value.columns:
        return None
    vals = pd.to_numeric(df_value["value"], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    data = df_value.loc[np.isfinite(pd.to_numeric(df_value["value"], errors="coerce"))].copy()

    x0 = np.zeros(vals.size, dtype=float)
    rng = np.random.default_rng(23)
    xj = x0 + rng.uniform(-0.10, 0.10, size=vals.size)
    subj_keys = (
        data.get("dataset", pd.Series(["dataset"] * len(data))).astype(str)
        + "::"
        + data.get("subject", pd.Series(["n/a"] * len(data))).astype(str)
    )
    subj_codes = pd.Categorical(subj_keys).codes.astype(float)
    hover = (
        "dataset=" + data.get("dataset", pd.Series(["dataset"] * len(data))).astype(str)
        + "<br>subject=" + data.get("subject", pd.Series(["n/a"] * len(data))).astype(str)
        + "<br>task=" + data.get("task", pd.Series(["unknown"] * len(data))).astype(str)
        + "<br>recording=" + data.get("recording_label", pd.Series(["n/a"] * len(data))).astype(str)
        + "<br>value=" + pd.to_numeric(data["value"], errors="coerce").map(lambda v: f"{float(v):.3g}" if np.isfinite(v) else "n/a")
    ).to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=x0,
            y=vals,
            name=f"all recordings (n={vals.size})",
            box_visible=False,
            meanline_visible=False,
            points=False,
            line={"width": 2.2, "color": color},
            fillcolor=_hex_to_rgba(color, 0.22),
            opacity=0.70,
            width=0.78,
            spanmode="hard",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Box(
            x=x0,
            y=vals,
            name="box",
            boxpoints=False,
            line={"width": 2.2, "color": color},
            marker={"color": color},
            fillcolor="rgba(0,0,0,0)",
            opacity=1.0,
            whiskerwidth=0.95,
            width=0.22,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=xj,
            y=vals,
            mode="markers",
            marker={
                "size": 8.0,
                "color": subj_codes,
                "colorscale": "Turbo",
                "opacity": 0.76,
                "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                "showscale": False,
            },
            customdata=np.stack([hover], axis=-1),
            hovertemplate="%{customdata[0]}<extra></extra>",
            name="recordings",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Recordings",
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 50, "r": 12, "t": 72, "b": 48},
        height=410,
    )
    fig.update_xaxes(tickmode="array", tickvals=[0.0], ticktext=["all recordings"], range=[-0.55, 0.55])
    if show_counts_annotation:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            text=f"N={vals.size} recordings",
            font={"size": 11, "color": "#2f4a68"},
        )
    if enable_style_controls:
        _attach_distribution_style_controls(fig, default_line_width=2.2, default_marker_size=8.0)
    return fig


def _summary_shared_controls_panel_html(summary_group_id: str) -> str:
    """Single shared controller panel for the summary-strip QC plots."""

    def _buttons(kind: str, *, n: int = 8, active_level: int = 4) -> str:
        return "".join(
            f"<button class='plot-control-btn{' active' if i == active_level else ''}' "
            f"data-summary-kind='{kind}' data-level='{i}'>{i}</button>"
            for i in range(1, n + 1)
        )

    return (
        f"<div class='summary-shared-controls' data-summary-group='{summary_group_id}'>"
        "<div class='plot-control-group'>"
        "<div class='plot-control-title'>Line thickness level</div>"
        f"<div class='plot-control-row'>{_buttons('line')}</div>"
        "</div>"
        "<div class='plot-control-group'>"
        "<div class='plot-control-title'>Dot size level</div>"
        f"<div class='plot-control-row'>{_buttons('dot')}</div>"
        "</div>"
        "<div class='plot-control-group'>"
        "<div class='plot-control-title'>Axis label/ticks size level</div>"
        f"<div class='plot-control-row'>{_buttons('axis')}</div>"
        "</div>"
        "<div class='plot-control-group'>"
        "<div class='plot-control-title'>Dot visibility</div>"
        "<div class='plot-control-row'>"
        "<button class='plot-control-btn active' data-summary-kind='dotvis' data-level='1'>Show dots</button>"
        "<button class='plot-control-btn' data-summary-kind='dotvis' data-level='0'>Hide dots</button>"
        "</div>"
        "</div>"
        "<div class='plot-control-group'>"
        "<div class='plot-control-title'>Dot displacement level</div>"
        f"<div class='plot-control-row'>{_buttons('disp', n=12, active_level=1)}</div>"
        "</div>"
        "</div>"
    )


def _summary_distribution_grid_html(
    df_view: pd.DataFrame,
    *,
    view: str,
    view_label: str,
    summary_group_id: str,
) -> str:
    def _median_positive(series: pd.Series) -> Optional[int]:
        vals = _coerce_numeric(series).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return None
        return int(round(float(np.nanmedian(vals))))

    def _estimate_from_num_pct(num_col: str, pct_col: str) -> Optional[int]:
        if num_col not in df_view.columns or pct_col not in df_view.columns:
            return None
        num = _coerce_numeric(df_view[num_col]).to_numpy(dtype=float)
        pct = _coerce_numeric(df_view[pct_col]).to_numpy(dtype=float)
        mask = np.isfinite(num) & np.isfinite(pct) & (pct > 0)
        if not np.any(mask):
            return None
        est = num[mask] / (pct[mask] / 100.0)
        est = est[np.isfinite(est) & (est > 0)]
        if est.size == 0:
            return None
        return int(round(float(np.nanmedian(est))))

    def _row_channel_total(idx: int) -> Optional[int]:
        row = df_view.loc[idx]

        def _row_val(col: str) -> Optional[float]:
            if col not in df_view.columns:
                return None
            v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            return float(v) if np.isfinite(v) else None

        def _num_pct(num_col: str, pct_col: str) -> Optional[float]:
            num = _row_val(num_col)
            pct = _row_val(pct_col)
            if num is None or pct is None or pct <= 0:
                return None
            est = num / (pct / 100.0)
            return float(est) if np.isfinite(est) and est > 0 else None

        if view == "mag":
            candidates = [
                _row_val("ECG_mag_total_channels"),
                _row_val("EOG_mag_total_channels"),
                _num_pct("STD_ts_noisy_channels_mag_num", "STD_ts_noisy_channels_mag_percentage"),
                _num_pct("PTP_ts_noisy_channels_mag_num", "PTP_ts_noisy_channels_mag_percentage"),
                _num_pct("STD_ts_flat_channels_mag_num", "STD_ts_flat_channels_mag_percentage"),
                _num_pct("PTP_ts_flat_channels_mag_num", "PTP_ts_flat_channels_mag_percentage"),
            ]
            c = next((x for x in candidates if x is not None), None)
            return int(round(c)) if c is not None else None
        if view == "grad":
            candidates = [
                _row_val("ECG_grad_total_channels"),
                _row_val("EOG_grad_total_channels"),
                _num_pct("STD_ts_noisy_channels_grad_num", "STD_ts_noisy_channels_grad_percentage"),
                _num_pct("PTP_ts_noisy_channels_grad_num", "PTP_ts_noisy_channels_grad_percentage"),
                _num_pct("STD_ts_flat_channels_grad_num", "STD_ts_flat_channels_grad_percentage"),
                _num_pct("PTP_ts_flat_channels_grad_num", "PTP_ts_flat_channels_grad_percentage"),
            ]
            c = next((x for x in candidates if x is not None), None)
            return int(round(c)) if c is not None else None
        if view == "eeg":
            candidates = [
                _row_val("ECG_eeg_total_channels"),
                _row_val("EOG_eeg_total_channels"),
                _num_pct("STD_ts_noisy_channels_eeg_num", "STD_ts_noisy_channels_eeg_percentage"),
                _num_pct("PTP_ts_noisy_channels_eeg_num", "PTP_ts_noisy_channels_eeg_percentage"),
                _num_pct("STD_ts_flat_channels_eeg_num", "STD_ts_flat_channels_eeg_percentage"),
                _num_pct("PTP_ts_flat_channels_eeg_num", "PTP_ts_flat_channels_eeg_percentage"),
            ]
            c = next((x for x in candidates if x is not None), None)
            return int(round(c)) if c is not None else None

        mag_candidates = [
            _row_val("ECG_mag_total_channels"),
            _row_val("EOG_mag_total_channels"),
            _num_pct("STD_ts_noisy_channels_mag_num", "STD_ts_noisy_channels_mag_percentage"),
            _num_pct("PTP_ts_noisy_channels_mag_num", "PTP_ts_noisy_channels_mag_percentage"),
            _num_pct("STD_ts_flat_channels_mag_num", "STD_ts_flat_channels_mag_percentage"),
            _num_pct("PTP_ts_flat_channels_mag_num", "PTP_ts_flat_channels_mag_percentage"),
        ]
        grad_candidates = [
            _row_val("ECG_grad_total_channels"),
            _row_val("EOG_grad_total_channels"),
            _num_pct("STD_ts_noisy_channels_grad_num", "STD_ts_noisy_channels_grad_percentage"),
            _num_pct("PTP_ts_noisy_channels_grad_num", "PTP_ts_noisy_channels_grad_percentage"),
            _num_pct("STD_ts_flat_channels_grad_num", "STD_ts_flat_channels_grad_percentage"),
            _num_pct("PTP_ts_flat_channels_grad_num", "PTP_ts_flat_channels_grad_percentage"),
        ]
        mag = next((x for x in mag_candidates if x is not None), None)
        grad = next((x for x in grad_candidates if x is not None), None)
        if mag is None and grad is None:
            return None
        return int(round((mag or 0.0) + (grad or 0.0)))

    def _frame_channel_total(frame: pd.DataFrame) -> Optional[int]:
        if frame.empty:
            return None
        total = 0
        found = False
        for idx in frame.index.tolist():
            c = _row_channel_total(int(idx)) if isinstance(idx, (int, np.integer)) else _row_channel_total(idx)
            if c is None:
                continue
            found = True
            total += int(c)
        return int(total) if found else None

    cards_gqi: List[str] = []
    cards_env: List[str] = []
    cards_phys: List[str] = []
    metric_colors = {
        "GQI": "#2A6FBB",
        "STD": "#2B9348",
        "PtP": "#E07A00",
        "PSD": "#C23B22",
        "ECG": "#7B5CC4",
        "EOG": "#0B8F8C",
        "Muscle": "#8D6E63",
    }
    for metric_name, y_label, frame in _summary_distribution_components(df_view, view=view):
        vals = _coerce_numeric(frame["value"]) if "value" in frame.columns else pd.Series(dtype=float)
        n_rec = int(np.isfinite(vals.to_numpy(dtype=float)).sum())
        frame_ch_total = _frame_channel_total(frame)
        n_ch = "n/a" if frame_ch_total is None else str(int(frame_ch_total))
        fig = _plot_summary_distribution_recordings_qc(
            frame,
            title=metric_name,
            y_label=y_label,
            color=metric_colors.get(metric_name, "#2A6FBB"),
            enable_style_controls=False,
            show_counts_annotation=False,
        )
        card = (
            "<div class='summary-dist-card'>"
            + f"<div class='summary-card-meta'>N recordings={n_rec}; N channels={n_ch}</div>"
            + "<div class='fig summary-strip-fig'>"
            + _figure_to_div(fig, include_axis_size_control=False, include_plot_controls=False)
            + "</div>"
            + "</div>"
        )
        if metric_name == "GQI":
            cards_gqi.append(card)
        elif metric_name in {"STD", "PtP", "PSD"}:
            cards_env.append(card)
        else:
            cards_phys.append(card)
    if not cards_gqi and not cards_env and not cards_phys:
        return "<p>No summary distributions available.</p>"
    return (
        (
            "<div class='summary-category-block'><h3>Global quality index</h3>"
            + f"<div class='summary-dist-grid' data-summary-group='{summary_group_id}'>"
            + "".join(cards_gqi)
            + "</div></div>"
            if cards_gqi
            else ""
        )
        + (
            "<div class='summary-category-block'><h3>Environmental/Hardware noise</h3>"
            + f"<div class='summary-dist-grid' data-summary-group='{summary_group_id}'>"
            + "".join(cards_env)
            + "</div></div>"
            if cards_env
            else ""
        )
        + (
            "<div class='summary-category-block'><h3>Physiological noise</h3>"
            + f"<div class='summary-dist-grid' data-summary-group='{summary_group_id}'>"
            + "".join(cards_phys)
            + "</div></div>"
            if cards_phys
            else ""
        )
        + _summary_shared_controls_panel_html(summary_group_id)
        + (
            "<div class='summary-shared-note'><strong>How to interpret:</strong> "
            "Each panel is one QC metric. One dot is one recording row (subject x task); hover shows entities. "
            "Violin + box represent pooled recording distribution. "
            "QC summary units are shown on each y-axis (percentages, burden units, or event counts)."
            "</div>"
        )
    )


def plot_task_profiles(df: pd.DataFrame, title: str, y_label: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    ordered_tasks = sorted(set(df["task_label"].astype(str).tolist()))
    if not ordered_tasks:
        return None
    subject_ids = sorted(set(df["subject"].astype(str).tolist()))
    colors = _palette(subject_ids)

    fig = go.Figure()
    for subject in subject_ids:
        sub = df.loc[df["subject"] == subject]
        vals_by_task = {}
        for task in ordered_tasks:
            vals = _coerce_numeric(sub.loc[sub["task_label"] == task, "value"]).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            vals_by_task[task] = float(np.mean(vals)) if vals.size else np.nan
        y = np.array([vals_by_task[t] for t in ordered_tasks], dtype=float)
        if not np.any(np.isfinite(y)):
            continue
        fig.add_trace(
            go.Scatter(
                x=ordered_tasks,
                y=y,
                mode="lines+markers",
                line={"width": 1.6, "color": colors[subject]},
                marker={"size": 6},
                name=subject,
                legendgroup=subject,
                showlegend=False,
                hovertemplate="subject=" + subject + "<br>task=%{x}<br>value=%{y:.3g}<extra></extra>",
            )
        )

    med = []
    for task in ordered_tasks:
        vals = _coerce_numeric(df.loc[df["task_label"] == task, "value"]).to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        med.append(np.nanmedian(vals) if vals.size else np.nan)
    fig.add_trace(
        go.Scatter(
            x=ordered_tasks,
            y=med,
            mode="lines+markers",
            line={"width": 3.2, "color": "#0B3D91"},
            marker={"size": 7},
            name="Cohort median",
            showlegend=True,
            hovertemplate="task=%{x}<br>median=%{y:.3g}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task",
        yaxis_title=y_label,
        **_adaptive_legend_layout(1),
    )
    return fig


def plot_subject_task_heatmap(df: pd.DataFrame, title: str, color_title: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    pivot = (
        df.pivot_table(index="subject", columns="task_label", values="value", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    z = pivot.to_numpy(dtype=float)
    x = list(pivot.columns)
    y = list(pivot.index)
    custom = np.empty(z.shape + (2,), dtype=object)
    for i, subj in enumerate(y):
        for j, task in enumerate(x):
            custom[i, j, 0] = subj
            custom[i, j, 1] = task

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            customdata=custom,
            colorscale="Viridis",
            colorbar={"title": color_title},
            hovertemplate="subject=%{customdata[0]}<br>task=%{customdata[1]}<br>value=%{z:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center"},
        template="plotly_white",
        xaxis_title="Task",
        yaxis_title="Subject",
        margin={"l": 70, "r": 40, "t": 90, "b": 60},
        height=max(560, 28 * len(y) + 220),
    )
    return fig


def _subject_ranking_table_html(df: pd.DataFrame, title: str, value_label: str) -> str:
    if df.empty:
        return "<p>No subject ranking available.</p>"
    grp = df.groupby("subject", dropna=False)
    tab = pd.DataFrame(
        {
            "subject": grp["subject"].first(),
            "n_tasks": grp["task_label"].nunique(),
            "mean": grp["value"].mean(),
            "median": grp["value"].median(),
            "upper_tail_p95": grp["value"].quantile(0.95),
        }
    ).reset_index(drop=True)
    tab = tab.sort_values("upper_tail_p95", ascending=False)
    rows: List[str] = []
    for _, r in tab.iterrows():
        mean_txt = f"{r['mean']:.3g}" if np.isfinite(r["mean"]) else "n/a"
        median_txt = f"{r['median']:.3g}" if np.isfinite(r["median"]) else "n/a"
        p95_txt = f"{r['upper_tail_p95']:.3g}" if np.isfinite(r["upper_tail_p95"]) else "n/a"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['subject']))}</td>"
            f"<td>{int(r['n_tasks'])}</td>"
            f"<td>{mean_txt}</td>"
            f"<td>{median_txt}</td>"
            f"<td>{p95_txt}</td>"
            + "</tr>"
        )
    return (
        "<div class='fig'>"
        f"<h4>{html.escape(title)}</h4>"
        "<div class='ranking-scroll'>"
        "<table class='ranking-table'>"
        "<thead><tr>"
        "<th>Subject</th>"
        "<th>N tasks</th>"
        f"<th>Mean ({html.escape(value_label)})</th>"
        f"<th>Median ({html.escape(value_label)})</th>"
        f"<th>p95 ({html.escape(value_label)})</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# HTML helpers (lazy loading + tabs)
# ---------------------------------------------------------------------------


def _reset_lazy_figure_store() -> None:
    global _LAZY_PLOT_COUNTER, _LAZY_PAYLOAD_COUNTER
    _LAZY_PLOT_PAYLOADS.clear()
    _LAZY_PLOT_COUNTER = count(1)
    _LAZY_PAYLOAD_COUNTER = count(1)


def _extract_figure_controls(fig: go.Figure) -> List[Dict[str, object]]:
    """Extract Plotly updatemenus as external HTML controls and clear them from layout."""
    menus = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    controls: List[Dict[str, object]] = []
    if not menus:
        return controls

    for idx, menu in enumerate(menus):
        mj = menu.to_plotly_json() if hasattr(menu, "to_plotly_json") else dict(menu)
        raw_buttons = list(mj.get("buttons", []) or [])
        buttons: List[Dict[str, object]] = []
        for b in raw_buttons:
            if not isinstance(b, dict):
                continue
            buttons.append(
                {
                    "label": str(b.get("label", "")),
                    "method": str(b.get("method", "restyle")),
                    "args": b.get("args", []),
                }
            )
        if not buttons:
            continue
        controls.append(
            {
                "title": _infer_updatemenu_title(menu, idx),
                "active": int(mj.get("active", 0) or 0),
                "buttons": buttons,
            }
        )
    # Important: update_layout(updatemenus=[]) may keep existing menus in
    # serialized layout; assign directly to clear them reliably.
    fig.layout.updatemenus = None
    return controls


def _register_lazy_figure(fig: go.Figure, *, height_px: str, controls: Optional[List[Dict[str, object]]] = None) -> str:
    fig_id = f"lazy-qc-plot-{next(_LAZY_PLOT_COUNTER)}"
    payload_id = f"lazy-qc-payload-{next(_LAZY_PAYLOAD_COUNTER)}"
    payload_json = json.dumps(
        {
            "figure": fig.to_plotly_json(),
            "config": {"responsive": True, "displaylogo": False},
            "controls": controls or [],
        },
        cls=PlotlyJSONEncoder,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    _LAZY_PLOT_PAYLOADS[payload_id] = payload_json
    return (
        "<div class='lazy-plot-wrap'>"
        f"<div id='{fig_id}' class='js-lazy-plot' data-payload-id='{payload_id}' "
        f"style='height:{height_px}; width:100%;'></div>"
        "<div class='plot-controls'></div>"
        "</div>"
    )


def _lazy_payload_script_tags_html() -> str:
    return "".join(
        f"<script id='{pid}' type='application/json'>{pjson}</script>"
        for pid, pjson in _LAZY_PLOT_PAYLOADS.items()
    )


def _figure_to_div(
    fig: Optional[go.Figure],
    *,
    include_axis_size_control: bool = True,
    include_plot_controls: bool = True,
) -> str:
    if fig is None:
        return "<p>No data available for this panel.</p>"
    fig_out = go.Figure(fig)

    # ── Zone 1: extract title into an independent HTML block ─────────
    title_html = ""
    fig_title = fig_out.layout.title
    if fig_title:
        title_text = ""
        if isinstance(fig_title, str):
            title_text = fig_title
        else:
            title_text = str(getattr(fig_title, "text", "") or "")
        if title_text:
            title_html = f"<div class='fig-title'>{html.escape(title_text)}</div>"
            fig_out.update_layout(title=None)

    # ── Zone 2 (legend) stays inside the Plotly figure at y>1 ────────
    # ── Zone 3 (plot) occupies the remaining Plotly area ─────────────
    if include_axis_size_control:
        _attach_axis_label_size_controller(fig_out)
    controls = _extract_figure_controls(fig_out) if include_plot_controls else []
    height = fig_out.layout.height
    if height is None or (isinstance(height, (float, int)) and not np.isfinite(height)):
        height = 620
    return title_html + _register_lazy_figure(fig_out, height_px=f"{int(max(420, float(height)))}px", controls=controls)


def _figure_block(fig: Optional[go.Figure], interpretation: str) -> str:
    return (
        f"<div class='fig'>{_figure_to_div(fig)}</div>"
        f"<p class='fig-note'><strong>How to interpret:</strong> {interpretation}</p>"
    )


def _build_subtabs_html(group_id: str, tabs: Sequence[Tuple[str, str]], *, level: int = 1) -> str:
    if not tabs:
        return "<p>No content available.</p>"
    btns = []
    panels = []
    for idx, (label, html) in enumerate(tabs):
        panel_id = f"{group_id}-panel-{idx}"
        active = " active" if idx == 0 else ""
        btns.append(
            f"<button class='subtab-btn{active}' data-tab-group='{group_id}' data-target='{panel_id}'>{label}</button>"
        )
        panels.append(
            f"<div id='{panel_id}' class='subtab-content{active}' data-tab-group='{group_id}'>{html}</div>"
        )
    lvl = max(1, int(level))
    return f"<div class='subtab-group level-{lvl}'><div class='subtab-row'>{''.join(btns)}</div>{''.join(panels)}</div>"


def _config_snapshot(cfg_path: Optional[Path]) -> str:
    if cfg_path is None or (not cfg_path.exists()):
        return "No attempt-matched GQI config file was found."
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    lines = [f"source={cfg_path}"]
    for section in cfg.sections():
        lines.append(f"[{section}]")
        for key, value in cfg[section].items():
            lines.append(f"  {key}={value}")
        lines.append("")
    return "\n".join(lines)


def _general_settings_snapshot(megqc_root: str) -> str:
    """Load the latest general MEGqc settings ini (same source used by QA report)."""
    config_dir = Path(megqc_root) / "config"
    ini_files = sorted(config_dir.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ini_files:
        return "No general MEGqc settings ini was found."

    chosen = ini_files[0]
    cfg = configparser.ConfigParser()
    cfg.read(chosen)

    lines = [f"source={chosen}"]
    disallowed_tokens = ("bad", "good", "reject", "flag", "exceed", "pass", "fail", "threshold")
    sections = ["default", "Filtering", "Epoching", "STD", "PTP_manual", "PSD", "ECG", "EOG", "Muscle"]
    for section in sections:
        if section not in cfg:
            continue
        safe_items: List[str] = []
        for key, value in cfg[section].items():
            key_l = key.lower()
            val_l = str(value).lower()
            if any(tok in key_l for tok in disallowed_tokens):
                continue
            if any(tok in val_l for tok in disallowed_tokens):
                continue
            safe_items.append(f"{key}={value}")
        if safe_items:
            lines.append(f"[{section}]")
            for item in safe_items:
                lines.append(f"  {item}")
            lines.append("")
        else:
            lines.append(f"[{section}] QA-safe settings view active")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _summary_table_html(df: pd.DataFrame, selected_cols: Sequence[str]) -> str:
    n_subjects = int(df["subject"].nunique()) if "subject" in df.columns else 0
    n_rows = int(len(df))
    tasks = sorted(set(df.get("task_label", pd.Series(dtype=str)).astype(str).tolist()))

    rows = []
    for col in selected_cols:
        if col not in df.columns:
            rows.append(f"<tr><td>{col}</td><td>0</td><td>{n_rows}</td></tr>")
            continue
        s = _coerce_numeric(df[col])
        avail = int(np.isfinite(s).sum())
        rows.append(f"<tr><td>{col}</td><td>{avail}</td><td>{max(0, n_rows-avail)}</td></tr>")

    task_rows = []
    if "task_label" in df.columns and "subject" in df.columns:
        grouped = (
            df.groupby("task_label", dropna=False)
            .agg(n_subjects=("subject", "nunique"), n_rows=("task_label", "size"))
            .reset_index()
            .sort_values("task_label")
        )
        for _, row in grouped.iterrows():
            task_rows.append(
                f"<tr><td>{html.escape(str(row['task_label']))}</td>"
                f"<td>{int(row['n_subjects'])}</td>"
                f"<td>{int(row['n_rows'])}</td></tr>"
            )
    task_table = (
        "<table><thead><tr><th>Task</th><th>N subjects</th><th>N rows</th></tr></thead>"
        f"<tbody>{''.join(task_rows) if task_rows else '<tr><td colspan=3>n/a</td></tr>'}</tbody></table>"
    )

    return (
        "<div class='summary-grid'>"
        "<div class='tile'>"
        f"<h3>Dataset summary</h3><p><strong>N subjects:</strong> {n_subjects}</p>"
        f"<p><strong>N task-level rows:</strong> {n_rows}</p>"
        f"<p><strong>Tasks:</strong> {', '.join(tasks) if tasks else 'n/a'}</p>"
        "</div>"
        "<div class='tile'><h3>Subjects per task</h3>"
        f"{task_table}</div>"
        "<div class='tile'><h3>Column availability</h3>"
        "<table><thead><tr><th>Column</th><th>Available n</th><th>Missing n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</div>"
    )


def _metric_component_panel(
    df_view: pd.DataFrame,
    spec: ComponentSpec,
    view_label: str,
    metric_name: str,
    dataset_order: Optional[Sequence[str]] = None,
) -> str:
    if df_view.empty:
        return "<p>No data available for this component.</p>"
    multi_dataset_mode = "dataset" in df_view.columns and (df_view["dataset"].astype(str).nunique() > 1)

    boxplot = plot_box_with_subject_points(
        df_view,
        title=f"{spec.label} by task ({view_label}) - boxplot",
        y_label=f"{spec.label} ({spec.unit})",
        multi_dataset_mode=multi_dataset_mode,
    )
    violin = plot_violin_with_subject_points(
        df_view,
        title=f"{spec.label} by task ({view_label}) - violin densities",
        y_label=f"{spec.label} ({spec.unit})",
        multi_dataset_mode=multi_dataset_mode,
    )
    hist = plot_histogram_by_task(
        df_view,
        title=f"{spec.label} by task ({view_label}) - histogram",
        x_label=f"{spec.label} ({spec.unit})",
    )
    density = plot_density_by_task(
        df_view,
        title=f"{spec.label} by task ({view_label}) - density",
        x_label=f"{spec.label} ({spec.unit})",
    )
    profile = plot_task_profiles(
        df_view,
        title=f"Within-subject profiles across tasks ({view_label})",
        y_label=f"{spec.label} ({spec.unit})",
    )
    ranking_html = _subject_ranking_table_html(
        df_view,
        title=f"Subject ranking by {spec.label} ({view_label})",
        value_label=spec.unit,
    )

    dist_tabs = _build_subtabs_html(
        f"dist-{spec.key}-{view_label.lower().replace(' ', '-')}",
        [
            (
                "Boxplot",
                _figure_block(
                    boxplot,
                    (
                        f"X-axis is task. Y-axis is {spec.label} in {spec.unit}. "
                        "Each box summarizes the empirical spread (median, quartiles, whiskers) for one task-level group. "
                        "Jittered dots are one robust subject summary per task group and are colored by subject identity."
                    ),
                ),
            ),
            (
                "Violin densities",
                _figure_block(
                    violin,
                    (
                        f"X-axis is task. Y-axis is {spec.label} in {spec.unit}. "
                        f"Each violin shows the full task-level distribution across subject-task rows; wider segments indicate higher density. "
                        "Jittered dots are one robust value per subject per task group (median if multiple rows exist) and are colored by subject identity."
                    ),
                ),
            ),
            (
                "Histogram",
                _figure_block(
                    hist,
                    (
                        f"X-axis is {spec.label} ({spec.unit}); Y-axis is probability density. "
                        "Bars compare task-wise empirical distributions. "
                        "Separation between task histograms indicates task-linked differences in QC burden."
                    ),
                ),
            ),
            (
                "Density",
                _figure_block(
                    density,
                    (
                        f"X-axis is {spec.label} ({spec.unit}); Y-axis is kernel density estimate. "
                        "Curves summarize distribution shape per task. "
                        "Broad or multi-peaked curves indicate high between-subject variability."
                    ),
                ),
            ),
        ],
        level=4,
    )

    detail_panels: str
    if multi_dataset_mode:
        profile_ds_tabs: List[Tuple[str, str]] = []
        ranking_ds_tabs: List[Tuple[str, str]] = []
        ds_iter = list(dataset_order) if dataset_order else sorted(df_view["dataset"].astype(str).unique())
        for ds in ds_iter:
            ds_df = df_view.loc[df_view["dataset"].astype(str) == ds].copy()
            if ds_df.empty:
                msg = f"<p>Dataset <strong>{html.escape(str(ds))}</strong> has no values for {spec.label} in {view_label} view.</p>"
                profile_ds_tabs.append((str(ds), msg))
                ranking_ds_tabs.append((str(ds), msg))
                continue
            ds_profile = plot_task_profiles(
                ds_df,
                title=f"Within-subject profiles across tasks ({view_label}) - {ds}",
                y_label=f"{spec.label} ({spec.unit})",
            )
            ds_ranking_html = _subject_ranking_table_html(
                ds_df,
                title=f"Subject ranking by {spec.label} ({view_label}) - {ds}",
                value_label=spec.unit,
            )
            profile_ds_tabs.append(
                (
                    ds,
                    _figure_block(
                        ds_profile,
                        (
                            f"X-axis is task; Y-axis is {spec.label} ({spec.unit}). "
                            "Thin lines are within-subject trajectories and thick line is cohort median for this dataset only."
                        ),
                    ),
                )
            )
            ranking_ds_tabs.append(
                (
                    ds,
                    ds_ranking_html
                    + (
                        "<p class='fig-note'><strong>How to interpret:</strong> "
                        f"Scroll the ranking table to inspect subjects with highest upper-tail (p95) {spec.label} within dataset {ds}.</p>"
                    ),
                )
            )
        detail_panels = _build_subtabs_html(
            f"detail-datasets-view-{spec.key}-{view_label.lower().replace(' ', '-')}",
            [
                (
                    "Within-subject profiles across tasks",
                    _build_subtabs_html(
                        f"detail-datasets-profile-{spec.key}-{view_label.lower().replace(' ', '-')}",
                        profile_ds_tabs,
                        level=5,
                    ),
                ),
                (
                    "Subject ranking table",
                    _build_subtabs_html(
                        f"detail-datasets-ranking-{spec.key}-{view_label.lower().replace(' ', '-')}",
                        ranking_ds_tabs,
                        level=5,
                    ),
                ),
            ],
            level=4,
        )
    else:
        detail_panels = (
            _figure_block(
                profile,
                (
                    f"X-axis is task; Y-axis is {spec.label} ({spec.unit}). "
                    "Thin lines are within-subject trajectories across tasks; the thick line is cohort median per task. "
                    "Use this panel to separate subject-specific shifts from cohort-level task effects."
                ),
            )
            + ranking_html
            + (
                "<p class='fig-note'><strong>How to interpret:</strong> "
                f"The ranking table is scrollable and lists subjects ordered by upper-tail (p95) {spec.label} across tasks. "
                f"Columns show mean, median, and p95 in {spec.unit}; use it to prioritize subject-level QC follow-up.</p>"
            )
        )

    return (
        "<div class='metric-block'>"
        f"<h4>{spec.label}</h4>"
        f"<p><strong>Unit:</strong> {spec.unit}</p>"
        + dist_tabs
        + detail_panels
        + "</div>"
    )


def _metric_tab_content(df: pd.DataFrame, metric_name: str, view: str, view_label: str) -> str:
    specs = COMPONENTS.get(metric_name, [])
    if not specs:
        return "<p>No metric definition available.</p>"

    comp_tabs: List[Tuple[str, str]] = []
    dataset_order = (
        sorted(df["dataset"].astype(str).unique())
        if ("dataset" in df.columns and df["dataset"].astype(str).nunique() > 1)
        else None
    )
    for spec in specs:
        comp_df = _component_frame(df, spec, view=view)
        if comp_df.empty:
            continue
        comp_tabs.append(
            (
                spec.label,
                _metric_component_panel(comp_df, spec, view_label, metric_name, dataset_order=dataset_order),
            )
        )

    if not comp_tabs:
        return "<p>No available columns for this metric in the selected attempt.</p>"

    return _build_subtabs_html(
        f"metric-{metric_name.lower()}-{view}",
        comp_tabs,
        level=3,
    )


def _build_summary_distributions_section_qc(df: pd.DataFrame, *, view: str, view_label: str) -> str:
    """First-level QC summary distributions (single or multi-dataset)."""
    if df.empty:
        return "<section><h2>Summary distributions</h2><p>No QC rows available for this view.</p></section>"

    multi_dataset = "dataset" in df.columns and (df["dataset"].astype(str).nunique() > 1)
    if multi_dataset:
        tabs: List[Tuple[str, str]] = [
            (
                "Pooled sample",
                _summary_distribution_grid_html(
                    df,
                    view=view,
                    view_label=view_label,
                    summary_group_id=f"qc-summary-strip-{re.sub(r'[^a-z0-9]+','-',view_label.lower())}-pooled",
                ),
            )
        ]
        for ds in sorted(df["dataset"].astype(str).unique()):
            ds_df = df.loc[df["dataset"].astype(str) == ds].copy()
            tabs.append(
                (
                    ds,
                    _summary_distribution_grid_html(
                        ds_df,
                        view=view,
                        view_label=f"{view_label} - {ds}",
                        summary_group_id=f"qc-summary-strip-{re.sub(r'[^a-z0-9]+','-',view_label.lower())}-{re.sub(r'[^a-z0-9]+','-',ds.lower())}",
                    ),
                )
            )
        body = _build_subtabs_html(
            f"qc-summary-dist-{re.sub(r'[^a-z0-9]+','-',view_label.lower())}",
            tabs,
            level=2,
        )
    else:
        body = _summary_distribution_grid_html(
            df,
            view=view,
            view_label=view_label,
            summary_group_id=f"qc-summary-strip-{re.sub(r'[^a-z0-9]+','-',view_label.lower())}",
        )

    return (
        "<section>"
        "<h2>Summary distributions</h2>"
        "<p>Compact pooled QC screen. Each dot is one recording row; hover reveals dataset/subject/task entities.</p>"
        + body
        + "</section>"
    )


def _tab_content(
    df: pd.DataFrame,
    top_tab: str,
    general_settings_snapshots: Dict[str, str],
    gqi_settings_snapshots: Dict[str, str],
) -> str:
    if top_tab == "Combined (mag+grad)":
        view = "combined"
    elif top_tab == "MAG":
        view = "mag"
    elif top_tab == "EEG":
        view = "eeg"
    else:
        view = "grad"

    has_view_data = _has_channel_type_data(df, view) if view in {"mag", "grad", "eeg"} else True
    metric_tabs: List[Tuple[str, str]] = []
    if has_view_data:
        for metric in METRIC_ORDER:
            metric_tabs.append((metric, _metric_tab_content(df, metric, view=view, view_label=top_tab)))
    else:
        metric_tabs.append(
            (
                "No data",
                (
                    "<section><h2>Channel-type availability</h2>"
                    f"<p>This dataset has no {view.upper()} information in the loaded QC TSV columns, so metric panels are not available for this tab.</p>"
                    "</section>"
                ),
            )
        )

    selected_cols = sorted({
        c
        for specs in COMPONENTS.values()
        for spec in specs
        for c in (spec.mag_col, spec.grad_col, spec.global_col)
        if c
    })

    snapshot_tiles: List[str] = []
    for ds_name in sorted(general_settings_snapshots):
        snapshot_tiles.append(
            "<div class='tile'>"
            f"<h3>General MEGqc settings snapshot ({html.escape(ds_name)})</h3>"
            f"<pre>{html.escape(general_settings_snapshots[ds_name])}</pre>"
            "</div>"
        )
    for ds_name in sorted(gqi_settings_snapshots):
        snapshot_tiles.append(
            "<div class='tile'>"
            f"<h3>GQI attempt settings snapshot ({html.escape(ds_name)})</h3>"
            f"<pre>{html.escape(gqi_settings_snapshots[ds_name])}</pre>"
            "</div>"
        )

    overview_html = (
        "<section>"
        "<h2>Cohort QC overview</h2>"
        "<p><strong>Purpose:</strong> summarize QC percentages and burdens for the selected channel-type view before diving into per-metric panels.</p>"
        + _summary_table_html(df, selected_cols)
        + "<div class='summary-grid'>"
        + "".join(snapshot_tiles)
        + "</div>"
        + "</section>"
    )

    metrics_html = (
        "<section>"
        "<h2>QC metrics details</h2>"
        "<p><strong>Important:</strong> each metric panel provides distribution views, subject trajectories, and ranking to complement QA with explicit QC burden summaries.</p>"
        + _build_subtabs_html(f"top-{re.sub(r'[^a-z0-9]+','-',top_tab.lower())}", metric_tabs, level=2)
        + "</section>"
    )

    return _build_subtabs_html(
        f"qc-sections-{re.sub(r'[^a-z0-9]+','-',top_tab.lower())}",
        [
            ("Summary distributions", _build_summary_distributions_section_qc(df, view=view, view_label=top_tab)),
            ("Cohort QC overview", overview_html),
            ("QC metrics details", metrics_html),
        ],
        level=1,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _build_report_html(
    report_name: str,
    df: pd.DataFrame,
    source_rows: Sequence[str],
    attempt_text: str,
    general_settings_snapshots: Dict[str, str],
    gqi_settings_snapshots: Dict[str, str],
    allowed_tabs: Optional[Sequence[str]] = None,
) -> str:
    _reset_lazy_figure_store()
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = getattr(meg_qc, "__version__", "unknown")

    # Determine which tabs to include
    tabs_to_use = list(allowed_tabs) if allowed_tabs is not None else list(TOP_TABS)
    if allowed_tabs is not None:
        # Caller explicitly selected these tabs — trust the selection
        active_tabs = list(allowed_tabs)
    else:
        active_tabs = []
        for top_tab in tabs_to_use:
            if top_tab == "Combined (mag+grad)":
                view = "combined"
            elif top_tab == "MAG":
                view = "mag"
            elif top_tab == "EEG":
                view = "eeg"
            else:
                view = "grad"
            if view == "combined":
                # Combined tab should appear only if mag or grad data exists
                if _has_channel_type_data(df, "mag") or _has_channel_type_data(df, "grad"):
                    active_tabs.append(top_tab)
            else:
                if _has_channel_type_data(df, view):
                    active_tabs.append(top_tab)
        if not active_tabs:
            active_tabs = list(tabs_to_use)  # fallback: show all requested tabs

    tab_buttons = []
    tab_panels = []
    for idx, top_tab in enumerate(active_tabs):
        tab_id = f"qc-top-tab-{idx}"
        active = " active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn{active}' data-target='{tab_id}'>{top_tab}</button>")
        tab_panels.append(
            f"<div id='{tab_id}' class='tab-content{active}'>{_tab_content(df, top_tab, general_settings_snapshots, gqi_settings_snapshots)}</div>"
        )

    lazy_payload_scripts = _lazy_payload_script_tags_html()
    sources_html = "".join(f"<li>{html.escape(line)}</li>" for line in source_rows)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QC group report {report_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      margin: 0;
      color: #1a1a1a;
      background: linear-gradient(135deg, #f6fbff, #edf6ff 45%, #f9fcff);
    }}
    main {{ max-width: 1360px; margin: 0 auto; padding: 28px 24px 56px; }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; letter-spacing: 0.2px; }}
    h2 {{ margin-top: 24px; font-size: 22px; }}
    h3 {{ margin-top: 12px; font-size: 18px; }}
    h4 {{ margin-top: 14px; font-size: 15px; color: #284b63; }}
    .report-header {{ display: flex; gap: 14px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; }}
    section {{
      background: rgba(255, 255, 255, 0.90);
      border: 1px solid #dce9f7;
      border-radius: 14px;
      padding: 16px 16px 10px;
      margin-top: 16px;
      box-shadow: 0 6px 24px rgba(5, 45, 79, 0.06);
    }}
    .metric-block {{ border-top: 1px solid #e7eef8; margin-top: 14px; padding-top: 12px; }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      background: #f4f8fd;
      border: 1px solid #d8e4f3;
      border-radius: 8px;
      padding: 10px;
      font-size: 12px;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
    th, td {{ border: 1px solid #d5e3f3; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #e7f1ff; color: #224c78; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
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
    .tile {{ border: 1px solid #d9e7f8; border-radius: 10px; padding: 10px; background: #fbfdff; }}
    .fig {{ border-top: 1px solid #e7eef8; margin-top: 10px; padding-top: 10px; overflow: visible; }}
    .fig-title {{
      font-size: 16px;
      font-weight: 700;
      color: #1a3a5c;
      text-align: center;
      padding: 10px 8px 2px;
      line-height: 1.3;
    }}
    .fig .js-plotly-plot {{ width: 100% !important; }}
    .lazy-plot-wrap {{ width: 100%; }}
    .plot-controls {{
      margin-top: 10px;
      padding: 10px 12px 12px;
      border: 1px solid #c7dbf2;
      border-radius: 10px;
      background: #f4f9ff;
      display: none;
    }}
    .plot-controls.active {{ display: block; }}
    .plot-control-group {{ margin-bottom: 10px; }}
    .plot-control-group:last-child {{ margin-bottom: 0; }}
    .plot-control-title {{
      font-size: 12px;
      font-weight: 700;
      color: #21486f;
      margin-bottom: 6px;
      letter-spacing: 0.1px;
    }}
    .plot-control-row {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
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
    .fig-note {{ margin: 6px 2px 14px; font-size: 13px; line-height: 1.45; color: #2a425f; }}
    .tab-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
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
    .tab-btn.active {{ background: #1d4ed8; border-color: #1d4ed8; color: #ffffff; font-weight: 700; }}
    .tab-content {{ display: none; margin-top: 8px; }}
    .tab-content.active {{ display: block; }}
    .subtab-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 10px; }}
    .subtab-group {{ border: 1px solid #d8e6f7; border-radius: 12px; padding: 10px 10px 8px; margin-top: 10px; }}
    .subtab-group.level-1 {{ background: #dbeafe; border-color: #7fb0ea; }}
    .subtab-group.level-2 {{ background: #ecf5ff; border-color: #a8c8ee; }}
    .subtab-group.level-3 {{ background: #f5f9ff; border-color: #c1d8f2; }}
    .subtab-group.level-4 {{ background: #ffffff; border-color: #d9e6f7; }}
    .subtab-group.level-1 > .subtab-row .subtab-btn {{ background: #cfe3ff; border-color: #79a9e4; font-weight: 600; }}
    .subtab-group.level-2 > .subtab-row .subtab-btn {{ background: #e2efff; border-color: #9fc2e8; }}
    .subtab-group.level-3 > .subtab-row .subtab-btn {{ background: #edf5ff; border-color: #b2cfea; }}
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
    .subtab-btn.active {{ background: #cfe3ff; border-color: #79a9e4; color: #16a34a; font-weight: 700; }}
    .ranking-scroll {{
      max-height: 520px;
      overflow-y: auto;
      border: 1px solid #d7e5f7;
      border-radius: 10px;
      background: #ffffff;
    }}
    .ranking-table {{ width: 100%; border-collapse: collapse; margin-top: 0; }}
    .ranking-table thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #e7f1ff;
      color: #1f3f61;
    }}
    .ranking-table td, .ranking-table th {{ padding: 7px 9px; border: 1px solid #d5e3f3; }}
    .subtab-content {{ display: none; }}
    .subtab-content.active {{ display: block; }}
    .report-tools {{ display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }}
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
    .tool-btn.active {{ background: #1d4ed8; border-color: #1d4ed8; color: #ffffff; }}
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
    .loading-overlay.hidden {{ opacity: 0; pointer-events: none; }}
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
    .loading-title {{ font-weight: 700; color: #1e3a5f; margin-bottom: 4px; }}
    .loading-subtitle {{ color: #334e68; font-size: 13px; }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div id="report-loading-overlay" class="loading-overlay">
    <div class="loading-card">
      <div class="loading-spinner"></div>
      <div class="loading-title">Loading QC report</div>
      <div class="loading-subtitle">Rendering visible figures...</div>
    </div>
  </div>
  <main>
    <section>
      <div class="report-header">
        <div>
          <h1>QC group report: {report_name}</h1>
          <p><strong>Generated:</strong> {generated}</p>
          <p><strong>MEGqc version:</strong> {version}</p>
          <p><strong>Attempt selection:</strong> {html.escape(attempt_text)}</p>
          <p><strong>Important:</strong> this QC report complements QA views with burden percentages and ranking summaries per metric.</p>
        </div>
      </div>
      <div class="report-tools">
        <button id="grid-toggle-btn" class="tool-btn active" type="button">Hide grids</button>
      </div>
      <div class="tab-row">
        {''.join(tab_buttons)}
      </div>
      {''.join(tab_panels)}
      <section>
        <h2>Machine-readable input</h2>
        <ul>
          {sources_html}
        </ul>
      </section>
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
        if (!payloadId) return null;
        if (lazyPayloadCache[payloadId]) return lazyPayloadCache[payloadId];
        const payloadEl = document.getElementById(payloadId);
        if (!payloadEl || !payloadEl.textContent) return null;
        try {{
          const payload = JSON.parse(payloadEl.textContent);
          lazyPayloadCache[payloadId] = payload;
          payloadEl.textContent = '';
          if (payloadEl.parentNode) payloadEl.parentNode.removeChild(payloadEl);
          return payload;
        }} catch (err) {{
          return null;
        }}
      }}

      function hideLoadingOverlay() {{
        if (!loadingOverlay || loadingOverlay.dataset.hidden === '1') return;
        loadingOverlay.dataset.hidden = '1';
        loadingOverlay.classList.add('hidden');
        window.setTimeout(() => {{
          if (loadingOverlay && loadingOverlay.parentNode) loadingOverlay.parentNode.removeChild(loadingOverlay);
        }}, 260);
      }}

      function runPlotlyControlAction(plotEl, control) {{
        if (!plotEl || !control || typeof Plotly === 'undefined') {{
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
        if (typeof Plotly === 'undefined') return Promise.resolve();
        const scope = scopeRoot || document;
        const placeholders = Array.from(scope.querySelectorAll('.js-lazy-plot'));
        const renderPromises = [];
        placeholders.forEach((el) => {{
          if (el.dataset.rendered === '1') return;
          if (el.offsetParent === null) return;
          const payloadId = el.dataset.payloadId;
          const payload = getPayloadFromScript(payloadId);
          if (!payload || !payload.figure) return;
          try {{
            const p = Plotly.newPlot(el, payload.figure.data || [], payload.figure.layout || {{}}, payload.config || {{responsive: true, displaylogo: false}});
            const post = (p && typeof p.then === 'function') ? p : Promise.resolve(p);
            const withControls = post.then(() => {{
              renderExternalControls(el, payload.controls || []);
            }});
            el.dataset.rendered = '1';
            renderPromises.push(withControls.catch(() => undefined));
          }} catch (err) {{}}
        }});
        return renderPromises.length ? Promise.all(renderPromises).then(() => undefined) : Promise.resolve();
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
        if (typeof Plotly === 'undefined' || !plotEl) return;
        const layout = plotEl.layout || {{}};
        const axisKeys = Object.keys(layout).filter((k) => k.startsWith('xaxis') || k.startsWith('yaxis'));
        const relayoutUpdate = {{}};
        if (axisKeys.length === 0) {{
          relayoutUpdate['xaxis.showgrid'] = show;
          relayoutUpdate['yaxis.showgrid'] = show;
        }} else {{
          axisKeys.forEach((k) => {{ relayoutUpdate[`${{k}}.showgrid`] = show; }});
        }}
        try {{ Plotly.relayout(plotEl, relayoutUpdate); }} catch (err) {{}}
      }}

      function applyGridState(show, scopeRoot) {{
        const scope = scopeRoot || document;
        const plots = Array.from(scope.querySelectorAll('.js-plotly-plot'));
        plots.forEach((plotEl) => applyGridToPlot(plotEl, show));
      }}

      function resizePlots(targetId) {{
        if (typeof Plotly === 'undefined') return Promise.resolve();
        const root = document.getElementById(targetId);
        if (!root) return Promise.resolve();
        return renderLazyInScope(root).then(() => {{
          bindSummarySharedControls(root);
          const plots = Array.from(root.querySelectorAll('.js-plotly-plot'));
          plots.forEach((plotEl) => {{ try {{ Plotly.Plots.resize(plotEl); }} catch (err) {{}} }});
          if (!gridsVisible) applyGridState(false, root);
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
          resizePlots(targetId).then(() => window.setTimeout(() => resizePlots(targetId), 120));
        }});
      }}

      buttons.forEach(btn => btn.addEventListener('click', () => activate(btn.dataset.target)));
      if (buttons.length > 0) activate(buttons[0].dataset.target);

      if (gridToggleBtn) {{
        gridToggleBtn.addEventListener('click', () => {{
          gridsVisible = !gridsVisible;
          applyGridState(gridsVisible, document);
          gridToggleBtn.textContent = gridsVisible ? 'Hide grids' : 'Show grids';
          gridToggleBtn.classList.toggle('active', gridsVisible);
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
        if (activePanel) renderLazyInScope(activePanel);
        const activeTop = tabs.find(t => t.classList.contains('active'));
        if (activeTop) window.requestAnimationFrame(() => resizePlots(activeTop.id));
      }}

      const subButtonsAll = Array.from(document.querySelectorAll('.subtab-btn'));
      const seenGroups = new Set();
      subButtonsAll.forEach(btn => {{
        const groupId = btn.dataset.tabGroup;
        if (!groupId) return;
        btn.addEventListener('click', () => activateSubtab(groupId, btn.dataset.target));
        seenGroups.add(groupId);
      }});
      seenGroups.forEach(groupId => {{
        const firstBtn = document.querySelector(`.subtab-btn[data-tab-group="${{groupId}}"]`);
        if (firstBtn) activateSubtab(groupId, firstBtn.dataset.target);
      }});

      bindSummarySharedControls(document);

      window.addEventListener('resize', () => {{
        const active = tabs.find(t => t.classList.contains('active'));
        if (active) resizePlots(active.id);
      }});

      window.requestAnimationFrame(() => {{
        const active = tabs.find(t => t.classList.contains('active'));
        const p = active ? resizePlots(active.id) : Promise.resolve();
        p.then(() => window.setTimeout(hideLoadingOverlay, 120)).catch(() => hideLoadingOverlay());
      }});
    }})();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_group_qc_plots_meg_qc(
    dataset_path: str,
    input_tsv: Optional[str] = None,
    output_html: Optional[str] = None,
    attempt: Optional[int] = None,
    derivatives_base: Optional[str] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> Optional[Path]:
    """Build one dataset-level QC HTML report from GQI summary TSV.

    Parameters
    ----------
    dataset_path
        BIDS dataset root.
    input_tsv
        Optional explicit GQI TSV path. If omitted, latest attempt is used.
    output_html
        Optional explicit output HTML path.
    attempt
        Optional attempt number (used only when ``input_tsv`` is omitted).
    derivatives_base
        Optional external derivatives parent root (same convention as QA CLI).
    analysis_mode
        Analysis root selection mode (``legacy``, ``new``, ``reuse``, ``latest``).
    analysis_id
        Profile ID used with ``analysis_mode='reuse'``.

    Returns
    -------
    Path or None
        Path to generated HTML report, or ``None`` when generation fails.
    """
    try:
        bundle = _load_one_dataset_bundle(
            dataset_path=dataset_path,
            derivatives_base=derivatives_base,
            input_tsv=input_tsv,
            attempt=attempt,
            analysis_mode=analysis_mode,
            analysis_id=analysis_id,
        )
    except Exception as exc:
        print(f"___MEGqc___: QC group report: unable to resolve inputs: {exc}")
        return None

    report_html = _build_report_html(
        report_name=bundle.dataset_name,
        df=bundle.df,
        source_rows=[
            f"{bundle.dataset_name}: tsv={bundle.tsv_path}",
            f"{bundle.dataset_name}: config={bundle.cfg_path if bundle.cfg_path is not None else 'No matched config file found'}",
        ],
        attempt_text=str(bundle.attempt) if bundle.attempt is not None else "n/a",
        general_settings_snapshots={bundle.dataset_name: bundle.general_settings_snapshot},
        gqi_settings_snapshots={bundle.dataset_name: bundle.gqi_settings_snapshot},
    )

    if output_html:
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_html, encoding="utf-8")
        print(f"___MEGqc___: QC group report created: {out_path}")
    else:
        # Only per-modality reports; no combined agnostic report at top level
        out_path = None

    if out_path:
        print(f"___MEGqc___: Source TSV used: {bundle.tsv_path}")
        if bundle.cfg_path is not None:
            print(f"___MEGqc___: Config snapshot used: {bundle.cfg_path}")
        else:
            print("___MEGqc___: Config snapshot used: none found for selected attempt")
        print(f"___MEGqc___: Derivatives root: {bundle.derivatives_root}")
    else:
        print(f"___MEGqc___: Source TSV used: {bundle.tsv_path}")
        if bundle.cfg_path is not None:
            print(f"___MEGqc___: Config snapshot used: {bundle.cfg_path}")
        else:
            print("___MEGqc___: Config snapshot used: none found for selected attempt")
        print(f"___MEGqc___: Derivatives root: {bundle.derivatives_root}")

    # ── Per-modality separate HTML reports ──────────────────────────────
    _common_src = [
        f"{bundle.dataset_name}: tsv={bundle.tsv_path}",
        f"{bundle.dataset_name}: config={bundle.cfg_path if bundle.cfg_path is not None else 'No matched config file found'}",
    ]
    _common_kw = dict(
        attempt_text=str(bundle.attempt) if bundle.attempt is not None else "n/a",
        general_settings_snapshots={bundle.dataset_name: bundle.general_settings_snapshot},
        gqi_settings_snapshots={bundle.dataset_name: bundle.gqi_settings_snapshot},
    )
    _sfx = f"_attempt_{bundle.attempt}" if bundle.attempt is not None else ""

    # MEG-only report (Combined+MAG+GRAD) — only if this dataset has MEG recordings.
    # _has_modality uses the BIDS 'modality' column (set from TSV subfolder/filename)
    # so MEG files containing a few EEG channels are NOT counted as EEG datasets.
    has_meg = _has_modality(bundle.df, "meg")
    if has_meg:
        meg_df = _filter_by_modality(bundle.df, "meg")
        meg_html = _build_report_html(
            report_name=bundle.dataset_name, df=meg_df, source_rows=_common_src,
            allowed_tabs=["Combined (mag+grad)", "MAG", "GRAD"], **_common_kw,
        )
        meg_dir = bundle.reports_dir / "meg"
        meg_dir.mkdir(parents=True, exist_ok=True)
        meg_path = meg_dir / f"QC_group_report_{bundle.dataset_name}{_sfx}_meg.html"
        meg_path.write_text(meg_html, encoding="utf-8")
        print(f"___MEGqc___:   MEG QC report: {meg_path}")

    # EEG-only report — only if this dataset has genuine EEG recordings.
    has_eeg = _has_modality(bundle.df, "eeg")
    if has_eeg:
        eeg_df = _filter_by_modality(bundle.df, "eeg")
        eeg_html = _build_report_html(
            report_name=bundle.dataset_name, df=eeg_df, source_rows=_common_src,
            allowed_tabs=["EEG"], **_common_kw,
        )
        eeg_dir = bundle.reports_dir / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        eeg_path = eeg_dir / f"QC_group_report_{bundle.dataset_name}{_sfx}_eeg.html"
        eeg_path.write_text(eeg_html, encoding="utf-8")
        print(f"___MEGqc___:   EEG QC report: {eeg_path}")

    if out_path:
        return out_path
    # Return primary per-modality path
    if has_meg:
        return bundle.reports_dir / "meg" / f"QC_group_report_{bundle.dataset_name}{_sfx}_meg.html"
    if has_eeg:
        return bundle.reports_dir / "eeg" / f"QC_group_report_{bundle.dataset_name}{_sfx}_eeg.html"
    return None


def make_group_qc_plots_multi_meg_qc(
    dataset_paths: Sequence[str],
    output_html: Optional[str] = None,
    attempt: Optional[int] = None,
    derivatives_base: Optional[str] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> Optional[Path]:
    """Build one multi-dataset QC HTML report from multiple GQI summary TSVs."""
    if not dataset_paths:
        print("___MEGqc___: QC multi report: no datasets provided.")
        return None

    bundles: List[QCDatasetBundle] = []
    for ds in dataset_paths:
        try:
            bundle = _load_one_dataset_bundle(
                dataset_path=ds,
                derivatives_base=derivatives_base,
                input_tsv=None,
                attempt=attempt,
                analysis_mode=analysis_mode,
                analysis_id=analysis_id,
            )
        except Exception as exc:
            print(f"___MEGqc___: QC multi report: failed to load dataset '{ds}': {exc}")
            return None
        bundles.append(bundle)

    df_all = pd.concat([b.df for b in bundles], ignore_index=True)
    names = [b.dataset_name for b in bundles]
    report_name = " + ".join(names)
    source_rows: List[str] = []
    for b in bundles:
        source_rows.append(f"{b.dataset_name}: tsv={b.tsv_path}")
        source_rows.append(
            f"{b.dataset_name}: config={b.cfg_path if b.cfg_path is not None else 'No matched config file found'}"
        )
    attempts = sorted({str(b.attempt) if b.attempt is not None else "n/a" for b in bundles})
    attempt_text = ", ".join(attempts)
    general_snapshots = {b.dataset_name: b.general_settings_snapshot for b in bundles}
    gqi_snapshots = {b.dataset_name: b.gqi_settings_snapshot for b in bundles}

    report_html = _build_report_html(
        report_name=report_name,
        df=df_all,
        source_rows=source_rows,
        attempt_text=attempt_text,
        general_settings_snapshots=general_snapshots,
        gqi_settings_snapshots=gqi_snapshots,
    )

    if output_html:
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_html, encoding="utf-8")
        print(f"___MEGqc___: QC multi-dataset report created: {out_path}")
        for b in bundles:
            print(f"___MEGqc___:   {b.dataset_name} -> {b.tsv_path}")
    else:
        # Only per-modality reports; no combined agnostic report at top level
        out_path = None
        for b in bundles:
            print(f"___MEGqc___:   {b.dataset_name} -> {b.tsv_path}")

    # ── Per-modality separate HTML reports ──────────────────────────────
    _common_kw = dict(
        source_rows=source_rows,
        attempt_text=attempt_text,
        general_settings_snapshots=general_snapshots,
        gqi_settings_snapshots=gqi_snapshots,
    )
    base_dir = bundles[0].reports_dir
    suffix = f"_attempt_{attempt}" if attempt is not None else ""
    safe_name = "_".join(names)

    # Use BIDS modality column for authoritative MEG/EEG classification.
    # _has_modality checks the 'modality' column set from TSV subfolder/filename,
    # so MEG datasets that contain a few EEG channels do NOT appear in EEG reports.
    has_meg = _has_modality(df_all, "meg")
    if has_meg:
        meg_df = _filter_by_modality(df_all, "meg")
        meg_html = _build_report_html(
            report_name=report_name, df=meg_df,
            allowed_tabs=["Combined (mag+grad)", "MAG", "GRAD"], **_common_kw,
        )
        meg_dir = base_dir / "meg"
        meg_dir.mkdir(parents=True, exist_ok=True)
        meg_path = meg_dir / f"QC_group_report_multi_{safe_name}{suffix}_meg.html"
        meg_path.write_text(meg_html, encoding="utf-8")
        print(f"___MEGqc___:   MEG QC multi report: {meg_path}")
        if out_path is None:
            out_path = meg_path

    has_eeg = _has_modality(df_all, "eeg")
    if has_eeg:
        eeg_df = _filter_by_modality(df_all, "eeg")
        eeg_html = _build_report_html(
            report_name=report_name, df=eeg_df,
            allowed_tabs=["EEG"], **_common_kw,
        )
        eeg_dir = base_dir / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        eeg_path = eeg_dir / f"QC_group_report_multi_{safe_name}{suffix}_eeg.html"
        eeg_path.write_text(eeg_html, encoding="utf-8")
        print(f"___MEGqc___:   EEG QC multi report: {eeg_path}")
        if out_path is None:
            out_path = eeg_path

    return out_path
