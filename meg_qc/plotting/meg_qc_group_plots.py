"""Dataset-level QA plotting for MEGqc derivatives.

This module builds dataset-level QA HTML reports from machine-readable outputs
already produced by the MEGqc calculation step. It reads derivative TSV files
from ``derivatives/Meg_QC/calculation`` and writes group-level HTML reports to
``derivatives/Meg_QC/reports``.

Public entrypoint
-----------------
``make_group_plots_meg_qc(dataset_path, derivatives_base=None, n_jobs=1)``
"""

from __future__ import annotations

import configparser
import datetime as dt
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from itertools import count
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from plotly.subplots import make_subplots
from pandas.errors import DtypeWarning
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

import meg_qc
from meg_qc.calculation.meg_qc_pipeline import resolve_analysis_root


MODULES = ("STD", "PTP", "PSD", "ECG", "EOG", "Muscle")
CH_TYPES = ("mag", "grad", "eeg")
MAX_POINTS_ECDF = 2500
MAX_POINTS_PROFILE = 2000
MAX_POINTS_VIOLIN = 3000
MAX_POINTS_SCATTER = 3500
MAX_HEATMAP_WINDOWS = 900
MAX_HEATMAP_CHANNELS = 350
MAX_RECORDINGS_OVERVIEW = 2200
MAX_SUBJECT_LINES = 160
_FIG_TOGGLE_COUNTER = count(1)
_LAZY_PLOT_COUNTER = count(1)
_LAZY_PAYLOAD_COUNTER = count(1)
_LAZY_PLOT_PAYLOADS: Dict[str, str] = {}
TESLA_TO_PICO = 1e12


@dataclass
class RunMeta:
    run_key: str
    subject: str = "n/a"
    session: str = "n/a"
    task: str = "n/a"
    run: str = "n/a"
    condition: str = "n/a"
    acquisition: str = "n/a"
    recording: str = "n/a"
    processing: str = "n/a"
    bids_suffix: str = "n/a"  # BIDS modality suffix: "meg" or "eeg" — from derivative filename


@dataclass
class RunRecord:
    meta: RunMeta
    files: Dict[str, Path] = field(default_factory=dict)


@dataclass
class LoadedRunData:
    std_data: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_data: Dict[str, np.ndarray] = field(default_factory=dict)
    psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    ecg_raw_data: Dict[str, np.ndarray] = field(default_factory=dict)
    eog_raw_data: Dict[str, np.ndarray] = field(default_factory=dict)
    muscle_data: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_desc: Optional[str] = None
    layout_cache: Dict[str, Dict[str, "SensorLayout"]] = field(default_factory=dict)


@dataclass
class SensorLayout:
    x: np.ndarray
    y: np.ndarray
    names: List[str]
    z: Optional[np.ndarray] = None


@dataclass
class TopomapPayload:
    layout: SensorLayout
    values: np.ndarray


@dataclass
class RunMetricRow:
    run_key: str
    subject: str
    session: str
    task: str
    run: str
    condition: str
    acquisition: str
    recording: str
    processing: str
    channel_type: str
    std_mean: float = np.nan
    std_median: float = np.nan
    std_upper_tail: float = np.nan
    std_median_norm: float = np.nan
    std_upper_tail_norm: float = np.nan
    ptp_mean: float = np.nan
    ptp_median: float = np.nan
    ptp_upper_tail: float = np.nan
    ptp_median_norm: float = np.nan
    ptp_upper_tail_norm: float = np.nan
    mains_ratio: float = np.nan
    mains_harmonics_ratio: float = np.nan
    ecg_mean_abs_corr: float = np.nan
    ecg_p95_abs_corr: float = np.nan
    eog_mean_abs_corr: float = np.nan
    eog_p95_abs_corr: float = np.nan
    muscle_mean: float = np.nan
    muscle_median: float = np.nan
    muscle_p95: float = np.nan


@dataclass
class ChTypeAccumulator:
    subjects: set = field(default_factory=set)
    run_count: int = 0
    runs_by_condition: Counter = field(default_factory=Counter)
    module_present: Counter = field(default_factory=Counter)
    module_missing: Counter = field(default_factory=Counter)

    std_dist_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    std_dist_mean_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    std_dist_upper_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    std_window_profiles: List[Dict[str, np.ndarray]] = field(default_factory=list)
    std_window_profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    ptp_dist_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    ptp_dist_mean_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    ptp_dist_upper_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    ptp_window_profiles: List[Dict[str, np.ndarray]] = field(default_factory=list)
    ptp_window_profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    psd_ratio_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    psd_harmonics_ratio_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    psd_profiles: List[Dict[str, np.ndarray]] = field(default_factory=list)
    psd_profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))

    ecg_corr_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    eog_corr_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    muscle_scalar_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    muscle_mean_by_condition: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    muscle_profiles: List[np.ndarray] = field(default_factory=list)
    muscle_profiles_by_condition: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))

    std_subject_profiles: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))
    ptp_subject_profiles: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=lambda: defaultdict(list))
    run_rows: List[RunMetricRow] = field(default_factory=list)

    std_heatmap_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_score_by_condition: Dict[str, float] = field(default_factory=dict)
    std_heatmap_score_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    std_heatmap_upper_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_upper_score_by_condition: Dict[str, float] = field(default_factory=dict)
    std_heatmap_sum_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    std_heatmap_runs_by_condition: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    std_epoch_counts_by_condition: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))

    ptp_heatmap_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_score_by_condition: Dict[str, float] = field(default_factory=dict)
    ptp_heatmap_score_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    ptp_heatmap_upper_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_upper_score_by_condition: Dict[str, float] = field(default_factory=dict)
    ptp_heatmap_sum_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_heatmap_runs_by_condition: Dict[str, List[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    ptp_epoch_counts_by_condition: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))

    std_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    ptp_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    psd_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    ecg_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    eog_topomap_by_condition: Dict[str, TopomapPayload] = field(default_factory=dict)
    std_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ptp_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    psd_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    ecg_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)
    eog_topomap_count_by_condition: Dict[str, np.ndarray] = field(default_factory=dict)

    source_paths: set = field(default_factory=set)


def resolve_output_roots(dataset_path: str, external_derivatives_root: Optional[str]) -> Tuple[str, str]:
    """Return output root and derivatives root respecting optional override."""
    ds_name = os.path.basename(os.path.normpath(dataset_path))
    output_root = (
        dataset_path
        if external_derivatives_root is None
        else os.path.join(external_derivatives_root, ds_name)
    )
    derivatives_root = os.path.join(output_root, "derivatives")
    os.makedirs(derivatives_root, exist_ok=True)
    return output_root, derivatives_root


def _parse_entities_from_run_key(run_key: str) -> RunMeta:
    entities = {}
    for token in run_key.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        entities[key] = value

    condition = entities.get("acq") or entities.get("proc") or entities.get("recording") or "n/a"
    return RunMeta(
        run_key=run_key,
        subject=entities.get("sub", "n/a"),
        session=entities.get("ses", "n/a"),
        task=entities.get("task", "n/a"),
        run=entities.get("run", "n/a"),
        condition=condition,
        acquisition=entities.get("acq", "n/a"),
        recording=entities.get("recording", "n/a"),
        processing=entities.get("proc", "n/a"),
    )


def _extract_run_and_desc(file_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (run_key, desc, bids_suffix) from a derivative TSV filename.

    The BIDS suffix ('meg' or 'eeg') is the authoritative modality indicator
    and must be used to route channel-type data into the correct accumulator.
    EEG channels embedded inside a MEG recording (_meg.tsv) must NOT be
    treated as EEG-modality data.
    """
    match = re.search(r"_desc-([^_]+)_(meg|eeg)\.tsv$", file_name, flags=re.IGNORECASE)
    if not match:
        return None, None, None
    desc = match.group(1)
    bids_suffix = match.group(2).lower()  # "meg" or "eeg"
    run_key = file_name.split("_desc-")[0]
    return run_key, desc, bids_suffix


def _discover_run_records(calculation_dir: Path) -> Dict[str, RunRecord]:
    """Scan calculation derivatives and group files by run key.

    A run may have repeated outputs for the same desc (reruns or updates).
    We keep only the newest file for each run+desc pair so group plots reflect
    the latest available derivative without requiring cleanup of old files.
    """
    tmp_files: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    run_meta: Dict[str, RunMeta] = {}

    for tsv_path in calculation_dir.rglob("*.tsv"):
        run_key, desc, bids_suffix = _extract_run_and_desc(tsv_path.name)
        if run_key is None or desc is None:
            continue
        # Use a composite key so that a MEG recording and an EEG recording
        # with the same BIDS entities (sub/ses/task/run) are kept as separate
        # RunRecords and routed to the correct modality accumulator.
        full_key = f"{run_key}_{bids_suffix}"
        tmp_files[full_key][desc].append(tsv_path)
        if full_key not in run_meta:
            meta = _parse_entities_from_run_key(run_key)
            meta.bids_suffix = bids_suffix
            run_meta[full_key] = meta

    run_records: Dict[str, RunRecord] = {}
    for full_key, desc_map in tmp_files.items():
        record = RunRecord(meta=run_meta[full_key], files={})
        for desc, paths in desc_map.items():
            # Keep only the newest file for a given run + desc.
            chosen = sorted(paths, key=lambda p: p.stat().st_mtime)[-1]
            record.files[desc] = chosen
        run_records[full_key] = record

    return run_records


def _read_tsv(path: Path) -> Optional[pd.DataFrame]:
    """Read TSV robustly for heterogeneous derivatives.

    We intentionally load everything as strings and convert per-column later.
    This avoids mixed-type parser instability on large datasets and keeps
    parsing behavior deterministic across modules.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
        # Drop stale index column from old derivatives saved with index=True
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
        return df
    except Exception:
        return None


def _col_sort_key(name: str) -> Tuple[int, str]:
    tail = name.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail), name
    return 10**9, name


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _as_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _robust_normalize_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(arr.shape, np.nan, dtype=float)
    core = arr[finite]
    center = float(np.nanmedian(core))
    q25, q75 = np.nanquantile(core, [0.25, 0.75])
    scale = float(q75 - q25)
    if (not np.isfinite(scale)) or scale <= np.finfo(float).eps:
        std = float(np.nanstd(core))
        scale = std if std > np.finfo(float).eps else 1.0
    out = np.full(arr.shape, np.nan, dtype=float)
    out[finite] = (arr[finite] - center) / scale
    return out


def _normalize_profile_quantiles(profile: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    fields = ["q05", "q25", "q50", "q75", "q95", "top5"]
    all_values = []
    for field in fields:
        if field in profile:
            all_values.append(np.asarray(profile[field], dtype=float).reshape(-1))
    if not all_values:
        return dict(profile)

    pool = _finite_array(np.concatenate(all_values))
    if pool.size == 0:
        return dict(profile)
    center = float(np.nanmedian(pool))
    q25, q75 = np.nanquantile(pool, [0.25, 0.75])
    scale = float(q75 - q25)
    if (not np.isfinite(scale)) or scale <= np.finfo(float).eps:
        std = float(np.nanstd(pool))
        scale = std if std > np.finfo(float).eps else 1.0

    out = {}
    for key, vals in profile.items():
        arr = np.asarray(vals, dtype=float)
        if key in fields:
            out[key] = (arr - center) / scale
        else:
            out[key] = arr
    return out


def _safe_channel_subset(df: pd.DataFrame, ch_type: str) -> pd.DataFrame:
    if "Type" not in df.columns:
        return pd.DataFrame()
    ch_series = df["Type"].astype(str).str.lower()
    return df.loc[ch_series == ch_type].copy()


def _load_sensor_layout(path: Path) -> Dict[str, SensorLayout]:
    df = _read_tsv(path)
    if df is None:
        return {}

    x_col = _find_column(df, ["Sensor_location_0", "sensor_location_0", "loc_x", "x"])
    y_col = _find_column(df, ["Sensor_location_1", "sensor_location_1", "loc_y", "y"])
    z_col = _find_column(df, ["Sensor_location_2", "sensor_location_2", "loc_z", "z"])
    name_col = _find_column(df, ["Name", "name", "Channel", "channel"])
    if x_col is None or y_col is None:
        return {}

    out: Dict[str, SensorLayout] = {}
    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue
        x = pd.to_numeric(df_ch[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df_ch[y_col], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(df_ch[z_col], errors="coerce").to_numpy(dtype=float) if z_col is not None else None
        if name_col is not None:
            names = df_ch[name_col].fillna("").astype(str).tolist()
        else:
            names = [f"{ch_type}_{idx}" for idx in range(len(df_ch))]
        out[ch_type] = SensorLayout(x=x, y=y, names=names, z=z)
    return out


def _store_topomap_payload_if_missing(
    payloads_by_condition: Dict[str, TopomapPayload],
    condition_label: str,
    layout: Optional[SensorLayout],
    values: np.ndarray,
) -> None:
    if condition_label in payloads_by_condition or layout is None:
        return
    vals = np.asarray(values, dtype=float).reshape(-1)
    n = min(vals.size, layout.x.size, layout.y.size, len(layout.names))
    if n < 3:
        return
    payloads_by_condition[condition_label] = TopomapPayload(
        layout=SensorLayout(
            x=np.asarray(layout.x[:n], dtype=float),
            y=np.asarray(layout.y[:n], dtype=float),
            names=list(layout.names[:n]),
            z=(np.asarray(layout.z[:n], dtype=float) if layout.z is not None else None),
        ),
        values=vals[:n],
    )


def _update_topomap_payload_mean(
    payloads_by_condition: Dict[str, TopomapPayload],
    counts_by_condition: Dict[str, np.ndarray],
    condition_label: str,
    layout: Optional[SensorLayout],
    values: np.ndarray,
) -> None:
    """Update condition-level topomap values as an online mean.

    Topomap vectors can contain NaNs and runs can appear in different order.
    We keep per-channel counts so each channel mean only uses finite inputs.
    """
    if layout is None:
        return
    vals = np.asarray(values, dtype=float).reshape(-1)
    n = min(vals.size, layout.x.size, layout.y.size, len(layout.names))
    if n < 3:
        return
    vals = vals[:n]
    finite = np.isfinite(vals).astype(float)

    if condition_label not in payloads_by_condition:
        init_vals = np.where(np.isfinite(vals), vals, np.nan)
        payloads_by_condition[condition_label] = TopomapPayload(
            layout=SensorLayout(
                x=np.asarray(layout.x[:n], dtype=float),
                y=np.asarray(layout.y[:n], dtype=float),
                names=list(layout.names[:n]),
                z=(np.asarray(layout.z[:n], dtype=float) if layout.z is not None else None),
            ),
            values=init_vals,
        )
        counts_by_condition[condition_label] = finite
        return

    payload = payloads_by_condition[condition_label]
    old_vals = np.asarray(payload.values, dtype=float).reshape(-1)
    old_cnt = np.asarray(counts_by_condition.get(condition_label, np.zeros_like(old_vals)), dtype=float).reshape(-1)
    m = min(old_vals.size, vals.size, old_cnt.size)
    if m < 3:
        return

    old_vals = old_vals[:m]
    old_cnt = old_cnt[:m]
    new_vals = vals[:m]
    new_finite = np.isfinite(new_vals)
    num = np.where(np.isfinite(old_vals), old_vals, 0.0) * old_cnt + np.where(new_finite, new_vals, 0.0)
    den = old_cnt + new_finite.astype(float)
    out = np.full(m, np.nan, dtype=float)
    np.divide(num, np.maximum(den, np.finfo(float).eps), out=out, where=den > 0)

    payloads_by_condition[condition_label] = TopomapPayload(
        layout=SensorLayout(
            x=np.asarray(payload.layout.x[:m], dtype=float),
            y=np.asarray(payload.layout.y[:m], dtype=float),
            names=list(payload.layout.names[:m]),
            z=(np.asarray(payload.layout.z[:m], dtype=float) if payload.layout.z is not None else None),
        ),
        values=out,
    )
    counts_by_condition[condition_label] = den


def _load_std_or_ptp_matrix(path: Path, epoch_prefix: str, overall_column: str) -> Dict[str, np.ndarray]:
    df = _read_tsv(path)
    if df is None:
        return {}

    epoch_cols = sorted([c for c in df.columns if c.startswith(epoch_prefix)], key=_col_sort_key)
    out: Dict[str, np.ndarray] = {}

    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue

        if epoch_cols:
            matrix = df_ch[epoch_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        elif overall_column in df_ch.columns:
            matrix = (
                df_ch[[overall_column]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            )
        else:
            continue

        if matrix.size > 0:
            out[ch_type] = matrix

    return out


def _load_psd_matrix(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    df = _read_tsv(path)
    if df is None:
        return {}

    freq_cols = []
    for col in df.columns:
        if col.startswith("PSD_Hz_"):
            try:
                freq = float(col.replace("PSD_Hz_", ""))
                freq_cols.append((freq, col))
            except ValueError:
                continue
    if not freq_cols:
        return {}

    freq_cols.sort(key=lambda x: x[0])
    freqs = np.asarray([fc[0] for fc in freq_cols], dtype=float)
    col_names = [fc[1] for fc in freq_cols]

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue
        matrix = df_ch[col_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if matrix.size > 0:
            out[ch_type] = (freqs, matrix)
    return out


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _load_correlation_scalar(path: Path, metric: str) -> Dict[str, np.ndarray]:
    corr_values = _load_correlation_values(path, metric)
    out: Dict[str, np.ndarray] = {}
    for ch_type, values in corr_values.items():
        vals = _finite_array(values)
        if vals.size:
            out[ch_type] = vals
    return out


def _load_correlation_values(path: Path, metric: str) -> Dict[str, np.ndarray]:
    df = _read_tsv(path)
    if df is None:
        return {}

    metric = metric.lower()
    col = _find_column(df, [f"{metric}_corr_coeff", f"{metric}_similarity_score"])
    if col is None:
        return {}

    out: Dict[str, np.ndarray] = {}
    for ch_type in CH_TYPES:
        df_ch = _safe_channel_subset(df, ch_type)
        if df_ch.empty:
            continue
        vals = pd.to_numeric(df_ch[col], errors="coerce").to_numpy(dtype=float)
        out[ch_type] = np.abs(vals)
    return out


def _parse_muscle_ch_type(value: str) -> Optional[str]:
    value = str(value).lower()
    if "grad" in value:
        return "grad"
    if "eeg" in value or "ecog" in value or "seeg" in value or "ieeg" in value:
        return "eeg"
    if "mag" in value:
        return "mag"
    return None


def _load_muscle_scores(path: Path) -> Dict[str, np.ndarray]:
    df = _read_tsv(path)
    if df is None or "scores_muscle" not in df.columns:
        return {}

    scores = pd.to_numeric(df["scores_muscle"], errors="coerce").to_numpy(dtype=float)
    scores = _finite_array(scores)
    if not scores.size:
        return {}

    ch_type = None
    if "ch_type" in df.columns:
        ch_non_na = df["ch_type"].dropna()
        if not ch_non_na.empty:
            ch_type = _parse_muscle_ch_type(str(ch_non_na.iloc[0]))
    if ch_type is None:
        # Infer from file path: check if file is under an eeg/ subfolder
        path_str = str(path).replace("\\", "/").lower()
        if "/eeg/" in path_str or "_eeg." in path_str or "_eeg_" in path_str:
            ch_type = "eeg"
        else:
            ch_type = "mag"  # Default to mag for MEG data

    return {ch_type: scores}


def _condition_label(meta: RunMeta) -> str:
    parts = []
    if meta.task != "n/a":
        parts.append(meta.task)
    if not parts:
        return "all recordings"
    return ", ".join(parts)


def _profile_quantiles(matrix: np.ndarray) -> Dict[str, np.ndarray]:
    q = np.nanquantile(matrix, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    mean = np.nanmean(matrix, axis=0)
    top5 = np.full(matrix.shape[1], np.nan, dtype=float)
    for idx in range(matrix.shape[1]):
        col = _finite_array(matrix[:, idx])
        if col.size == 0:
            continue
        k = min(5, col.size)
        top5[idx] = float(np.mean(np.sort(col)[-k:]))
    return {"q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4], "mean": mean, "top5": top5}


def _compute_mains_ratio(psd_matrix: np.ndarray, freqs: np.ndarray, dropna: bool = True) -> np.ndarray:
    if psd_matrix.size == 0 or freqs.size == 0:
        return np.array([], dtype=float)

    fmax = float(np.nanmax(freqs))
    if not np.isfinite(fmax) or fmax <= 0:
        return np.array([], dtype=float)

    targets = []
    for base in (50.0, 60.0):
        harmonic = base
        while harmonic <= fmax + 1.0:
            targets.append(harmonic)
            harmonic += base

    if not targets:
        return np.array([], dtype=float)

    mask = np.zeros(freqs.shape, dtype=bool)
    for target in targets:
        mask |= np.abs(freqs - target) <= 1.0
    if not np.any(mask):
        return np.array([], dtype=float)

    total = np.nansum(psd_matrix, axis=1)
    mains = np.nansum(psd_matrix[:, mask], axis=1)
    ratio = mains / np.maximum(total, np.finfo(float).eps)
    return _finite_array(ratio) if dropna else ratio


def _compute_mains_and_harmonics_ratio(psd_matrix: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    psd_matrix = np.asarray(psd_matrix, dtype=float)
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    if psd_matrix.size == 0 or freqs.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ref_profile = np.nanmedian(psd_matrix, axis=0)
    mains_base = _infer_mains_frequency(freqs, ref_profile)
    if mains_base is None:
        fallback = _compute_mains_ratio(psd_matrix, freqs, dropna=False)
        return fallback, np.full(fallback.shape, np.nan, dtype=float)

    mains_mask = np.abs(freqs - mains_base) <= 1.0
    harmonic_mask = np.zeros(freqs.shape, dtype=bool)
    harmonic = mains_base * 2.0
    fmax = float(np.nanmax(freqs))
    while harmonic <= fmax + 0.5:
        harmonic_mask |= np.abs(freqs - harmonic) <= 1.0
        harmonic += mains_base

    total = np.nansum(psd_matrix, axis=1)
    denom = np.maximum(total, np.finfo(float).eps)
    mains_ratio = np.nansum(psd_matrix[:, mains_mask], axis=1) / denom
    harmonics_ratio = np.nansum(psd_matrix[:, harmonic_mask], axis=1) / denom
    return mains_ratio, harmonics_ratio


def _infer_mains_frequency(freqs: np.ndarray, psd_q50: np.ndarray) -> Optional[float]:
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    psd_q50 = np.asarray(psd_q50, dtype=float).reshape(-1)
    if freqs.size == 0 or psd_q50.size == 0:
        return None

    n = min(freqs.size, psd_q50.size)
    freqs = freqs[:n]
    psd_q50 = psd_q50[:n]
    finite = np.isfinite(freqs) & np.isfinite(psd_q50)
    if not np.any(finite):
        return None
    freqs = freqs[finite]
    psd_q50 = psd_q50[finite]

    candidate_scores: Dict[float, float] = {}
    for base in (50.0, 60.0):
        band = (freqs >= base - 3.0) & (freqs <= base + 3.0)
        if np.any(band):
            candidate_scores[base] = float(np.nanmax(psd_q50[band]))
    if not candidate_scores:
        return None
    return max(candidate_scores.items(), key=lambda item: item[1])[0]


def _aggregate_window_profiles(profiles: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if not profiles:
        return None

    max_len = max(len(p["q50"]) for p in profiles)
    if max_len == 0:
        return None

    def _stack(field: str) -> np.ndarray:
        arr = np.full((len(profiles), max_len), np.nan, dtype=float)
        for i, prof in enumerate(profiles):
            vals = np.asarray(prof[field], dtype=float).reshape(-1)
            if vals.size == 0:
                continue
            arr[i, : len(vals)] = vals
        return np.nanmedian(arr, axis=0)

    agg = {
        "x": np.arange(max_len, dtype=int),
        "q05": _stack("q05"),
        "q25": _stack("q25"),
        "q50": _stack("q50"),
        "q75": _stack("q75"),
        "q95": _stack("q95"),
    }
    if any("mean" in p for p in profiles):
        agg["mean"] = _stack("mean")
    if any("top5" in p for p in profiles):
        agg["top5"] = _stack("top5")
    return agg


def _aggregate_psd_profiles(profiles: List[Dict[str, np.ndarray]]) -> Optional[Dict[str, np.ndarray]]:
    if not profiles:
        return None

    ref = max(profiles, key=lambda p: len(p["freqs"]))
    ref_freqs = _finite_array(ref["freqs"])
    if ref_freqs.size == 0:
        return None

    def _to_ref_grid(freqs: np.ndarray, values: np.ndarray) -> np.ndarray:
        freqs = _finite_array(freqs)
        values = np.asarray(values, dtype=float).reshape(-1)
        if freqs.size == 0 or values.size == 0:
            return np.full(ref_freqs.shape, np.nan, dtype=float)

        # Keep only finite pairs and ensure increasing x for interpolation.
        pair_mask = np.isfinite(freqs) & np.isfinite(values[: len(freqs)])
        x = freqs[pair_mask]
        y = values[: len(freqs)][pair_mask]
        if x.size == 0:
            return np.full(ref_freqs.shape, np.nan, dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if x.size == ref_freqs.size and np.allclose(x, ref_freqs):
            return y
        return np.interp(ref_freqs, x, y, left=np.nan, right=np.nan)

    def _stack(field: str) -> np.ndarray:
        arr = np.full((len(profiles), ref_freqs.size), np.nan, dtype=float)
        for i, prof in enumerate(profiles):
            arr[i, :] = _to_ref_grid(prof["freqs"], prof[field])
        return np.nanmedian(arr, axis=0)

    out = {
        "x": ref_freqs,
        "q05": _stack("q05"),
        "q25": _stack("q25"),
        "q50": _stack("q50"),
        "q75": _stack("q75"),
        "q95": _stack("q95"),
    }
    if any("top5" in p for p in profiles):
        out["top5"] = _stack("top5")
    return out


def _aggregate_muscle_profiles(profiles: List[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
    if not profiles:
        return None

    max_len = max(len(p) for p in profiles)
    if max_len == 0:
        return None

    arr = np.full((len(profiles), max_len), np.nan, dtype=float)
    for i, profile in enumerate(profiles):
        vals = np.asarray(profile, dtype=float).reshape(-1)
        if vals.size == 0:
            continue
        arr[i, : len(vals)] = vals

    return {
        "x": np.arange(max_len, dtype=int),
        "q05": np.nanquantile(arr, 0.05, axis=0),
        "q25": np.nanquantile(arr, 0.25, axis=0),
        "q50": np.nanquantile(arr, 0.50, axis=0),
        "q75": np.nanquantile(arr, 0.75, axis=0),
        "q95": np.nanquantile(arr, 0.95, axis=0),
    }


def _update_representative_matrix(
    matrix_by_condition: Dict[str, np.ndarray],
    score_by_condition: Dict[str, float],
    score_history: Dict[str, List[float]],
    condition_label: str,
    matrix: np.ndarray,
    run_score: float,
) -> None:
    if matrix.size == 0 or not np.isfinite(run_score):
        return

    history = score_history[condition_label]
    history.append(float(run_score))
    finite_history = _finite_array(history)
    if finite_history.size == 0:
        return
    target = float(np.nanmedian(finite_history))

    if condition_label not in matrix_by_condition:
        matrix_by_condition[condition_label] = np.asarray(matrix, dtype=float)
        score_by_condition[condition_label] = float(run_score)
        return

    prev_score = score_by_condition.get(condition_label, np.nan)
    if (not np.isfinite(prev_score)) or abs(run_score - target) <= abs(prev_score - target):
        matrix_by_condition[condition_label] = np.asarray(matrix, dtype=float)
        score_by_condition[condition_label] = float(run_score)


def _update_upper_tail_matrix(
    matrix_by_condition: Dict[str, np.ndarray],
    score_by_condition: Dict[str, float],
    condition_label: str,
    matrix: np.ndarray,
    run_score: float,
) -> None:
    """Keep the matrix with the largest run-level upper-tail score per condition."""
    if matrix.size == 0 or not np.isfinite(run_score):
        return
    prev = score_by_condition.get(condition_label, float("-inf"))
    if condition_label not in matrix_by_condition or (not np.isfinite(prev)) or run_score >= prev:
        matrix_by_condition[condition_label] = np.asarray(matrix, dtype=float)
        score_by_condition[condition_label] = float(run_score)


def _accumulate_matrix_mean(
    sum_by_condition: Dict[str, np.ndarray],
    count_by_condition: Dict[str, np.ndarray],
    condition_label: str,
    matrix: np.ndarray,
) -> None:
    """Accumulate sum/count matrices with zero padding for ragged epoch sizes.

    Different recordings can have different numbers of channels/epochs. We pad
    to the current max shape and maintain sum/count separately so downstream
    means are unbiased and missing cells remain undefined.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return
    vals = np.nan_to_num(arr, nan=0.0)
    cnt = np.isfinite(arr).astype(float)

    if condition_label not in sum_by_condition:
        sum_by_condition[condition_label] = vals
        count_by_condition[condition_label] = cnt
        return

    prev_sum = sum_by_condition[condition_label]
    prev_cnt = count_by_condition[condition_label]
    n_rows = max(prev_sum.shape[0], vals.shape[0])
    n_cols = max(prev_sum.shape[1], vals.shape[1])

    def _pad(a: np.ndarray) -> np.ndarray:
        if a.shape == (n_rows, n_cols):
            return a
        return np.pad(a, ((0, n_rows - a.shape[0]), (0, n_cols - a.shape[1])), mode="constant", constant_values=0.0)

    sum_by_condition[condition_label] = _pad(prev_sum) + _pad(vals)
    count_by_condition[condition_label] = _pad(prev_cnt) + _pad(cnt)


def _mean_matrix_from_sum_count(sum_matrix: np.ndarray, count_matrix: np.ndarray) -> np.ndarray:
    s = np.asarray(sum_matrix, dtype=float)
    c = np.asarray(count_matrix, dtype=float)
    if s.shape != c.shape or s.size == 0:
        return np.array([], dtype=float)
    out = np.full(s.shape, np.nan, dtype=float)
    np.divide(s, np.maximum(c, np.finfo(float).eps), out=out, where=c > 0)
    return out


def _pad_matrix_with_nan(matrix: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        return np.full((n_rows, n_cols), np.nan, dtype=float)
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    r = min(n_rows, arr.shape[0])
    c = min(n_cols, arr.shape[1])
    out[:r, :c] = arr[:r, :c]
    return out


def _aggregate_heatmap_variants_from_runs(
    run_matrices_by_condition: Dict[str, List[np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate one condition-level matrix per summary variant.

    Output structure is:
    ``{condition: {"Median": matrix, "Mean": matrix, "Upper tail": matrix}}``.
    All variants for a condition share the same shape, so epoch axis stays
    stable while users switch between variants.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for cond, matrices in run_matrices_by_condition.items():
        mats = [np.asarray(m, dtype=float) for m in matrices if np.asarray(m).ndim == 2 and np.asarray(m).size > 0]
        if not mats:
            continue
        n_rows = max(m.shape[0] for m in mats)
        n_cols = max(m.shape[1] for m in mats)
        stack = np.full((len(mats), n_rows, n_cols), np.nan, dtype=float)
        for idx, mat in enumerate(mats):
            stack[idx, :, :] = _pad_matrix_with_nan(mat, n_rows, n_cols)
        out[cond] = {
            "Median": np.nanmedian(stack, axis=0),
            "Mean": np.nanmean(stack, axis=0),
            "Upper tail": np.nanquantile(stack, 0.95, axis=0),
        }
    return out


def _epoch_consistency_notes(
    run_matrices_by_condition: Dict[str, List[np.ndarray]],
) -> Dict[str, str]:
    """Build per-condition note describing whether epoch counts are aligned."""
    notes: Dict[str, str] = {}
    for cond, matrices in run_matrices_by_condition.items():
        counts = sorted({int(np.asarray(m).shape[1]) for m in matrices if np.asarray(m).ndim == 2 and np.asarray(m).size > 0})
        if not counts:
            continue
        if len(counts) == 1:
            notes[cond] = (
                f"Epoch counts are aligned across recordings in this task "
                f"(n_epochs={counts[0]})."
            )
        else:
            span = ", ".join(str(c) for c in counts)
            notes[cond] = (
                f"Epoch counts vary across recordings in this task ({span}). "
                "Matrices are padded during aggregation; padded cells are excluded "
                "from matrix statistics."
            )
    return notes


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = _finite_array(values)
    if arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = np.sort(arr)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def _downsample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    idx = np.linspace(0, length - 1, num=max_points, dtype=int)
    return np.unique(idx)


def _make_ecdf_figure(values_by_condition: Dict[str, List[float]], title: str, x_title: str) -> Optional[go.Figure]:
    fig = go.Figure()
    for label in sorted(values_by_condition):
        x, y = _ecdf(values_by_condition[label])
        if x.size == 0:
            continue
        keep = _downsample_indices(x.size, MAX_POINTS_ECDF)
        x = x[keep]
        y = y[keep]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{label} (n={x.size})",
                line={"width": 2},
            )
        )

    if not fig.data:
        return None

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title="ECDF",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}},
        margin={"l": 55, "r": 30, "t": 50, "b": 50},
    )
    return fig


def plot_violin_channel_distribution(
    values_by_condition: Dict[str, List[float]],
    title: str,
    y_title: str,
) -> Optional[go.Figure]:
    fig = go.Figure()
    for label in sorted(values_by_condition):
        vals = _finite_array(values_by_condition[label])
        if vals.size == 0:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        fig.add_trace(
            go.Violin(
                x=np.repeat(label, vals.size),
                y=vals,
                name=f"{label} (n={vals.size})",
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"width": 1.1},
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task",
        yaxis_title=y_title,
        template="plotly_white",
        violinmode="group",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}},
        margin={"l": 55, "r": 30, "t": 50, "b": 50},
    )
    return fig


def plot_quantile_band_timecourse(
    quantiles: Dict[str, np.ndarray],
    title: str,
    x_title: str,
    y_title: str,
) -> Optional[go.Figure]:
    if quantiles is None:
        return None

    x = np.asarray(quantiles["x"], dtype=float)
    q05 = np.asarray(quantiles["q05"], dtype=float)
    q25 = np.asarray(quantiles["q25"], dtype=float)
    q50 = np.asarray(quantiles["q50"], dtype=float)
    q75 = np.asarray(quantiles["q75"], dtype=float)
    q95 = np.asarray(quantiles["q95"], dtype=float)
    top5 = np.asarray(quantiles["top5"], dtype=float) if "top5" in quantiles else None

    valid = np.isfinite(x) & (
        np.isfinite(q05) | np.isfinite(q25) | np.isfinite(q50) | np.isfinite(q75) | np.isfinite(q95)
    )
    if not np.any(valid):
        return None
    x = x[valid]
    q05 = q05[valid]
    q25 = q25[valid]
    q50 = q50[valid]
    q75 = q75[valid]
    q95 = q95[valid]
    if top5 is not None:
        top5 = top5[valid]

    keep = _downsample_indices(x.size, MAX_POINTS_PROFILE)
    x = x[keep]
    q05 = q05[keep]
    q25 = q25[keep]
    q50 = q50[keep]
    q75 = q75[keep]
    q95 = q95[keep]
    if top5 is not None:
        top5 = top5[keep]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=q95, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"},
            showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q05,
            mode="lines",
            line={"width": 0, "color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor="rgba(88, 166, 255, 0.20)",
            name="Middle 90% of channels",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=q75, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"},
            showlegend=False, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q25,
            mode="lines",
            line={"width": 0, "color": "rgba(0,0,0,0)"},
            fill="tonexty",
            fillcolor="rgba(30, 136, 229, 0.28)",
            name="Middle 50% of channels (IQR)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=q50,
            mode="lines",
            line={"width": 2.3, "color": "#164B84"},
            name="Median across channels",
        )
    )
    if top5 is not None and np.any(np.isfinite(top5)):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=top5,
                mode="lines",
                line={"width": 1.8, "color": "#D1495B", "dash": "dash"},
                name="Mean of top 5 channels",
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def _make_quantile_band_figure(
    quantiles: Dict[str, np.ndarray],
    title: str,
    x_title: str,
    y_title: str,
) -> Optional[go.Figure]:
    return plot_quantile_band_timecourse(quantiles, title, x_title, y_title)


def plot_psd_median_band(
    psd_quantiles: Dict[str, np.ndarray],
    title: str,
) -> Optional[go.Figure]:
    if psd_quantiles is None:
        return None
    fig = plot_quantile_band_timecourse(
        psd_quantiles,
        title=title,
        x_title="Frequency (Hz)",
        y_title="Relative power",
    )
    if fig is None:
        return None

    freqs = np.asarray(psd_quantiles["x"], dtype=float)
    q50 = np.asarray(psd_quantiles["q50"], dtype=float)
    mains = _infer_mains_frequency(freqs, q50)
    finite_freqs = _finite_array(freqs)
    if mains is not None and finite_freqs.size:
        max_freq = float(np.nanmax(finite_freqs))
        harmonic = mains
        first = True
        while harmonic <= max_freq + 0.5:
            fig.add_vline(
                x=float(harmonic),
                line_width=1.2,
                line_dash="dot",
                line_color="rgba(108,117,125,0.60)",
            )
            if first:
                fig.add_annotation(
                    x=float(harmonic),
                    y=1.02,
                    yref="paper",
                    showarrow=False,
                    text=f"Mains frequency ({int(round(mains))} Hz)",
                    font={"size": 11, "color": "#495057"},
                    xanchor="left",
                )
                first = False
            harmonic += mains

    positive = np.isfinite(q50) & (q50 > 0)
    if np.any(positive):
        fig.update_yaxes(type="log")
    return fig


def _robust_bounds(values: np.ndarray) -> Optional[Tuple[float, float]]:
    vals = _finite_array(values)
    if vals.size == 0:
        return None
    lo, hi = np.nanquantile(vals, [0.02, 0.98])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        hi = lo + np.finfo(float).eps
    return float(lo), float(hi)


def plot_heatmap_sorted_channels_windows(
    matrix: np.ndarray | Dict[str, np.ndarray],
    title: str,
    color_title: str,
    summary_mode: str = "median",
    channel_names: Optional[Sequence[str]] = None,
) -> Optional[go.Figure]:
    """Render a channel-by-epoch heatmap with top/right summary controls.

    Parameters
    ----------
    matrix
        Either one matrix (``channels x epochs``) or a dict with multiple
        condition-level variants keyed by labels such as ``Median``, ``Mean``,
        ``Upper tail``.
    """
    if isinstance(matrix, dict):
        raw_variants = {
            str(label): np.asarray(arr, dtype=float)
            for label, arr in matrix.items()
            if np.asarray(arr).ndim == 2 and np.asarray(arr).size > 0
        }
    else:
        arr = np.asarray(matrix, dtype=float)
        raw_variants = {"Median": arr} if arr.ndim == 2 and arr.size > 0 else {}
    if not raw_variants:
        return None

    ordered_labels = [lbl for lbl in ("Median", "Mean", "Upper tail") if lbl in raw_variants]
    ordered_labels.extend([lbl for lbl in raw_variants.keys() if lbl not in ordered_labels])
    base_label = "Median" if "Median" in raw_variants else ordered_labels[0]

    n_rows = max(arr.shape[0] for arr in raw_variants.values())
    n_cols = max(arr.shape[1] for arr in raw_variants.values())
    padded = {label: _pad_matrix_with_nan(arr, n_rows, n_cols) for label, arr in raw_variants.items()}
    if channel_names is not None:
        supplied = [str(n) for n in channel_names]
    else:
        supplied = []
    if len(supplied) < n_rows:
        supplied = supplied + [f"channel-{idx:04d}" for idx in range(len(supplied), n_rows)]
    base_channel_names = np.asarray(supplied[:n_rows], dtype=object)

    base = padded[base_label]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if summary_mode == "upper_tail":
            base_summary = np.nanquantile(base, 0.95, axis=1)
        else:
            base_summary = np.nanmedian(base, axis=1)
    order = np.argsort(np.nan_to_num(base_summary, nan=-np.inf))[::-1]
    natural_order = np.arange(n_rows, dtype=int)

    row_keep = _downsample_indices(n_rows, MAX_HEATMAP_CHANNELS)
    col_keep = _downsample_indices(n_cols, MAX_HEATMAP_WINDOWS)

    def _payload_for(arr: np.ndarray, ch_order: np.ndarray) -> Dict[str, np.ndarray]:
        arr_sorted = arr[ch_order, :]
        z = arr_sorted[row_keep][:, col_keep]
        x = np.arange(n_cols, dtype=int)[col_keep]
        y = np.arange(n_rows, dtype=int)[row_keep]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            top_q05 = np.nanquantile(arr_sorted[:, col_keep], 0.05, axis=0)
            top_q25 = np.nanquantile(arr_sorted[:, col_keep], 0.25, axis=0)
            top_q50 = np.nanmedian(arr_sorted[:, col_keep], axis=0)
            top_q75 = np.nanquantile(arr_sorted[:, col_keep], 0.75, axis=0)
            top_q95 = np.nanquantile(arr_sorted[:, col_keep], 0.95, axis=0)
            top_mean = np.nanmean(arr_sorted[:, col_keep], axis=0)

            right_q05 = np.nanquantile(arr_sorted[row_keep, :], 0.05, axis=1)
            right_q25 = np.nanquantile(arr_sorted[row_keep, :], 0.25, axis=1)
            right_q50 = np.nanmedian(arr_sorted[row_keep, :], axis=1)
            right_q75 = np.nanquantile(arr_sorted[row_keep, :], 0.75, axis=1)
            right_q95 = np.nanquantile(arr_sorted[row_keep, :], 0.95, axis=1)
            right_mean = np.nanmean(arr_sorted[row_keep, :], axis=1)

        sorted_names = base_channel_names[ch_order]
        row_names = sorted_names[row_keep]
        heat_custom = np.tile(np.asarray(row_names, dtype=object)[:, None], (1, x.size))

        return {
            "x": x,
            "y": y,
            "z": z,
            "heat_custom": heat_custom,
            "right_custom": np.asarray(row_names, dtype=object),
            "top_q05": top_q05,
            "top_q25": top_q25,
            "top_q50": top_q50,
            "top_q75": top_q75,
            "top_q95": top_q95,
            "top_mean": top_mean,
            "right_q05": right_q05,
            "right_q25": right_q25,
            "right_q50": right_q50,
            "right_q75": right_q75,
            "right_q95": right_q95,
            "right_mean": right_mean,
        }

    payload_by_label = {label: _payload_for(padded[label], order) for label in ordered_labels}
    # Natural-order payloads for the channel order toggle.
    nat_payload_by_label = {label: _payload_for(padded[label], natural_order) for label in ordered_labels}
    z_all = np.concatenate(
        [vals["z"][np.isfinite(vals["z"])] for vals in payload_by_label.values() if np.any(np.isfinite(vals["z"]))],
        axis=0,
    ) if any(np.any(np.isfinite(vals["z"])) for vals in payload_by_label.values()) else np.array([], dtype=float)
    bounds = _robust_bounds(z_all)
    if bounds is None:
        return None
    zmin, zmax = bounds

    first = payload_by_label[ordered_labels[0]]
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.24, 0.76],
        column_widths=[0.86, 0.14],
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "xy"}]],
        vertical_spacing=0.18,
        horizontal_spacing=0.04,
    )

    # Top panel (epoch profile with quantile bands).
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_q95"], mode="lines", line=dict(width=0), name="Middle 90% of channels", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_q05"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.15)", name="Middle 90% of channels", hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_q75"], mode="lines", line=dict(width=0), name="Middle 50% of channels", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_q25"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.34)", name="Middle 50% of channels", hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_q50"], mode="lines", line=dict(color="#0B3D91", width=2.4), name="Median across channels", hovertemplate="Epoch: %{x}<br>Median: %{y:.3g}<extra></extra>", visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_mean"], mode="lines", line=dict(color="#D35400", width=2.4), name="Mean across channels", hovertemplate="Epoch: %{x}<br>Mean: %{y:.3g}<extra></extra>", visible=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["top_q95"], mode="lines", line=dict(color="#8E44AD", width=2.4), name="Upper tail (q95) across channels", hovertemplate="Epoch: %{x}<br>Upper tail: %{y:.3g}<extra></extra>", visible=False), row=1, col=1)

    # Heatmap panel.
    fig.add_trace(
        go.Heatmap(
            z=first["z"],
            x=first["x"],
            y=first["y"],
            customdata=first["heat_custom"],
            coloraxis="coloraxis",
            hovertemplate="Channel: %{customdata}<br>Epoch: %{x}<br>Value: %{z:.3g}<extra></extra>",
            name="Channel x epoch heatmap",
            showscale=False,
        ),
        row=2,
        col=1,
    )

    # Right panel (channel profile with quantile bands).
    fig.add_trace(go.Scatter(x=first["right_q95"], y=first["y"], mode="lines", line=dict(width=0), name="Middle 90% of epochs", showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=first["right_q05"], y=first["y"], mode="lines", line=dict(width=0), fill="tonextx", fillcolor="rgba(31,119,180,0.15)", name="Middle 90% of epochs", showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=first["right_q75"], y=first["y"], mode="lines", line=dict(width=0), name="Middle 50% of epochs", showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=first["right_q25"], y=first["y"], mode="lines", line=dict(width=0), fill="tonextx", fillcolor="rgba(31,119,180,0.34)", name="Middle 50% of epochs", showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=first["right_q50"], y=first["y"], customdata=first["right_custom"], mode="lines", line=dict(color="#17A589", width=2.1), hovertemplate="Channel: %{customdata}<br>Median across epochs: %{x:.3g}<extra></extra>", showlegend=False, visible=True), row=2, col=2)
    fig.add_trace(go.Scatter(x=first["right_mean"], y=first["y"], customdata=first["right_custom"], mode="lines", line=dict(color="#D35400", width=2.1), hovertemplate="Channel: %{customdata}<br>Mean across epochs: %{x:.3g}<extra></extra>", showlegend=False, visible=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=first["right_q95"], y=first["y"], customdata=first["right_custom"], mode="lines", line=dict(color="#8E44AD", width=2.1), hovertemplate="Channel: %{customdata}<br>Upper tail (q95) across epochs: %{x:.3g}<extra></extra>", showlegend=False, visible=False), row=2, col=2)

    trace_indices = list(range(15))

    def _variant_restyle_args(payload: Dict[str, np.ndarray]) -> Dict[str, List[object]]:
        """Build one restyle payload for all heatmap/top/right traces.

        Using direct restyle updates avoids intermittent redraw issues from
        frame animation in very large HTML reports.
        """
        none = None
        return {
            "x": [
                payload["x"],  # top q95
                payload["x"],  # top q05
                payload["x"],  # top q75
                payload["x"],  # top q25
                payload["x"],  # top median
                payload["x"],  # top mean
                payload["x"],  # top upper-tail
                payload["x"],  # heatmap
                payload["right_q95"],  # right q95
                payload["right_q05"],  # right q05
                payload["right_q75"],  # right q75
                payload["right_q25"],  # right q25
                payload["right_q50"],  # right median
                payload["right_mean"],  # right mean
                payload["right_q95"],  # right upper-tail
            ],
            "y": [
                payload["top_q95"],
                payload["top_q05"],
                payload["top_q75"],
                payload["top_q25"],
                payload["top_q50"],
                payload["top_mean"],
                payload["top_q95"],
                payload["y"],
                payload["y"],
                payload["y"],
                payload["y"],
                payload["y"],
                payload["y"],
                payload["y"],
                payload["y"],
            ],
            "z": [
                none,
                none,
                none,
                none,
                none,
                none,
                none,
                payload["z"],
                none,
                none,
                none,
                none,
                none,
                none,
                none,
            ],
            "customdata": [
                none,
                none,
                none,
                none,
                none,
                none,
                none,
                payload["heat_custom"],
                none,
                none,
                none,
                none,
                payload["right_custom"],
                payload["right_custom"],
                payload["right_custom"],
            ],
        }

    metric_short = "STD" if "std" in color_title.lower() else ("PtP" if "ptp" in color_title.lower() else color_title)
    unit_short = "mixed pT" if "mixed" in color_title.lower() else ("pT" if "pt" in color_title.lower() else "")
    side_title = f"{metric_short} ({unit_short})" if unit_short else str(metric_short)
    colorbar_title = str(metric_short)

    menus = []
    if len(ordered_labels) > 1:
        menus.append(
            dict(
                type="buttons",
                direction="right",
                x=0.00,
                y=-0.18,
                xanchor="left",
                yanchor="top",
                showactive=True,
                bgcolor="#EAF4FF",
                bordercolor="#1F5D9C",
                borderwidth=2.0,
                font=dict(size=14, color="#0F3D6E"),
                pad=dict(r=16, t=10, l=14, b=10),
                buttons=[
                    dict(
                        label=f"   Heat: {label}   ",
                        method="restyle",
                        args=[
                            _variant_restyle_args(payload_by_label[label]),
                            trace_indices,
                        ],
                    )
                    for label in ordered_labels
                ],
            )
        )

    menus.append(
        dict(
            type="buttons",
            direction="right",
            x=0.00,
            y=-0.32,
            xanchor="left",
            yanchor="top",
            showactive=True,
            bgcolor="#F3F8FE",
            bordercolor="#2B6CB0",
            borderwidth=1.8,
            font=dict(size=14, color="#0F3D6E"),
            pad=dict(r=16, t=10, l=14, b=10),
            buttons=[
                dict(label="   Top: Median   ", method="restyle", args=[{"visible": [True, False, False]}, [4, 5, 6]]),
                dict(label="   Top: Mean   ", method="restyle", args=[{"visible": [False, True, False]}, [4, 5, 6]]),
                dict(label="   Top: Upper tail   ", method="restyle", args=[{"visible": [False, False, True]}, [4, 5, 6]]),
            ],
        )
    )
    menus.append(
        dict(
            name="Line thickness level",
            type="buttons",
            direction="right",
            x=0.00,
            y=-0.60,
            xanchor="left",
            yanchor="top",
            showactive=True,
            active=1,
            bgcolor="#F3F8FE",
            bordercolor="#2B6CB0",
            borderwidth=1.8,
            font=dict(size=14, color="#0F3D6E"),
            pad=dict(r=16, t=10, l=14, b=10),
            buttons=[
                dict(
                    label=f"   {level}   ",
                    method="restyle",
                    args=[{"line.width": [float(level)] * 6}, [4, 5, 6, 12, 13, 14]],
                )
                for level in range(1, 9)
            ],
        )
    )
    menus.append(
        dict(
            type="buttons",
            direction="right",
            x=0.00,
            y=-0.46,
            xanchor="left",
            yanchor="top",
            showactive=True,
            bgcolor="#F3F8FE",
            bordercolor="#2B6CB0",
            borderwidth=1.8,
            font=dict(size=14, color="#0F3D6E"),
            pad=dict(r=16, t=10, l=14, b=10),
            buttons=[
                dict(label="   Right: Median   ", method="restyle", args=[{"visible": [True, False, False]}, [12, 13, 14]]),
                dict(label="   Right: Mean   ", method="restyle", args=[{"visible": [False, True, False]}, [12, 13, 14]]),
                dict(label="   Right: Upper tail   ", method="restyle", args=[{"visible": [False, False, True]}, [12, 13, 14]]),
            ],
        )
    )

    # ── Channel order toggle (sorted vs natural) ─────────────────────────
    # The active (default) variant is whichever label is currently rendered.
    active_label = ordered_labels[0]
    sorted_full = _variant_restyle_args(payload_by_label[active_label])
    natural_full = _variant_restyle_args(nat_payload_by_label[active_label])
    menus.append(
        dict(
            type="buttons",
            direction="right",
            x=0.00,
            y=-0.74,
            xanchor="left",
            yanchor="top",
            showactive=True,
            bgcolor="#FFF8E1",
            bordercolor="#D4A017",
            borderwidth=1.8,
            font=dict(size=14, color="#7B6B1A"),
            pad=dict(r=16, t=10, l=14, b=10),
            buttons=[
                dict(
                    label="   Channels: Sorted (high→low)   ",
                    method="update",
                    args=[
                        sorted_full,
                        {"yaxis3.title.text": "Sorted channel index"},
                        trace_indices,
                    ],
                ),
                dict(
                    label="   Channels: Natural order   ",
                    method="update",
                    args=[
                        natural_full,
                        {"yaxis3.title.text": "Channel index (original)"},
                        trace_indices,
                    ],
                ),
            ],
        )
    )

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text=side_title, row=1, col=1, automargin=True, title_standoff=10)
    fig.update_xaxes(title_text="Epoch index", row=2, col=1)
    fig.update_yaxes(title_text="Sorted channel index", row=2, col=1, autorange="reversed", automargin=True, title_standoff=10)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(title_text=side_title, row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2, autorange="reversed")

    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.97, "xanchor": "center", "yanchor": "top"},
        template="plotly_white",
        margin={"l": 70, "r": 55, "t": 165, "b": 400},
        height=1020,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "xanchor": "left", "x": 0, "font": {"size": 11}},
        coloraxis={
            "colorscale": "Viridis",
            "cmin": float(zmin),
            "cmax": float(zmax),
            "colorbar": {"title": colorbar_title, "x": 1.02, "len": 0.86},
        },
        updatemenus=menus,
    )
    return fig


def plot_topomap_if_available(
    payload: Optional[TopomapPayload],
    title: str,
    color_title: str,
) -> Optional[go.Figure]:
    if payload is None:
        return None
    x = np.asarray(payload.layout.x, dtype=float).reshape(-1)
    y = np.asarray(payload.layout.y, dtype=float).reshape(-1)
    values = np.asarray(payload.values, dtype=float).reshape(-1)
    n = min(x.size, y.size, values.size, len(payload.layout.names))
    if n < 3:
        return None

    x = x[:n]
    y = y[:n]
    values = values[:n]
    names = np.asarray(payload.layout.names[:n], dtype=object)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if np.sum(mask) < 3:
        return None

    # Spread overlapping points slightly so Elekta triplets (1 mag + 2 grad) remain visible.
    x_plot = x[mask].copy()
    y_plot = y[mask].copy()
    v_plot = values[mask]
    n_plot = names[mask]
    xy_key = np.round(np.column_stack([x_plot, y_plot]), 6)
    unique, inv = np.unique(xy_key, axis=0, return_inverse=True)
    xrange = float(np.nanmax(x_plot) - np.nanmin(x_plot)) if x_plot.size else 1.0
    yrange = float(np.nanmax(y_plot) - np.nanmin(y_plot)) if y_plot.size else 1.0
    base_r = max(xrange, yrange) * 0.012
    if not np.isfinite(base_r) or base_r <= 0:
        base_r = 1e-3
    for k in range(unique.shape[0]):
        idx = np.where(inv == k)[0]
        if idx.size <= 1:
            continue
        angles = np.linspace(0.0, 2.0 * np.pi, num=idx.size, endpoint=False)
        x_plot[idx] += base_r * np.cos(angles)
        y_plot[idx] += base_r * np.sin(angles)

    fig = go.Figure(
        go.Scatter(
            x=x_plot,
            y=y_plot,
            mode="markers",
            text=n_plot,
            customdata=v_plot,
            hovertemplate="%{text}<br>value=%{customdata:.3g}<extra></extra>",
            marker={
                "size": 11,
                "color": v_plot,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": color_title},
                "line": {"width": 0.5, "color": "#2F3E46"},
            },
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        margin={"l": 40, "r": 40, "t": 70, "b": 35},
    )
    return fig


def _add_solid_cap_toggle_to_topomap_3d(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> None:
    """Attach an optional solid-cap overlay (Mesh3d) to a 3D topomap figure.

    The cap is hidden by default and can be enabled with a button so users can
    inspect channel occlusion similar to an EEG-cap-like shell.
    """
    if fig is None:
        return
    xyz = np.column_stack(
        [np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1), np.asarray(z, dtype=float).reshape(-1)]
    )
    mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[mask]
    if xyz.shape[0] < 4:
        return

    point_trace_idx = len(fig.data) - 1
    # Inset the shell slightly so markers remain visually outside the cap.
    center = np.nanmean(xyz, axis=0, keepdims=True)
    inset_factor = 0.965
    cap_xyz = center + (xyz - center) * inset_factor

    fig.add_trace(
        go.Mesh3d(
            x=cap_xyz[:, 0],
            y=cap_xyz[:, 1],
            z=cap_xyz[:, 2],
            alphahull=0,
            color="#B0BEC5",
            opacity=1.0,
            hoverinfo="skip",
            showscale=False,
            name="Solid cap",
            visible=False,
            flatshading=True,
            lighting=dict(ambient=0.45, diffuse=0.55, specular=0.08, roughness=0.85, fresnel=0.03),
            lightposition=dict(x=100, y=120, z=220),
        )
    )
    cap_trace_idx = len(fig.data) - 1
    existing_menus = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    existing_menus.append(
        dict(
            type="buttons",
            direction="right",
            x=0.02,
            y=1.03,
            xanchor="left",
            yanchor="bottom",
            showactive=True,
            bgcolor="#F7FBFF",
            bordercolor="#2B6CB0",
            borderwidth=1.2,
            font=dict(size=12, color="#0F3D6E"),
            pad=dict(r=8, t=4, l=8, b=4),
            buttons=[
                dict(
                    label="Cap: Off",
                    method="restyle",
                    args=[{"visible": [True, False]}, [point_trace_idx, cap_trace_idx]],
                ),
                dict(
                    label="Cap: On",
                    method="restyle",
                    args=[{"visible": [True, True]}, [point_trace_idx, cap_trace_idx]],
                ),
            ],
        )
    )
    fig.update_layout(updatemenus=existing_menus)


def plot_topomap_3d_if_available(
    payload: Optional[TopomapPayload],
    title: str,
    color_title: str,
) -> Optional[go.Figure]:
    """Render a 3D channel-position topomap aligned to sensor geometry.

    The rendering mirrors the single-run MEGqc 3D convention:
    - preserve the physical point-cloud geometry (centered only),
    - group exact same sensor coordinates into one marker (common for Elekta
      grad pairs), averaging values per location,
    - keep channel identities in hover text.
    """
    if payload is None:
        return None
    x = np.asarray(payload.layout.x, dtype=float).reshape(-1)
    y = np.asarray(payload.layout.y, dtype=float).reshape(-1)
    z_raw = np.asarray(payload.layout.z, dtype=float).reshape(-1) if payload.layout.z is not None else None
    values = np.asarray(payload.values, dtype=float).reshape(-1)
    if z_raw is not None:
        n = min(x.size, y.size, z_raw.size, values.size, len(payload.layout.names))
    else:
        n = min(x.size, y.size, values.size, len(payload.layout.names))
    if n < 3:
        return None

    x = x[:n]
    y = y[:n]
    values = values[:n]
    names = np.asarray(payload.layout.names[:n], dtype=object)
    if z_raw is not None:
        z_raw = z_raw[:n]
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z_raw) & np.isfinite(values)
    else:
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if np.sum(mask) < 3:
        return None

    x = x[mask]
    y = y[mask]
    values = values[mask]
    names = names[mask]

    if z_raw is not None:
        z = z_raw[mask]
    else:
        # Fallback pseudo-depth if true z is unavailable.
        xr = x - float(np.nanmean(x))
        yr = y - float(np.nanmean(y))
        scale = float(np.nanmax(np.sqrt(xr ** 2 + yr ** 2)))
        if (not np.isfinite(scale)) or scale <= np.finfo(float).eps:
            scale = 1.0
        xn = xr / scale
        yn = yr / scale
        radial2 = np.clip(xn ** 2 + yn ** 2, 0.0, 1.0)
        z = np.sqrt(1.0 - radial2)

    # Group identical locations (Elekta grad pairs) and average per-location value.
    coords = np.column_stack([x, y, z])
    rounded = np.round(coords, decimals=6)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    xg: List[float] = []
    yg: List[float] = []
    zg: List[float] = []
    vg: List[float] = []
    hover: List[str] = []
    for key in np.unique(inv):
        idx = np.where(inv == key)[0]
        if idx.size == 0:
            continue
        xg.append(float(np.nanmean(x[idx])))
        yg.append(float(np.nanmean(y[idx])))
        zg.append(float(np.nanmean(z[idx])))
        vg.append(float(np.nanmean(values[idx])))
        hover_lines = [f"{names[i]}: {values[i]:.3g}" for i in idx]
        hover.append("<br>".join(hover_lines))

    if len(xg) < 3:
        return None

    xg_arr = np.asarray(xg, dtype=float)
    yg_arr = np.asarray(yg, dtype=float)
    zg_arr = np.asarray(zg, dtype=float)
    vg_arr = np.asarray(vg, dtype=float)

    # Center the cloud while keeping physical geometry and orientation.
    xg_arr = xg_arr - float(np.nanmean(xg_arr))
    yg_arr = yg_arr - float(np.nanmean(yg_arr))
    zg_arr = zg_arr - float(np.nanmean(zg_arr))

    fig = go.Figure(
        go.Scatter3d(
            x=xg_arr,
            y=yg_arr,
            z=zg_arr,
            mode="markers",
            text=hover,
            hovertemplate="%{text}<br>Grouped value: %{marker.color:.3g}<extra></extra>",
            marker={
                "size": 9.5,
                "color": vg_arr,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": color_title, "x": 0.95, "len": 0.82},
                "line": {"width": 0.3, "color": "#2F3E46"},
                "opacity": 0.90,
            },
            showlegend=False,
        )
    )
    fig.update_layout(
        height=860,
        title={"text": title, "x": 0.5, "y": 0.98, "xanchor": "center", "yanchor": "top"},
        template="plotly_white",
        margin={"l": 24, "r": 24, "t": 38, "b": 16},
        scene={
            "domain": {"x": [0.02, 0.90], "y": [0.0, 1.0]},
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.35, "y": 1.25, "z": 0.78}},
        },
    )
    _add_solid_cap_toggle_to_topomap_3d(fig, xg_arr, yg_arr, zg_arr)
    return fig


def _reset_lazy_figure_store() -> None:
    """Reset per-report lazy plot storage before composing HTML."""
    global _LAZY_PLOT_COUNTER, _LAZY_PAYLOAD_COUNTER
    _LAZY_PLOT_PAYLOADS.clear()
    _LAZY_PLOT_COUNTER = count(1)
    _LAZY_PAYLOAD_COUNTER = count(1)


def _lazy_payload_script_tags_html() -> str:
    """Return inline JSON script tags for lazily rendered Plotly payloads."""
    return "".join(
        f"<script id='{payload_id}' type='application/json'>{payload_json}</script>"
        for payload_id, payload_json in _LAZY_PLOT_PAYLOADS.items()
    )


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
    fig_id = f"lazy-plot-{next(_LAZY_PLOT_COUNTER)}"
    payload_id = f"lazy-payload-{next(_LAZY_PAYLOAD_COUNTER)}"
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


def _figure_to_div(
    fig: Optional[go.Figure],
    *,
    include_axis_size_control: bool = True,
    include_plot_controls: bool = True,
) -> str:
    if fig is None:
        return "<p>No distribution is available for this section.</p>"
    fig_out = go.Figure(fig)

    # ── Zone 1: extract title into an independent HTML block ─────────
    import html as _html_mod
    title_html = ""
    fig_title = fig_out.layout.title
    if fig_title:
        title_text = ""
        if isinstance(fig_title, str):
            title_text = fig_title
        else:
            title_text = str(getattr(fig_title, "text", "") or "")
        if title_text:
            title_html = f"<div class='fig-title'>{_html_mod.escape(title_text)}</div>"
            fig_out.update_layout(title=None)

    # ── Zone 2 (legend) stays inside the Plotly figure at y>1 ────────
    # ── Zone 3 (plot) occupies the remaining Plotly area ─────────────

    # Universal controls: axis-label sizing for all axis-bearing plots.
    if include_axis_size_control:
        _attach_axis_label_size_controller(fig_out)
    controls = _extract_figure_controls(fig_out) if include_plot_controls else []

    height = fig_out.layout.height
    if height is None or not np.isfinite(height):
        height = 640
    height_px = f"{int(max(420, float(height)))}px"
    return title_html + _register_lazy_figure(fig_out, height_px=height_px, controls=controls)


def _trace_numeric(values) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.any(np.isfinite(arr)):
        return None
    return arr


def _normalize_figure_for_mode(fig: Optional[go.Figure], mode: str) -> Optional[go.Figure]:
    """Create a normalized copy of a Plotly figure for UI toggles.

    The goal is visual comparability, not replacement of raw values. We only
    transform coordinates relevant to the requested mode and keep structure,
    labels, and trace grouping intact so users can switch views consistently.
    """
    if fig is None:
        return None
    out = go.Figure(fig)
    for tr in out.data:
        t = getattr(tr, "type", "")
        if mode in ("y", "auto"):
            if t in ("scatter", "scattergl", "violin", "box"):
                arr = _trace_numeric(getattr(tr, "y", None))
                if arr is not None:
                    tr.y = _robust_normalize_array(arr)
        if mode in ("x", "auto"):
            if t == "histogram":
                arr = _trace_numeric(getattr(tr, "x", None))
                if arr is not None:
                    tr.x = _robust_normalize_array(arr)
            elif mode == "x" and t in ("scatter", "scattergl"):
                arr = _trace_numeric(getattr(tr, "x", None))
                if arr is not None:
                    tr.x = _robust_normalize_array(arr)
        if mode in ("z", "auto") and t == "heatmap":
            z = np.asarray(getattr(tr, "z", np.array([])), dtype=float)
            if z.size:
                tr.z = _robust_normalize_array(z.reshape(-1)).reshape(z.shape)
        if mode == "color" and t in ("scatter", "scattergl"):
            marker = getattr(tr, "marker", None)
            if marker is not None and getattr(marker, "color", None) is not None:
                arr = _trace_numeric(marker.color)
                if arr is not None:
                    marker.color = _robust_normalize_array(arr)
            arr = _trace_numeric(getattr(tr, "customdata", None))
            if arr is not None:
                tr.customdata = _robust_normalize_array(arr)

    if mode in ("y", "auto"):
        for axis_name in ("yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5", "yaxis6"):
            axis = getattr(out.layout, axis_name, None)
            if axis is not None and getattr(axis, "title", None) is not None and getattr(axis.title, "text", None):
                axis.title.text = f"{axis.title.text} [normalized]"
    if mode == "x":
        for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5", "xaxis6"):
            axis = getattr(out.layout, axis_name, None)
            if axis is not None and getattr(axis, "title", None) is not None and getattr(axis.title, "text", None):
                axis.title.text = f"{axis.title.text} [normalized]"
    return out


def _auto_figure_details(fig: Optional[go.Figure]) -> str:
    if fig is None:
        return ""

    json_fig = fig.to_plotly_json()
    layout = json_fig.get("layout", {})
    title_text = ""
    title_obj = layout.get("title", {})
    if isinstance(title_obj, dict):
        title_text = str(title_obj.get("text", "") or "")

    x_titles = []
    y_titles = []
    for key, val in layout.items():
        if key.startswith("xaxis") and isinstance(val, dict):
            t = val.get("title", {})
            if isinstance(t, dict) and t.get("text"):
                x_titles.append(str(t.get("text")))
        if key.startswith("yaxis") and isinstance(val, dict):
            t = val.get("title", {})
            if isinstance(t, dict) and t.get("text"):
                y_titles.append(str(t.get("text")))
    x_titles = sorted(set(x_titles))
    y_titles = sorted(set(y_titles))

    trace_types = [getattr(tr, "type", "") for tr in fig.data]
    has_heatmap = any(t == "heatmap" for t in trace_types)
    has_violin = any(t == "violin" for t in trace_types)
    has_hist = any(t == "histogram" for t in trace_types)
    has_table = any(t == "table" for t in trace_types)
    scatter_modes = [str(getattr(tr, "mode", "")) for tr in fig.data if getattr(tr, "type", "") in ("scatter", "scattergl")]
    has_lines = any("lines" in m for m in scatter_modes)
    has_markers = any("markers" in m for m in scatter_modes)

    color_title = None
    for tr in fig.data:
        marker = getattr(tr, "marker", None)
        if marker is not None and getattr(marker, "colorbar", None) is not None:
            cb = marker.colorbar
            if getattr(cb, "title", None) is not None and getattr(cb.title, "text", None):
                color_title = str(cb.title.text)
                break
        if getattr(tr, "colorbar", None) is not None:
            cb = tr.colorbar
            if getattr(cb, "title", None) is not None and getattr(cb.title, "text", None):
                color_title = str(cb.title.text)
                break

    all_axis_text = " | ".join(x_titles + y_titles + ([color_title] if color_title else []))
    context_blob = f"{title_text} | {all_axis_text}".lower()
    looks_global = any(tok in context_blob for tok in ("global", "across subjects", "all recordings", "pooled", "all channels"))
    looks_subject = any(tok in context_blob for tok in ("subject", "recording", "fingerprint", "drill-down", "within-subject"))

    details = []
    if x_titles:
        details.append(f"X-axis encodes {', '.join(x_titles)}.")
    if y_titles:
        details.append(f"Y-axis encodes {', '.join(y_titles)}.")
    if color_title:
        details.append(f"The color scale encodes {color_title}.")

    if has_violin:
        details.append(
            "In violin views, each violin is a full empirical distribution; violin width reflects density, and jittered points represent recording-level values with subject identity in hover text."
        )
    if has_hist:
        details.append(
            "In histogram views, bars show probability density and the smooth overlaid curve is a kernel density estimate of the same observations."
        )
    if has_heatmap:
        details.append(
            "In heatmaps, each cell is one channel-by-epoch value. Vertical stripe patterns indicate simultaneous across-channel shifts by epoch, while isolated horizontal structure indicates channel-specific burden."
        )
    if has_lines:
        details.append(
            "In line/band profiles, the central line is the median trend and shaded envelopes indicate spread (IQR and 5-95% where present), so widening bands reflect increased cross-channel variability."
        )
    if has_markers and not has_violin and not has_hist and not has_table:
        details.append(
            "In scatter views, each point is one recording-level summary. Outlying points indicate recordings with atypical metric combinations relative to the cohort cloud."
        )
    if has_table:
        details.append(
            "In tables, each row corresponds to one subject/recording entity and each column is a robust metric summary in native units."
        )

    if "std" in context_blob:
        details.append(
            "STD summarizes channel variability amplitude per epoch; stable cohorts usually show gradual epoch trends, while heavy upper tails indicate a subset of channels with larger variability."
        )
    if "ptp" in context_blob:
        details.append(
            "PtP summarizes excursion amplitude. A heavy upper tail indicates channels with larger peak-to-peak bursts than the cohort median."
        )
    if "psd" in context_blob or "frequency" in context_blob or "mains" in context_blob:
        details.append(
            "PSD-based plots use frequency in Hz on the x-axis. Relative-power peaks around mains frequency and harmonics indicate narrow-band dominance in spectral content."
        )
    if "ecg" in context_blob or "eog" in context_blob or "|r|" in context_blob or "corr" in context_blob:
        details.append(
            "Correlation magnitude plots are unitless (|r|). Higher values indicate stronger temporal similarity between channel signals and physiological reference traces."
        )
    if "muscle" in context_blob or "z-score" in context_blob:
        details.append(
            "Muscle score is unitless (z-score). Elevated upper-tail epochs indicate brief high-frequency burden, while persistently elevated profiles indicate sustained high-frequency activity."
        )

    if looks_global:
        details.append(
            "This is a global pooled view: it summarizes the cohort footprint and should not be interpreted as subject identification."
        )
    if looks_subject:
        details.append(
            "This is a subject-aware view: it supports locating outlying subjects/recordings, but it does not by itself determine downstream handling."
        )

    if not details:
        details.append(
            "Interpret values using axis titles and hover text: these define units, entities, and the aggregation level of each visual element."
        )
    return " ".join(details)


def _figure_block(
    fig: Optional[go.Figure],
    interpretation: str,
    *,
    normalized_variant: bool = False,
    norm_mode: str = "y",
) -> str:
    """Build one HTML figure block with interpretation text.

    When ``normalized_variant`` is enabled, we emit paired raw/normalized views
    controlled by a lightweight client-side toggle. Each block gets a unique id
    so toggles remain independent even in deeply nested tab layouts.
    """
    auto_details = _auto_figure_details(fig)
    detailed_note = f"{interpretation} {auto_details}".strip()
    if (not normalized_variant) or fig is None:
        return (
            f'<div class="fig">{_figure_to_div(fig)}</div>'
            f'<p class="fig-note"><strong>How to interpret:</strong> {detailed_note}</p>'
        )

    norm_fig = _normalize_figure_for_mode(fig, norm_mode)
    toggle_id = f"fig-toggle-{next(_FIG_TOGGLE_COUNTER)}"
    raw_div = _figure_to_div(fig)
    norm_div = _figure_to_div(norm_fig)
    return (
        f"<div class='fig-switch' data-fig-toggle='{toggle_id}'>"
        f"<button class='fig-switch-btn active' data-target='{toggle_id}-raw'>Raw</button>"
        f"<button class='fig-switch-btn' data-target='{toggle_id}-norm'>Normalized</button>"
        f"</div>"
        f"<div id='{toggle_id}-raw' class='fig-view active'>{raw_div}</div>"
        f"<div id='{toggle_id}-norm' class='fig-view'>{norm_div}</div>"
        f"<p class='fig-note'><strong>How to interpret:</strong> {detailed_note} "
        "The normalized view applies a robust scaling (median/IQR) to improve comparability of shape across tasks without changing rank structure.</p>"
    )


def _condition_figure_blocks(
    figures_by_condition: Dict[str, Optional[go.Figure]],
    interpretation: str,
    *,
    interpretation_by_condition: Optional[Dict[str, str]] = None,
    normalized_variant: bool = False,
    norm_mode: str = "y",
) -> str:
    chunks = []
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
    return "".join(chunks) if chunks else "<p>No distribution is available for this section.</p>"


def _build_subtabs_html(group_id: str, tabs: Sequence[Tuple[str, str]], *, level: int = 1) -> str:
    """Render a tab group and attach a hierarchy level for styling.

    The level value is used only for visual hierarchy (panel/background tone),
    which helps users orient inside multi-layer tab structures.
    """
    if not tabs:
        return "<p>No content available.</p>"
    buttons = []
    contents = []
    for idx, (label, html) in enumerate(tabs):
        panel_id = f"{group_id}-panel-{idx}"
        active = " active" if idx == 0 else ""
        buttons.append(
            f"<button class='subtab-btn{active}' data-tab-group='{group_id}' data-target='{panel_id}'>{label}</button>"
        )
        contents.append(
            f"<div id='{panel_id}' class='subtab-content{active}' data-tab-group='{group_id}'>{html}</div>"
        )
    level = max(1, int(level))
    return f"<div class='subtab-group level-{level}'><div class='subtab-row'>{''.join(buttons)}</div>{''.join(contents)}</div>"


def _topomap_blocks(
    payloads_by_condition: Dict[str, TopomapPayload],
    title_prefix: str,
    color_title: str,
    interpretation: str,
    *,
    normalized_variant: bool = False,
) -> str:
    chunks: List[str] = []
    for idx, (cond, payload) in enumerate(sorted(payloads_by_condition.items())):
        fig_2d = plot_topomap_if_available(
            payload,
            title=f"{title_prefix} ({cond})",
            color_title=color_title,
        )
        fig_3d = plot_topomap_3d_if_available(
            payload,
            title=f"{title_prefix} (3D) ({cond})",
            color_title=color_title,
        )
        tabs: List[Tuple[str, str]] = []
        if fig_2d is not None:
            tabs.append(
                (
                    "2D",
                    _figure_block(
                        fig_2d,
                        interpretation,
                        normalized_variant=normalized_variant,
                        norm_mode="color",
                    ),
                )
            )
        if fig_3d is not None:
            tabs.append(
                (
                    "3D",
                    _figure_block(
                        fig_3d,
                        interpretation,
                        normalized_variant=normalized_variant,
                        norm_mode="color",
                    ),
                )
            )
        if not tabs:
            continue
        chunks.append(f"<h4>{cond}</h4>")
        chunks.append(_build_subtabs_html(f"topomap-view-{idx}-{re.sub(r'[^a-z0-9]+','-',cond.lower())}", tabs, level=4))

    if not chunks:
        return "<p>Topographic maps not shown: channel positions not available in stored outputs.</p>"
    return "".join(chunks)


def _load_settings_snapshot(megqc_root: str) -> str:
    """Load the latest MEGqc settings snapshot from one analysis root."""
    config_dir = Path(megqc_root) / "config"
    ini_files = sorted(config_dir.glob("*.ini"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ini_files:
        return "No settings snapshot file was found."

    chosen = ini_files[0]
    cfg = configparser.ConfigParser()
    cfg.read(chosen)

    sections = ["default", "Epoching", "STD", "PTP_manual", "PSD", "ECG", "EOG", "Muscle"]
    lines = [f"source={chosen}"]
    disallowed_tokens = ("bad", "good", "reject", "flag", "exceed", "pass", "fail", "threshold")
    for section in sections:
        if section in cfg:
            qa_safe_items = []
            for key, value in cfg[section].items():
                key_l = key.lower()
                val_l = str(value).lower()
                if any(tok in key_l for tok in disallowed_tokens):
                    continue
                if any(tok in val_l for tok in disallowed_tokens):
                    continue
                qa_safe_items.append(f"{key}={value}")
            if qa_safe_items:
                lines.append(f"[{section}] {', '.join(qa_safe_items)}")
            else:
                lines.append(f"[{section}] QA-safe settings view active")

    return "\n".join(lines) if lines else f"source={chosen}"


def _summary_table_html(acc: ChTypeAccumulator) -> str:
    rows = []
    for module in MODULES:
        rows.append(
            "<tr>"
            f"<td>{module}</td>"
            f"<td>{acc.module_present.get(module, 0)}</td>"
            f"<td>{acc.module_missing.get(module, 0)}</td>"
            "</tr>"
        )
    module_table = (
        "<table>"
        "<thead><tr><th>Module</th><th>Available n</th><th>Missing n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    condition_rows = []
    for cond, count in sorted(acc.runs_by_condition.items(), key=lambda x: x[0]):
        condition_rows.append(f"<tr><td>{cond}</td><td>{count}</td></tr>")
    condition_table = (
        "<table>"
        "<thead><tr><th>Task</th><th>Recordings</th></tr></thead>"
        f"<tbody>{''.join(condition_rows)}</tbody>"
        "</table>"
    )

    return (
        "<div class='summary-grid'>"
        "<div class='tile'>"
        f"<h3>Dataset summary</h3>"
        f"<p><strong>N subjects:</strong> {len(acc.subjects)}</p>"
        f"<p><strong>N recordings:</strong> {acc.run_count}</p>"
        "</div>"
        "<div class='tile'>"
        "<h3>Recordings per task</h3>"
        f"{condition_table}"
        "</div>"
        "<div class='tile'>"
        "<h3>Missingness per module</h3>"
        f"{module_table}"
        "</div>"
        "</div>"
    )


def _paths_html(source_paths: set) -> str:
    if not source_paths:
        return "<p>No machine-readable derivatives were consumed.</p>"

    paths = sorted(source_paths)
    max_lines = 80
    visible = paths[:max_lines]
    hidden = max(0, len(paths) - max_lines)
    items = "".join(f"<li><code>{p}</code></li>" for p in visible)
    if hidden > 0:
        items += f"<li>... {hidden} additional paths omitted for brevity.</li>"
    return f"<ul>{items}</ul>"


def _plot_summary_distribution_recordings(
    df: pd.DataFrame,
    value_col: str,
    *,
    title: str,
    y_label: str,
    color: str = "#2A6FBB",
    enable_style_controls: bool = True,
    show_counts_annotation: bool = True,
) -> Optional[go.Figure]:
    """One-panel pooled distribution where each dot is one recording."""
    if df.empty or value_col not in df.columns:
        return None
    vals = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(vals)
    if not np.any(mask):
        return None
    data = df.loc[mask].copy()
    vals = vals[mask]
    if vals.size == 0:
        return None

    x_center = np.zeros(vals.size, dtype=float)
    rng = np.random.default_rng(17)
    x_jitter = x_center + rng.uniform(-0.10, 0.10, size=vals.size)
    if "hover_entities" in data.columns:
        hover = data["hover_entities"].astype(str).to_numpy()
    else:
        hover = (
            "sub=" + data.get("subject", "n/a").astype(str)
            + "<br>task=" + data.get("task", "n/a").astype(str)
            + "<br>run=" + data.get("run", "n/a").astype(str)
            + "<br>channel_type=" + data.get("channel_type", "n/a").astype(str)
        ).to_numpy()
    subject_codes = pd.Categorical(data.get("subject", pd.Series(["n/a"] * len(data))).astype(str)).codes.astype(float)
    _sc_min = float(subject_codes.min()) if subject_codes.size > 0 else 0.0
    _sc_max = float(subject_codes.max()) if subject_codes.size > 0 else 1.0
    if _sc_min == _sc_max:
        _sc_max = _sc_min + 1.0  # prevent degenerate colorscale (cmin==cmax → NaN in WebGL)
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=x_center,
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
            x=x_center,
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
            x=x_jitter,
            y=vals,
            mode="markers",
            marker={
                "size": 8.0,
                "color": subject_codes,
                "colorscale": "Turbo",
                "cmin": _sc_min,
                "cmax": _sc_max,
                "opacity": 0.76,
                "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                "showscale": False,
            },
            customdata=np.stack([hover], axis=-1),
            hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
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


def _pooled_topomap_payload(
    payloads_by_condition: Dict[str, TopomapPayload],
    counts_by_condition: Optional[Dict[str, np.ndarray]] = None,
) -> Optional[TopomapPayload]:
    """Pool condition-level topomap values into one weighted channel map."""
    if not payloads_by_condition:
        return None

    weighted_sum: Dict[str, float] = {}
    weight_sum: Dict[str, float] = {}
    coords: Dict[str, Tuple[float, float, float]] = {}
    order: List[str] = []
    has_any_z = False

    for cond, payload in payloads_by_condition.items():
        if payload is None:
            continue
        names = list(payload.layout.names)
        vals = np.asarray(payload.values, dtype=float).reshape(-1)
        x = np.asarray(payload.layout.x, dtype=float).reshape(-1)
        y = np.asarray(payload.layout.y, dtype=float).reshape(-1)
        z = np.asarray(payload.layout.z, dtype=float).reshape(-1) if payload.layout.z is not None else None
        cnt = None
        if counts_by_condition is not None and cond in counts_by_condition:
            cnt = np.asarray(counts_by_condition[cond], dtype=float).reshape(-1)

        n = min(len(names), vals.size, x.size, y.size, (z.size if z is not None else len(names)))
        if cnt is not None:
            n = min(n, cnt.size)
        if n <= 0:
            continue

        for idx in range(n):
            name = str(names[idx])
            val = float(vals[idx])
            if not np.isfinite(val):
                continue
            w = float(cnt[idx]) if (cnt is not None and np.isfinite(cnt[idx]) and float(cnt[idx]) > 0) else 1.0
            weighted_sum[name] = weighted_sum.get(name, 0.0) + val * w
            weight_sum[name] = weight_sum.get(name, 0.0) + w
            if name not in coords:
                zi = float(z[idx]) if (z is not None and np.isfinite(z[idx])) else np.nan
                if np.isfinite(zi):
                    has_any_z = True
                coords[name] = (float(x[idx]), float(y[idx]), zi)
                order.append(name)

    keep = [name for name in order if name in weight_sum and weight_sum[name] > 0]
    if not keep:
        return None

    vals_out = np.asarray([weighted_sum[name] / weight_sum[name] for name in keep], dtype=float)
    x_out = np.asarray([coords[name][0] for name in keep], dtype=float)
    y_out = np.asarray([coords[name][1] for name in keep], dtype=float)
    z_out = np.asarray([coords[name][2] for name in keep], dtype=float) if has_any_z else None
    if z_out is not None and not np.any(np.isfinite(z_out)):
        z_out = None

    return TopomapPayload(
        layout=SensorLayout(x=x_out, y=y_out, names=list(keep), z=z_out),
        values=vals_out,
    )


def _run_rows_dataframe(rows: List[RunMetricRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([row.__dict__ for row in rows])
    numeric_cols = [
        "std_mean", "std_median", "std_upper_tail", "std_median_norm", "std_upper_tail_norm",
        "ptp_mean", "ptp_median", "ptp_upper_tail", "ptp_median_norm", "ptp_upper_tail_norm",
        "mains_ratio", "mains_harmonics_ratio", "ecg_mean_abs_corr", "ecg_p95_abs_corr",
        "eog_mean_abs_corr", "eog_p95_abs_corr", "muscle_mean", "muscle_median", "muscle_p95",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["subject", "session", "task", "run", "condition", "acquisition", "recording", "processing", "channel_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("n/a").astype(str)

    def _cond_display(row: pd.Series) -> str:
        parts = []
        if row["task"] != "n/a":
            parts.append(row["task"])
        return ", ".join(parts) if parts else "all recordings"

    df["condition_label"] = df.apply(_cond_display, axis=1)
    df["hover_entities"] = (
        "sub=" + df["subject"]
        + "<br>ses=" + df["session"]
        + "<br>task=" + df["task"]
        + "<br>run=" + df["run"]
        + "<br>acq=" + df["acquisition"]
        + "<br>recording=" + df["recording"]
        + "<br>proc=" + df["processing"]
        + "<br>channel_type=" + df["channel_type"]
        + "<br>run_key=" + df["run_key"].astype(str)
    )
    return df


def _subject_color_map(subjects: List[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#003f5c", "#58508d",
        "#bc5090", "#ff6361", "#ffa600", "#2f4b7c", "#665191", "#a05195",
        "#d45087", "#f95d6a", "#ff7c43", "#ffa600",
    ]
    out = {}
    for idx, subject in enumerate(sorted(subjects)):
        out[subject] = palette[idx % len(palette)]
    return out


def _condition_symbol_map(conditions: List[str]) -> Dict[str, str]:
    symbols = ["circle", "diamond", "square", "cross", "triangle-up", "triangle-down", "x", "star"]
    out = {}
    for idx, cond in enumerate(sorted(conditions)):
        out[cond] = symbols[idx % len(symbols)]
    return out


def plot_subject_condition_violin(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
) -> Optional[go.Figure]:
    if df.empty or value_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[value_col])].copy()
    if data.empty:
        return None

    fig = go.Figure()
    for label in sorted(data["condition_label"].unique()):
        vals = data.loc[data["condition_label"] == label, value_col].to_numpy(dtype=float)
        vals = _finite_array(vals)
        if vals.size == 0:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        fig.add_trace(
            go.Violin(
                x=np.repeat(label, vals.size),
                y=vals,
                name=f"{label} (n={vals.size})",
                box_visible=True,
                meanline_visible=False,
                points=False,
                line={"width": 1.0},
                opacity=0.45,
            )
        )

    subject_codes = pd.Categorical(data["subject"]).codes.astype(float)
    keep_points = _downsample_indices(len(data), min(MAX_POINTS_SCATTER, len(data)))
    scatter_data = data.iloc[keep_points]
    scatter_codes = subject_codes[keep_points]
    fig.add_trace(
        go.Scattergl(
            x=scatter_data["condition_label"],
            y=scatter_data[value_col],
            mode="markers",
            name="Recording summaries",
            marker={
                "size": 6,
                "color": scatter_codes,
                "colorscale": "Turbo",
                "opacity": 0.72,
                "line": {"width": 0.35, "color": "rgba(24,24,24,0.45)"},
                "showscale": False,
            },
            customdata=np.stack([scatter_data["hover_entities"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
            showlegend=False,
        )
    )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task",
        yaxis_title=y_label,
        template="plotly_white",
        violinmode="group",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def plot_run_fingerprint_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> Optional[go.Figure]:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    if data.empty:
        return None

    symbol_map = _condition_symbol_map(sorted(data["condition_label"].unique()))
    subject_map = _subject_color_map(sorted(data["subject"].astype(str).unique().tolist()))
    keep_points = _downsample_indices(len(data), min(MAX_POINTS_SCATTER, len(data)))
    data = data.iloc[keep_points].copy()

    fig = go.Figure()
    for cond in sorted(symbol_map):
        dcond = data.loc[data["condition_label"] == cond].copy()
        if dcond.empty:
            continue
        # One trace per task label enables native legend filtering (click to hide
        # all points for that task).
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
                    "color": [subject_map.get(str(s), "#4F6F84") for s in dcond["subject"].astype(str)],
                    "line": {"width": 0.6, "color": "#2b2d42"},
                    "opacity": 0.86,
                    "showscale": False,
                },
                customdata=np.stack([dcond["hover_entities"]], axis=-1),
                hovertemplate="%{customdata[0]}<br>task="
                + cond
                + "<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
                showlegend=True,
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 55, "r": 30, "t": 50, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}},
    )
    return fig


def plot_subject_ranking_table(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if df.empty:
        return None

    def _p95(series: pd.Series) -> float:
        vals = _finite_array(series.to_numpy(dtype=float))
        return float(np.nanquantile(vals, 0.95)) if vals.size else np.nan

    grouped = df.groupby("subject", dropna=False).agg(
        n_recordings=("run_key", "nunique"),
        std_p95=("std_upper_tail", _p95),
        ptp_p95=("ptp_upper_tail", _p95),
        mains_mean=("mains_ratio", "mean"),
        muscle_p95=("muscle_p95", _p95),
        eog_mean=("eog_mean_abs_corr", "mean"),
        ecg_mean=("ecg_mean_abs_corr", "mean"),
    ).reset_index()

    if grouped.empty:
        return None

    grouped = grouped.sort_values(
        by=["std_p95", "ptp_p95", "mains_mean", "muscle_p95", "eog_mean", "ecg_mean"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))

    display_cols = ["rank", "subject", "n_recordings", "std_p95", "ptp_p95", "mains_mean", "muscle_p95", "eog_mean", "ecg_mean"]
    header = ["Rank", "Subject", "N recordings", "STD p95", "PtP p95", "Mains mean", "Muscle p95", "EOG |r| mean", "ECG |r| mean"]
    cells = []
    for col in display_cols:
        if col in ("rank", "subject", "n_recordings"):
            cells.append(grouped[col].tolist())
        else:
            cells.append([f"{v:.3g}" if np.isfinite(v) else "n/a" for v in grouped[col].to_numpy(dtype=float)])

    fig = go.Figure(
        data=[
            go.Table(
                header={"values": header, "fill_color": "#eef5fd", "align": "left"},
                cells={"values": cells, "fill_color": "#ffffff", "align": "left"},
            )
        ]
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
    )
    return fig


def plot_subject_metric_heatmap(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if df.empty:
        return None
    metrics = {
        "STD p95": "std_upper_tail",
        "PtP p95": "ptp_upper_tail",
        "Mains mean": "mains_ratio",
        "Muscle p95": "muscle_p95",
        "EOG |r| mean": "eog_mean_abs_corr",
        "ECG |r| mean": "ecg_mean_abs_corr",
    }
    agg = df.groupby("subject", dropna=False).agg({col: "median" for col in metrics.values()})
    if agg.empty:
        return None

    raw = np.full((agg.shape[0], len(metrics)), np.nan, dtype=float)
    z = np.full((agg.shape[0], len(metrics)), np.nan, dtype=float)
    for idx, col in enumerate(metrics.values()):
        raw[:, idx] = agg[col].to_numpy(dtype=float)
        # Normalize per metric column for comparability across heterogeneous units.
        z[:, idx] = _robust_normalize_array(raw[:, idx])

    bounds = _robust_bounds(z)
    zmin = bounds[0] if bounds is not None else None
    zmax = bounds[1] if bounds is not None else None

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(metrics.keys()),
            y=agg.index.tolist(),
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "Normalized value (robust z)"},
            # Keep raw values in hover so users retain physical interpretability.
            customdata=np.stack([raw], axis=-1),
            hovertemplate="subject=%{y}<br>metric=%{x}<br>raw=%{customdata[0]:.3g}<br>normalized=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Metric",
        yaxis_title="Subject",
        template="plotly_white",
        margin={"l": 70, "r": 20, "t": 65, "b": 50},
    )
    fig.update_xaxes(tickangle=-22)
    return fig


def plot_subject_condition_effect(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
) -> Optional[go.Figure]:
    if df.empty or metric_col not in df.columns:
        return None
    data = df.loc[np.isfinite(df[metric_col])].copy()
    if data.empty:
        return None
    pivot = data.pivot_table(
        index="subject",
        columns="condition_label",
        values=metric_col,
        aggfunc="median",
    )
    if pivot.shape[1] < 2:
        return None
    conditions = list(pivot.columns)
    color_map = _subject_color_map(list(pivot.index))

    fig = go.Figure()
    for subject in pivot.index:
        vals = pivot.loc[subject].to_numpy(dtype=float)
        if np.sum(np.isfinite(vals)) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=conditions,
                y=vals,
                mode="lines+markers",
                name=f"sub-{subject}",
                line={"width": 1.2, "color": color_map[subject]},
                marker={"size": 6},
                showlegend=False,
                hovertemplate=f"sub={subject}<br>task=%{{x}}<br>value=%{{y:.3g}}<extra></extra>",
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task",
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 55, "r": 20, "t": 65, "b": 50},
    )
    return fig


def _values_with_task_agnostic(values_by_condition: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {"all tasks": []}
    for cond, vals in sorted(values_by_condition.items()):
        finite = _finite_array(vals).tolist()
        out[cond] = finite
        out["all tasks"].extend(finite)
    if not out["all tasks"]:
        del out["all tasks"]
    return out


def _run_values_by_condition(df: pd.DataFrame, value_col: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    if df.empty or value_col not in df.columns:
        return {}
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    labels = df["condition_label"].astype(str).tolist() if "condition_label" in df.columns else ["all recordings"] * len(df)
    for label, val in zip(labels, values):
        if np.isfinite(val):
            out[label].append(float(val))
    return dict(out)


def _epoch_values_from_profiles(
    profiles_by_condition: Dict[str, List[Dict[str, np.ndarray]]],
    field: str = "q50",
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = defaultdict(list)
    for cond, profiles in profiles_by_condition.items():
        for prof in profiles:
            if field not in prof:
                continue
            out[cond].extend(_finite_array(prof[field]).tolist())
    return dict(out)


def _mean_matrices_by_condition(
    sum_by_condition: Dict[str, np.ndarray],
    count_by_condition: Dict[str, np.ndarray],
    include_all_tasks: bool = True,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for cond, sum_mat in sum_by_condition.items():
        cnt = count_by_condition.get(cond)
        if cnt is None:
            continue
        mean_mat = _mean_matrix_from_sum_count(sum_mat, cnt)
        if mean_mat.size:
            out[cond] = mean_mat

    if include_all_tasks and out:
        sum_total = None
        cnt_total = None
        for cond in out.keys():
            s = np.asarray(sum_by_condition[cond], dtype=float)
            c = np.asarray(count_by_condition[cond], dtype=float)
            if sum_total is None:
                sum_total = s
                cnt_total = c
                continue

            n_rows = max(sum_total.shape[0], s.shape[0])
            n_cols = max(sum_total.shape[1], s.shape[1])

            def _pad(a: np.ndarray) -> np.ndarray:
                if a.shape == (n_rows, n_cols):
                    return a
                return np.pad(a, ((0, n_rows - a.shape[0]), (0, n_cols - a.shape[1])), mode="constant", constant_values=0.0)

            sum_total = _pad(sum_total) + _pad(s)
            cnt_total = _pad(cnt_total) + _pad(c)

        if sum_total is not None and cnt_total is not None:
            all_mean = _mean_matrix_from_sum_count(sum_total, cnt_total)
            if all_mean.size:
                out = {"all tasks": all_mean, **out}
    return out


def _heatmap_variants_by_condition_from_acc(
    run_matrices_by_condition: Dict[str, List[np.ndarray]],
    representative_by_condition: Dict[str, np.ndarray],
    sum_by_condition: Dict[str, np.ndarray],
    count_by_condition: Dict[str, np.ndarray],
    upper_by_condition: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-condition matrix variants without creating an all-tasks matrix."""
    variants = _aggregate_heatmap_variants_from_runs(run_matrices_by_condition)
    if variants:
        return variants

    mean_mats = _mean_matrices_by_condition(sum_by_condition, count_by_condition, include_all_tasks=False)
    conditions = sorted(
        set(representative_by_condition.keys())
        | set(mean_mats.keys())
        | set(upper_by_condition.keys())
    )
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for cond in conditions:
        by_variant: Dict[str, np.ndarray] = {}
        if cond in representative_by_condition:
            by_variant["Median"] = np.asarray(representative_by_condition[cond], dtype=float)
        if cond in mean_mats:
            by_variant["Mean"] = np.asarray(mean_mats[cond], dtype=float)
        if cond in upper_by_condition:
            by_variant["Upper tail"] = np.asarray(upper_by_condition[cond], dtype=float)
        if by_variant:
            out[cond] = by_variant
    return out


def _kde_curve(values: np.ndarray, n_points: int = 220) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    vals = _finite_array(values)
    if vals.size < 2:
        return None
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    std = float(np.nanstd(vals))
    iqr = float(np.nanquantile(vals, 0.75) - np.nanquantile(vals, 0.25))
    scale = std if std > 0 else iqr / 1.34
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return None
    bw = 1.06 * scale * (vals.size ** (-1.0 / 5.0))
    bw = max(bw, np.finfo(float).eps)
    x = np.linspace(vmin, vmax, num=n_points)
    u = (x[:, None] - vals[None, :]) / bw
    y = np.exp(-0.5 * (u ** 2)).sum(axis=1) / (vals.size * bw * np.sqrt(2.0 * np.pi))
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
                dict(
                    label="Centered dots",
                    method="restyle",
                    args=[{"visible": centered_visible}, target_indices],
                ),
                dict(
                    label="Side dots",
                    method="restyle",
                    args=[{"visible": side_visible}, target_indices],
                ),
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

    # Skip only if every menu has an explicit negative y (already panelized).
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
    current_bottom = int(getattr(margin, "b", 50) or 50)
    needed_bottom = 140 + int(len(new_menus) * 92)
    fig.update_layout(
        updatemenus=new_menus,
        annotations=anns,
        shapes=shapes,
        margin={
            "l": margin.l if hasattr(margin, "l") and margin.l is not None else 55,
            "r": margin.r if hasattr(margin, "r") and margin.r is not None else 20,
            "t": margin.t if hasattr(margin, "t") and margin.t is not None else 65,
            "b": max(current_bottom, needed_bottom),
        },
    )


# ---------------------------------------------------------------------------
# Legend layout helper — guarantees title is never covered
# ---------------------------------------------------------------------------

# Entries ≤ this: legend to the right of the plot (vertical, outside).
# Entries >  this: legend below the plot with extra bottom margin.
# In both cases the top margin is fixed so the title has its own space.


def _safe_legend_layout(n_entries: int, *, extra_groupclick: Optional[str] = None) -> dict:
    """Return Plotly ``legend`` + ``margin`` + ``height`` kwargs for
    non-overlapping title / legend / plot zones.

    The legend is **always horizontal**, placed above the plot area with
    ``y > 1`` so it lives in the top-margin band.  The top margin grows
    dynamically to accommodate multi-row wrapping.  The title is rendered
    as independent HTML above the figure (see ``_figure_to_div``).
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
    if extra_groupclick:
        d["groupclick"] = extra_groupclick

    return {
        "legend": d,
        "margin": {"l": 55, "r": 30, "t": margin_t, "b": 60},
        "height": max(620, margin_t + 60 + 400),
    }


def plot_histogram_distribution(
    values_by_group: Dict[str, List[float]],
    title: str,
    x_title: str,
) -> Optional[go.Figure]:
    fig = go.Figure()
    density_curves = []
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    for idx, label in enumerate(sorted(values_by_group)):
        vals = _finite_array(values_by_group[label])
        if vals.size == 0:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        color = palette[idx % len(palette)]
        density_curves.append((label, vals, color))
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=f"{label} (n={vals.size})",
                histnorm="probability density",
                opacity=1.0,
                marker={
                    "color": _hex_to_rgba(color, 0.34),
                    "line": {"width": 1.2, "color": color},
                },
                nbinsx=55,
            )
        )
    for label, vals, color in density_curves:
        kde = _kde_curve(vals)
        if kde is None:
            continue
        x, y = kde
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{label} density",
                line={"width": 2.6, "color": color},
            )
        )
    if not fig.data:
        return None
    n_legend_entries = len([t for t in fig.data if getattr(t, "showlegend", False)])
    layout_kw = _safe_legend_layout(max(n_legend_entries, len(sorted(values_by_group))))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title="Density",
        barmode="overlay",
        template="plotly_white",
        **layout_kw,
    )
    _attach_distribution_style_controls(fig, default_line_width=2.6, default_marker_size=8.0)
    return fig


def plot_density_distribution(
    values_by_group: Dict[str, List[float]],
    title: str,
    x_title: str,
) -> Optional[go.Figure]:
    fig = go.Figure()
    for label in sorted(values_by_group):
        vals = _finite_array(values_by_group[label])
        if vals.size < 2:
            continue
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        kde = _kde_curve(vals)
        if kde is None:
            continue
        x, y = kde
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{label} (n={vals.size})", line={"width": 2.6}))
    if not fig.data:
        return None
    layout_kw = _safe_legend_layout(len(sorted(values_by_group)))
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title=x_title,
        yaxis_title="Density",
        template="plotly_white",
        **layout_kw,
    )
    _attach_distribution_style_controls(fig, default_line_width=2.6, default_marker_size=8.0)
    return fig


def plot_violin_with_subject_jitter(
    values_by_group: Dict[str, List[float]],
    points_df: pd.DataFrame,
    point_col: str,
    title: str,
    y_title: str,
    *,
    show_density_ridge: bool = True,
    spanmode: str = "soft",
) -> Optional[go.Figure]:
    labels = [k for k in values_by_group if _finite_array(values_by_group[k]).size > 0]
    if not labels:
        return None

    xpos = {label: float(i) for i, label in enumerate(labels)}
    fig = go.Figure()
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    for idx, label in enumerate(labels):
        vals = _finite_array(values_by_group[label])
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        color = palette[idx % len(palette)]
        group_id = f"group::{label}"
        fig.add_trace(
            go.Violin(
                x=np.full(vals.size, xpos[label], dtype=float),
                y=vals,
                name=f"{label} (n={vals.size})",
                box_visible=True,
                meanline_visible=False,
                points=False,
                spanmode=spanmode,
                line={"width": 2.0, "color": color},
                opacity=0.45,
                legendgroup=group_id,
                showlegend=True,
            )
        )
        if show_density_ridge:
            kde = _kde_curve(vals)
            if kde is not None:
                kde_x, kde_y = kde
                y_max = float(np.nanmax(kde_y))
                if np.isfinite(y_max) and y_max > 0:
                    ridge_x = xpos[label] + 0.18 + 0.22 * (kde_y / y_max)
                    fig.add_trace(
                        go.Scatter(
                            x=ridge_x,
                            y=kde_x,
                            mode="lines",
                            line={"width": 2.4, "color": color},
                            opacity=0.9,
                            legendgroup=group_id,
                            showlegend=False,
                            hovertemplate=f"{label}<br>{y_title}=%{{y:.3g}}<br>density=%{{customdata:.3g}}<extra></extra>",
                            customdata=kde_y,
                        )
                    )

    if points_df.empty or point_col not in points_df.columns or "subject" not in points_df.columns:
        points_data = pd.DataFrame()
    else:
        # Subject points: one robust value per subject per condition label.
        tmp = points_df.loc[np.isfinite(points_df[point_col])].copy()
        tmp["condition_label"] = tmp.get("condition_label", "all recordings").astype(str)
        tmp["subject"] = tmp["subject"].fillna("n/a").astype(str)
        tmp["task"] = tmp.get("task", "n/a").fillna("n/a").astype(str)

        recs = []
        by_condition_subject = tmp.groupby(["condition_label", "subject"], dropna=False)
        for (label, subject), group in by_condition_subject:
            vals = pd.to_numeric(group[point_col], errors="coerce").to_numpy(dtype=float)
            vals = _finite_array(vals)
            if vals.size == 0:
                continue
            value = float(np.nanmedian(vals))
            tasks = sorted({t for t in group["task"].tolist() if t != "n/a"})
            hover = (
                f"sub={subject}"
                f"<br>task={label}"
                f"<br>n_recordings={len(group)}"
                + (f"<br>tasks={', '.join(tasks)}" if tasks else "")
                + f"<br>value={value:.3g}"
            )
            recs.append((str(label), value, str(subject), hover))

        by_subject = tmp.groupby("subject", dropna=False)
        for subject, group in by_subject:
            vals = pd.to_numeric(group[point_col], errors="coerce").to_numpy(dtype=float)
            vals = _finite_array(vals)
            if vals.size == 0:
                continue
            value = float(np.nanmedian(vals))
            tasks = sorted({t for t in group["task"].tolist() if t != "n/a"})
            hover = (
                f"sub={subject}"
                "<br>task=all tasks"
                f"<br>n_recordings={len(group)}"
                + (f"<br>tasks={', '.join(tasks)}" if tasks else "")
                + f"<br>value={value:.3g}"
            )
            recs.append(("all tasks", value, str(subject), hover))

        points_data = pd.DataFrame(recs, columns=["label", "value", "subject", "hover"])
        points_data = points_data.loc[points_data["label"].isin(labels)]

    centered_trace_indices: List[int] = []
    side_trace_indices: List[int] = []
    side_x_by_level: Dict[int, List[np.ndarray]] = {level: [] for level in range(1, 9)}
    displacement_map = {1: 0.10, 2: 0.14, 3: 0.18, 4: 0.22, 5: 0.26, 6: 0.30, 7: 0.34, 8: 0.38}
    default_side_level = 4
    if not points_data.empty:
        keep = _downsample_indices(len(points_data), min(MAX_POINTS_SCATTER, len(points_data)))
        points_data = points_data.iloc[keep].copy()
        points_data["subj_code"] = pd.Categorical(points_data["subject"]).codes.astype(float)
        rng = np.random.default_rng(0)
        for label in labels:
            group_points = points_data.loc[points_data["label"] == label].copy()
            if group_points.empty:
                continue
            x_center = np.full(group_points.shape[0], xpos[label], dtype=float)
            x_center = x_center + rng.uniform(-0.08, 0.08, size=x_center.size)
            side_jitter = rng.uniform(-0.06, 0.06, size=x_center.size)
            x_side_levels = {
                level: (np.full(group_points.shape[0], xpos[label] + offset, dtype=float) + side_jitter)
                for level, offset in displacement_map.items()
            }
            fig.add_trace(
                go.Scattergl(
                    x=x_center,
                    y=group_points["value"],
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": group_points["subj_code"],
                        "colorscale": "Turbo",
                        "opacity": 0.7,
                        "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                        "showscale": False,
                    },
                    customdata=np.stack([group_points["hover"]], axis=-1),
                    hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=True,
                )
            )
            centered_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scattergl(
                    x=x_side_levels[default_side_level],
                    y=group_points["value"],
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": group_points["subj_code"],
                        "colorscale": "Turbo",
                        "opacity": 0.72,
                        "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                        "showscale": False,
                    },
                    customdata=np.stack([group_points["hover"]], axis=-1),
                    hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=False,
                )
            )
            side_trace_indices.append(len(fig.data) - 1)
            for level in range(1, 9):
                side_x_by_level[level].append(x_side_levels[level])

    layout_kw = _safe_legend_layout(len(labels), extra_groupclick="togglegroup")
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task",
        yaxis_title=y_title,
        template="plotly_white",
        **layout_kw,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.1 + (0.50 if side_trace_indices else 0.0)],
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


def plot_box_with_subject_jitter(
    values_by_group: Dict[str, List[float]],
    points_df: pd.DataFrame,
    point_col: str,
    title: str,
    y_title: str,
) -> Optional[go.Figure]:
    """Boxplot view with the same subject-level jitter overlay used in violin views."""
    labels = [k for k in values_by_group if _finite_array(values_by_group[k]).size > 0]
    if not labels:
        return None

    xpos = {label: float(i) for i, label in enumerate(labels)}
    fig = go.Figure()
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
    ]
    for idx, label in enumerate(labels):
        vals = _finite_array(values_by_group[label])
        keep = _downsample_indices(vals.size, MAX_POINTS_VIOLIN)
        vals = vals[keep]
        color = palette[idx % len(palette)]
        group_id = f"group::{label}"
        fig.add_trace(
            go.Box(
                x=np.full(vals.size, xpos[label], dtype=float),
                y=vals,
                name=f"{label} (n={vals.size})",
                boxpoints=False,
                line={"width": 2.0, "color": color},
                marker={"color": color},
                fillcolor=_hex_to_rgba(color, 0.26),
                opacity=0.9,
                legendgroup=group_id,
                showlegend=True,
            )
        )

    if points_df.empty or point_col not in points_df.columns or "subject" not in points_df.columns:
        points_data = pd.DataFrame()
    else:
        tmp = points_df.loc[np.isfinite(points_df[point_col])].copy()
        tmp["condition_label"] = tmp.get("condition_label", "all recordings").astype(str)
        tmp["subject"] = tmp["subject"].fillna("n/a").astype(str)
        tmp["task"] = tmp.get("task", "n/a").fillna("n/a").astype(str)

        recs = []
        by_condition_subject = tmp.groupby(["condition_label", "subject"], dropna=False)
        for (label, subject), group in by_condition_subject:
            vals = pd.to_numeric(group[point_col], errors="coerce").to_numpy(dtype=float)
            vals = _finite_array(vals)
            if vals.size == 0:
                continue
            value = float(np.nanmedian(vals))
            tasks = sorted({t for t in group["task"].tolist() if t != "n/a"})
            hover = (
                f"sub={subject}"
                f"<br>task={label}"
                f"<br>n_recordings={len(group)}"
                + (f"<br>tasks={', '.join(tasks)}" if tasks else "")
                + f"<br>value={value:.3g}"
            )
            recs.append((str(label), value, str(subject), hover))

        by_subject = tmp.groupby("subject", dropna=False)
        for subject, group in by_subject:
            vals = pd.to_numeric(group[point_col], errors="coerce").to_numpy(dtype=float)
            vals = _finite_array(vals)
            if vals.size == 0:
                continue
            value = float(np.nanmedian(vals))
            tasks = sorted({t for t in group["task"].tolist() if t != "n/a"})
            hover = (
                f"sub={subject}"
                "<br>task=all tasks"
                f"<br>n_recordings={len(group)}"
                + (f"<br>tasks={', '.join(tasks)}" if tasks else "")
                + f"<br>value={value:.3g}"
            )
            recs.append(("all tasks", value, str(subject), hover))

        points_data = pd.DataFrame(recs, columns=["label", "value", "subject", "hover"])
        points_data = points_data.loc[points_data["label"].isin(labels)]

    centered_trace_indices: List[int] = []
    side_trace_indices: List[int] = []
    side_x_by_level: Dict[int, List[np.ndarray]] = {level: [] for level in range(1, 9)}
    displacement_map = {1: 0.10, 2: 0.14, 3: 0.18, 4: 0.22, 5: 0.26, 6: 0.30, 7: 0.34, 8: 0.38}
    default_side_level = 4
    if not points_data.empty:
        keep = _downsample_indices(len(points_data), min(MAX_POINTS_SCATTER, len(points_data)))
        points_data = points_data.iloc[keep].copy()
        points_data["subj_code"] = pd.Categorical(points_data["subject"]).codes.astype(float)
        rng = np.random.default_rng(0)
        for label in labels:
            group_points = points_data.loc[points_data["label"] == label].copy()
            if group_points.empty:
                continue
            x_center = np.full(group_points.shape[0], xpos[label], dtype=float)
            x_center = x_center + rng.uniform(-0.08, 0.08, size=x_center.size)
            side_jitter = rng.uniform(-0.06, 0.06, size=x_center.size)
            x_side_levels = {
                level: (np.full(group_points.shape[0], xpos[label] + offset, dtype=float) + side_jitter)
                for level, offset in displacement_map.items()
            }
            fig.add_trace(
                go.Scattergl(
                    x=x_center,
                    y=group_points["value"],
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": group_points["subj_code"],
                        "colorscale": "Turbo",
                        "opacity": 0.7,
                        "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                        "showscale": False,
                    },
                    customdata=np.stack([group_points["hover"]], axis=-1),
                    hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=True,
                )
            )
            centered_trace_indices.append(len(fig.data) - 1)
            fig.add_trace(
                go.Scattergl(
                    x=x_side_levels[default_side_level],
                    y=group_points["value"],
                    mode="markers",
                    marker={
                        "size": 8.0,
                        "color": group_points["subj_code"],
                        "colorscale": "Turbo",
                        "opacity": 0.72,
                        "line": {"width": 0.35, "color": "rgba(20,20,20,0.5)"},
                        "showscale": False,
                    },
                    customdata=np.stack([group_points["hover"]], axis=-1),
                    hovertemplate="%{customdata[0]}<br>value=%{y:.3g}<extra></extra>",
                    legendgroup=f"group::{label}",
                    showlegend=False,
                    visible=False,
                )
            )
            side_trace_indices.append(len(fig.data) - 1)
            for level in range(1, 9):
                side_x_by_level[level].append(x_side_levels[level])

    layout_kw = _safe_legend_layout(len(labels), extra_groupclick="togglegroup")
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task",
        yaxis_title=y_title,
        template="plotly_white",
        **layout_kw,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=[xpos[l] for l in labels],
        ticktext=labels,
        range=[-0.5, len(labels) - 0.1 + (0.50 if side_trace_indices else 0.0)],
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


def plot_recording_metric_heatmap(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    title: str,
) -> Optional[go.Figure]:
    if df.empty:
        return None

    valid_specs = [(col, label) for col, label in metric_specs if col in df.columns]
    if not valid_specs:
        return None

    data = df.copy()
    z_cols = []
    raw_cols = []
    labels = []
    for col, label in valid_specs:
        vals = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        raw_cols.append(vals)
        # Per-metric robust normalization improves visual contrast in mixed-unit matrices.
        z_cols.append(_robust_normalize_array(vals))
        labels.append(label)

    z = np.column_stack(z_cols)
    raw = np.column_stack(raw_cols)
    row_score = np.nanmean(z, axis=1)
    finite_row = np.any(np.isfinite(z), axis=1)
    if not np.any(finite_row):
        return None

    idx = np.where(finite_row)[0]
    tmp = data.iloc[idx].copy()
    tmp["__row_score__"] = row_score[idx]
    tmp["__orig_idx__"] = idx
    sort_cols = [c for c in ["subject", "session", "task", "condition", "run"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols + ["__row_score__"], ascending=[True] * len(sort_cols) + [False], kind="mergesort")
    else:
        tmp = tmp.sort_values("__row_score__", ascending=False, kind="mergesort")
    if len(tmp) > MAX_RECORDINGS_OVERVIEW:
        tmp = tmp.iloc[:MAX_RECORDINGS_OVERVIEW]

    order = tmp["__orig_idx__"].to_numpy(dtype=int)
    z = z[order, :]
    raw = raw[order, :]
    ordered_data = data.iloc[order].reset_index(drop=True)
    y = ordered_data["run_key"].fillna("n/a").astype(str).tolist()
    if len(y) != z.shape[0]:
        y = [f"recording_{idx}" for idx in range(z.shape[0])]
    show_ticklabels = len(y) <= 120

    bounds = _robust_bounds(z)
    zmin = bounds[0] if bounds is not None else None
    zmax = bounds[1] if bounds is not None else None

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=y,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar={"title": "Normalized value (robust z)"},
            customdata=np.stack([raw], axis=-1),
            hovertemplate="recording=%{y}<br>metric=%{x}<br>raw=%{customdata[0]:.3g}<br>normalized=%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="QA metric summary",
        yaxis_title="Recordings (organized by subject / task / run)",
        template="plotly_white",
        margin={"l": 65, "r": 20, "t": 65, "b": 55},
        height=620,
    )
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(showticklabels=show_ticklabels)
    return fig


def plot_condition_effect_grid(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str]],
    title: str,
) -> Optional[go.Figure]:
    if df.empty:
        return None
    conditions = sorted(df["condition_label"].unique())
    if len(conditions) < 2:
        return None

    valid_specs = [(col, label) for col, label in metric_specs if col in df.columns]
    if not valid_specs:
        return None

    n = len(valid_specs)
    cols = 3
    rows = int(np.ceil(n / cols))
    all_subjects = sorted(df["subject"].dropna().astype(str).unique().tolist())
    color_map = _subject_color_map(all_subjects)
    show_subject_legend = len(all_subjects) <= 10
    subplot_titles = [re.sub(r"\s*\([^)]*\)", "", label).strip() for _, label in valid_specs]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.20,
    )

    for idx, (metric_col, y_label) in enumerate(valid_specs):
        pivot = df.pivot_table(
            index="subject",
            columns="condition_label",
            values=metric_col,
            aggfunc="median",
        )
        if pivot.empty:
            continue
        pivot = pivot.reindex(columns=conditions)
        pivot = pivot.dropna(how="all")
        if pivot.empty:
            continue

        subject_var = pivot.var(axis=1, skipna=True).fillna(-np.inf)
        selected_subjects = subject_var.sort_values(ascending=False).index.tolist()
        if len(selected_subjects) > MAX_SUBJECT_LINES:
            selected_subjects = selected_subjects[:MAX_SUBJECT_LINES]

        row_i = idx // cols + 1
        col_i = idx % cols + 1
        for subject in selected_subjects:
            vals = pivot.loc[subject].to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if np.sum(finite) < 2:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[conditions[i] for i in range(len(conditions)) if finite[i]],
                    y=vals[finite],
                    mode="lines+markers",
                    line={"width": 1.0, "color": color_map.get(str(subject), "#4F6F84")},
                    marker={"size": 4.5, "color": color_map.get(str(subject), "#4F6F84")},
                    name=f"sub-{subject}",
                    legendgroup=f"sub-{subject}",
                    showlegend=(show_subject_legend and idx == 0),
                    hovertemplate=f"sub={subject}<br>task=%{{x}}<br>value=%{{y:.3g}}<extra></extra>",
                ),
                row=row_i,
                col=col_i,
            )

        median_vals = np.nanmedian(pivot.to_numpy(dtype=float), axis=0)
        fig.add_trace(
            go.Scatter(
                x=conditions,
                y=median_vals,
                mode="lines+markers",
                line={"width": 2.8, "color": "#0A2F51"},
                marker={"size": 6, "color": "#0A2F51"},
                name="Median subject profile",
                legendgroup="median-profile",
                showlegend=(idx == 0),
                hovertemplate="task=%{x}<br>median=%{y:.3g}<extra></extra>",
            ),
            row=row_i,
            col=col_i,
        )
        if row_i == rows:
            fig.update_xaxes(title_text="Task", row=row_i, col=col_i)
        else:
            fig.update_xaxes(title_text="", row=row_i, col=col_i)

        short_y = y_label
        if col_i > 1:
            short_y = ""
        fig.update_yaxes(title_text=short_y, row=row_i, col=col_i)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        margin={"l": 54, "r": 20, "t": 126, "b": 230},
        height=max(640, 360 * rows),
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
            "tracegroupgap": 6,
        },
    )
    fig.update_annotations(font={"size": 13})
    return fig


def plot_condition_effect_single(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
) -> Optional[go.Figure]:
    """Render one subject-profile condition plot for a single metric."""
    if df.empty or metric_col not in df.columns:
        return None
    conditions = sorted(df["condition_label"].unique())
    if len(conditions) < 2:
        return None

    pivot = df.pivot_table(
        index="subject",
        columns="condition_label",
        values=metric_col,
        aggfunc="median",
    )
    if pivot.empty:
        return None
    pivot = pivot.reindex(columns=conditions)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return None

    subject_var = pivot.var(axis=1, skipna=True).fillna(-np.inf)
    selected_subjects = subject_var.sort_values(ascending=False).index.tolist()
    if len(selected_subjects) > MAX_SUBJECT_LINES:
        selected_subjects = selected_subjects[:MAX_SUBJECT_LINES]

    all_subjects = sorted(pivot.index.astype(str).tolist())
    color_map = _subject_color_map(all_subjects)
    show_subject_legend = len(selected_subjects) <= 10

    fig = go.Figure()
    for subject in selected_subjects:
        vals = pivot.loc[subject].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if np.sum(finite) < 2:
            continue
        s = str(subject)
        fig.add_trace(
            go.Scatter(
                x=[conditions[i] for i in range(len(conditions)) if finite[i]],
                y=vals[finite],
                mode="lines+markers",
                line={"width": 1.2, "color": color_map.get(s, "#4F6F84")},
                marker={"size": 5, "color": color_map.get(s, "#4F6F84")},
                name=f"sub-{s}",
                legendgroup=f"sub-{s}",
                showlegend=show_subject_legend,
                hovertemplate=f"sub={s}<br>task=%{{x}}<br>value=%{{y:.3g}}<extra></extra>",
            )
        )

    median_vals = np.nanmedian(pivot.to_numpy(dtype=float), axis=0)
    fig.add_trace(
        go.Scatter(
            x=conditions,
            y=median_vals,
            mode="lines+markers",
            line={"width": 3.0, "color": "#0A2F51"},
            marker={"size": 7, "color": "#0A2F51"},
            name="Median subject profile",
            legendgroup="median-profile",
            showlegend=True,
            hovertemplate="task=%{x}<br>median=%{y:.3g}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Task",
        yaxis_title=y_label,
        template="plotly_white",
        margin={"l": 62, "r": 22, "t": 78, "b": 80},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "font": {"size": 11},
        },
    )
    return fig


def plot_subject_epoch_small_multiples(
    profiles_by_subject: Dict[str, List[Dict[str, np.ndarray]]],
    subject_scores: Dict[str, float],
    title: str,
    y_label: str,
    top_n: int = 12,
) -> Optional[go.Figure]:
    if not profiles_by_subject:
        return None

    subjects = [s for s, profiles in profiles_by_subject.items() if profiles]
    if not subjects:
        return None
    subjects = sorted(
        subjects,
        key=lambda s: subject_scores.get(s, float("-inf")),
        reverse=True,
    )
    selected = subjects[:top_n]
    n = len(selected)
    if n == 0:
        return None
    cols = 3
    rows = int(np.ceil(n / cols))
    subplot_titles = [
        f"sub-{sub} (n={len(profiles_by_subject[sub])})"
        for sub in selected
    ]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, horizontal_spacing=0.08, vertical_spacing=0.1)

    for idx, subject in enumerate(selected):
        profile = _aggregate_window_profiles(profiles_by_subject[subject])
        if profile is None:
            continue
        r = idx // cols + 1
        c = idx % cols + 1

        x = np.asarray(profile["x"], dtype=float)
        q05 = np.asarray(profile["q05"], dtype=float)
        q25 = np.asarray(profile["q25"], dtype=float)
        q50 = np.asarray(profile["q50"], dtype=float)
        q75 = np.asarray(profile["q75"], dtype=float)
        q95 = np.asarray(profile["q95"], dtype=float)
        keep = _downsample_indices(x.size, MAX_POINTS_PROFILE)
        x = x[keep]
        q05 = q05[keep]
        q25 = q25[keep]
        q50 = q50[keep]
        q75 = q75[keep]
        q95 = q95[keep]

        fig.add_trace(go.Scatter(x=x, y=q95, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q05, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, fill="tonexty", fillcolor="rgba(88,166,255,0.16)", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q75, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q25, mode="lines", line={"width": 0, "color": "rgba(0,0,0,0)"}, fill="tonexty", fillcolor="rgba(30,136,229,0.22)", showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(x=x, y=q50, mode="lines", line={"width": 1.7, "color": "#164B84"}, showlegend=False), row=r, col=c)

        fig.update_xaxes(title_text="Epoch index", row=r, col=c)
        fig.update_yaxes(title_text=y_label, row=r, col=c)

    fig.update_layout(
        title={"text": title, "x": 0.5},
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 80, "b": 40},
        height=max(420, 260 * rows),
    )
    return fig


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    flat = _robust_normalize_array(arr.reshape(-1))
    return flat.reshape(arr.shape)


def _concat_topomap_payload(
    existing: Optional[TopomapPayload],
    incoming: TopomapPayload,
    name_prefix: str = "",
) -> TopomapPayload:
    in_names = [f"{name_prefix}{n}" for n in incoming.layout.names]
    if existing is None:
        return TopomapPayload(
            layout=SensorLayout(
                x=np.asarray(incoming.layout.x, dtype=float),
                y=np.asarray(incoming.layout.y, dtype=float),
                names=in_names,
                z=(np.asarray(incoming.layout.z, dtype=float) if incoming.layout.z is not None else None),
            ),
            values=np.asarray(incoming.values, dtype=float),
        )
    z_left = np.asarray(existing.layout.z, dtype=float) if existing.layout.z is not None else np.full(existing.layout.x.shape, np.nan, dtype=float)
    z_right = np.asarray(incoming.layout.z, dtype=float) if incoming.layout.z is not None else np.full(np.asarray(incoming.layout.x, dtype=float).shape, np.nan, dtype=float)
    return TopomapPayload(
        layout=SensorLayout(
            x=np.concatenate([np.asarray(existing.layout.x, dtype=float), np.asarray(incoming.layout.x, dtype=float)]),
            y=np.concatenate([np.asarray(existing.layout.y, dtype=float), np.asarray(incoming.layout.y, dtype=float)]),
            names=list(existing.layout.names) + in_names,
            z=np.concatenate([z_left, z_right]),
        ),
        values=np.concatenate([np.asarray(existing.values, dtype=float), np.asarray(incoming.values, dtype=float)]),
    )


def _vstack_nan_padded(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack 2D matrices by rows, padding shorter epoch dimension with NaN."""
    a = np.atleast_2d(np.asarray(left, dtype=float))
    b = np.atleast_2d(np.asarray(right, dtype=float))

    if a.size == 0:
        return b
    if b.size == 0:
        return a

    n_cols = max(a.shape[1], b.shape[1])
    if a.shape[1] < n_cols:
        a = np.pad(a, ((0, 0), (0, n_cols - a.shape[1])), mode="constant", constant_values=np.nan)
    if b.shape[1] < n_cols:
        b = np.pad(b, ((0, 0), (0, n_cols - b.shape[1])), mode="constant", constant_values=np.nan)

    return np.vstack([a, b])


def _combine_accumulators(acc_by_type: Dict[str, ChTypeAccumulator]) -> ChTypeAccumulator:
    """Merge MAG and GRAD accumulators into a cumulative all-channel view.

    This is intentionally cumulative (channel concatenation), not unit
    harmonization. Matrix-like payloads are row-stacked with epoch padding so
    combined tabs can preserve all available channels from each type.

    Only MEG channel types (mag, grad) are merged here. EEG is excluded
    because it has different physical units and should not be mixed into the
    MEG combined view.
    """
    MEG_ONLY_TYPES = ("mag", "grad")
    combined = ChTypeAccumulator()
    std_runs_by_cond_by_type: Dict[str, Dict[str, List[np.ndarray]]] = {
        ch: defaultdict(list) for ch in MEG_ONLY_TYPES
    }
    ptp_runs_by_cond_by_type: Dict[str, Dict[str, List[np.ndarray]]] = {
        ch: defaultdict(list) for ch in MEG_ONLY_TYPES
    }
    for ch_type in MEG_ONLY_TYPES:
        if ch_type not in acc_by_type:
            continue
        acc = acc_by_type[ch_type]
        combined.subjects.update(acc.subjects)
        combined.module_present.update(acc.module_present)
        combined.module_missing.update(acc.module_missing)
        combined.source_paths.update(acc.source_paths)

        for cond, vals in acc.std_dist_by_condition.items():
            combined.std_dist_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, vals in acc.std_dist_mean_by_condition.items():
            combined.std_dist_mean_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, vals in acc.std_dist_upper_by_condition.items():
            combined.std_dist_upper_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, mats in acc.std_heatmap_runs_by_condition.items():
            for mat in mats:
                std_runs_by_cond_by_type[ch_type][cond].append(np.asarray(mat, dtype=float))
        for cond, vals in acc.ptp_dist_by_condition.items():
            combined.ptp_dist_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, vals in acc.ptp_dist_mean_by_condition.items():
            combined.ptp_dist_mean_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, vals in acc.ptp_dist_upper_by_condition.items():
            combined.ptp_dist_upper_by_condition[cond].extend(_finite_array(vals).tolist())
        for cond, mats in acc.ptp_heatmap_runs_by_condition.items():
            for mat in mats:
                ptp_runs_by_cond_by_type[ch_type][cond].append(np.asarray(mat, dtype=float))

        for profile in acc.std_window_profiles:
            combined.std_window_profiles.append(profile)
        for profile in acc.ptp_window_profiles:
            combined.ptp_window_profiles.append(profile)
        for cond, profiles in acc.std_window_profiles_by_condition.items():
            for profile in profiles:
                combined.std_window_profiles_by_condition[cond].append(profile)
        for cond, profiles in acc.ptp_window_profiles_by_condition.items():
            for profile in profiles:
                combined.ptp_window_profiles_by_condition[cond].append(profile)

        for cond, matrix in acc.std_heatmap_by_condition.items():
            # Representative matrices are merged by concatenating channel rows.
            # Epoch axis may differ across runs/ch-types, so we pad columns.
            if cond in combined.std_heatmap_by_condition:
                combined.std_heatmap_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_by_condition[cond],
                    matrix,
                )
            else:
                combined.std_heatmap_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, matrix in acc.std_heatmap_upper_by_condition.items():
            if cond in combined.std_heatmap_upper_by_condition:
                combined.std_heatmap_upper_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_upper_by_condition[cond],
                    matrix,
                )
            else:
                combined.std_heatmap_upper_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, sum_matrix in acc.std_heatmap_sum_by_condition.items():
            if cond in combined.std_heatmap_sum_by_condition:
                combined.std_heatmap_sum_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_sum_by_condition[cond],
                    sum_matrix,
                )
                combined.std_heatmap_count_by_condition[cond] = _vstack_nan_padded(
                    combined.std_heatmap_count_by_condition[cond],
                    acc.std_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                )
            else:
                combined.std_heatmap_sum_by_condition[cond] = np.asarray(sum_matrix, dtype=float)
                combined.std_heatmap_count_by_condition[cond] = np.asarray(
                    acc.std_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                    dtype=float,
                )
        for cond, matrix in acc.ptp_heatmap_by_condition.items():
            if cond in combined.ptp_heatmap_by_condition:
                combined.ptp_heatmap_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_by_condition[cond],
                    matrix,
                )
            else:
                combined.ptp_heatmap_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, matrix in acc.ptp_heatmap_upper_by_condition.items():
            if cond in combined.ptp_heatmap_upper_by_condition:
                combined.ptp_heatmap_upper_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_upper_by_condition[cond],
                    matrix,
                )
            else:
                combined.ptp_heatmap_upper_by_condition[cond] = np.asarray(matrix, dtype=float)
        for cond, sum_matrix in acc.ptp_heatmap_sum_by_condition.items():
            if cond in combined.ptp_heatmap_sum_by_condition:
                combined.ptp_heatmap_sum_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_sum_by_condition[cond],
                    sum_matrix,
                )
                combined.ptp_heatmap_count_by_condition[cond] = _vstack_nan_padded(
                    combined.ptp_heatmap_count_by_condition[cond],
                    acc.ptp_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                )
            else:
                combined.ptp_heatmap_sum_by_condition[cond] = np.asarray(sum_matrix, dtype=float)
                combined.ptp_heatmap_count_by_condition[cond] = np.asarray(
                    acc.ptp_heatmap_count_by_condition.get(cond, np.zeros_like(sum_matrix, dtype=float)),
                    dtype=float,
                )

        for cond, vals in acc.psd_ratio_by_condition.items():
            combined.psd_ratio_by_condition[cond].extend(vals)
        for cond, vals in acc.psd_harmonics_ratio_by_condition.items():
            combined.psd_harmonics_ratio_by_condition[cond].extend(vals)
        combined.psd_profiles.extend(acc.psd_profiles)
        for cond, profiles in acc.psd_profiles_by_condition.items():
            combined.psd_profiles_by_condition[cond].extend(profiles)
        for cond, vals in acc.ecg_corr_by_condition.items():
            combined.ecg_corr_by_condition[cond].extend(vals)
        for cond, vals in acc.eog_corr_by_condition.items():
            combined.eog_corr_by_condition[cond].extend(vals)
        for cond, vals in acc.muscle_scalar_by_condition.items():
            combined.muscle_scalar_by_condition[cond].extend(vals)
        for cond, vals in acc.muscle_mean_by_condition.items():
            combined.muscle_mean_by_condition[cond].extend(vals)
        combined.muscle_profiles.extend(acc.muscle_profiles)
        for cond, profiles in acc.muscle_profiles_by_condition.items():
            combined.muscle_profiles_by_condition[cond].extend(profiles)

        for subject, profiles in acc.std_subject_profiles.items():
            for profile in profiles:
                combined.std_subject_profiles[subject].append(profile)
        for subject, profiles in acc.ptp_subject_profiles.items():
            for profile in profiles:
                combined.ptp_subject_profiles[subject].append(profile)

        topomap_pairs = [
            (acc.std_topomap_by_condition, combined.std_topomap_by_condition),
            (acc.ptp_topomap_by_condition, combined.ptp_topomap_by_condition),
            (acc.psd_topomap_by_condition, combined.psd_topomap_by_condition),
            (acc.ecg_topomap_by_condition, combined.ecg_topomap_by_condition),
            (acc.eog_topomap_by_condition, combined.eog_topomap_by_condition),
        ]
        for src, dst in topomap_pairs:
            for cond, payload in src.items():
                dst[cond] = _concat_topomap_payload(dst.get(cond), payload, name_prefix=f"{ch_type}_")

    def _merge_type_run_matrices(
        by_type: Dict[str, Dict[str, List[np.ndarray]]],
    ) -> Dict[str, List[np.ndarray]]:
        out: Dict[str, List[np.ndarray]] = defaultdict(list)
        conditions = set(by_type.get("mag", {}).keys()) | set(by_type.get("grad", {}).keys())
        for cond in sorted(conditions):
            mag_runs = list(by_type.get("mag", {}).get(cond, []))
            grad_runs = list(by_type.get("grad", {}).get(cond, []))
            if mag_runs and grad_runs:
                pair_n = min(len(mag_runs), len(grad_runs))
                for idx in range(pair_n):
                    out[cond].append(_vstack_nan_padded(mag_runs[idx], grad_runs[idx]))
                for idx in range(pair_n, len(mag_runs)):
                    out[cond].append(np.asarray(mag_runs[idx], dtype=float))
                for idx in range(pair_n, len(grad_runs)):
                    out[cond].append(np.asarray(grad_runs[idx], dtype=float))
            elif mag_runs:
                out[cond].extend(np.asarray(m, dtype=float) for m in mag_runs)
            elif grad_runs:
                out[cond].extend(np.asarray(m, dtype=float) for m in grad_runs)
        return out

    combined.std_heatmap_runs_by_condition = defaultdict(list, _merge_type_run_matrices(std_runs_by_cond_by_type))
    combined.ptp_heatmap_runs_by_condition = defaultdict(list, _merge_type_run_matrices(ptp_runs_by_cond_by_type))
    for cond, mats in combined.std_heatmap_runs_by_condition.items():
        combined.std_epoch_counts_by_condition[cond].extend([int(np.asarray(m).shape[1]) for m in mats if np.asarray(m).ndim == 2])
    for cond, mats in combined.ptp_heatmap_runs_by_condition.items():
        combined.ptp_epoch_counts_by_condition[cond].extend([int(np.asarray(m).shape[1]) for m in mats if np.asarray(m).ndim == 2])

    rows = []
    for ch_type in ("mag", "grad"):
        rows.extend(acc_by_type[ch_type].run_rows)
    df = _run_rows_dataframe(rows)
    if not df.empty:
        combined.run_rows.extend(rows)
        run_level = df.drop_duplicates(subset=["run_key", "condition_label"])
        combined.run_count = int(run_level["run_key"].nunique())
        cond_counts = run_level["condition_label"].value_counts().to_dict()
        for cond, count in cond_counts.items():
            combined.runs_by_condition[str(cond)] += int(count)
    else:
        combined.run_count = 0

    return combined


def _build_global_qa_section(acc: ChTypeAccumulator, tab_name: str, amplitude_unit: str, is_combined: bool) -> str:
    suffix = " (all channels)" if is_combined else ""

    std_violin = plot_violin_channel_distribution(
        acc.std_dist_by_condition,
        f"STD channel distribution by task{suffix}",
        f"STD summary ({amplitude_unit})",
    )
    std_profile = plot_quantile_band_timecourse(
        _aggregate_window_profiles(acc.std_window_profiles),
        f"STD epoch profile (channel quantile bands){suffix}",
        "Global pooled epoch index",
        f"STD ({amplitude_unit})",
    )
    std_heatmap_variants = _heatmap_variants_by_condition_from_acc(
        acc.std_heatmap_runs_by_condition,
        acc.std_heatmap_by_condition,
        acc.std_heatmap_sum_by_condition,
        acc.std_heatmap_count_by_condition,
        acc.std_heatmap_upper_by_condition,
    )
    std_epoch_notes = _epoch_consistency_notes(acc.std_heatmap_runs_by_condition)
    std_heatmaps = {
        cond: plot_heatmap_sorted_channels_windows(
            variants,
            title=f"STD channel-by-epoch footprint ({cond}){suffix}",
            color_title=f"STD ({amplitude_unit})",
            summary_mode="median",
            channel_names=(
                acc.std_topomap_by_condition[cond].layout.names
                if cond in acc.std_topomap_by_condition
                else None
            ),
        )
        for cond, variants in sorted(std_heatmap_variants.items())
    }
    std_topomaps = _topomap_blocks(
        payloads_by_condition=acc.std_topomap_by_condition,
        title_prefix=f"STD spatial footprint (median over epochs){suffix}",
        color_title=f"STD ({amplitude_unit})",
        interpretation=(
            f"Axes: spatial coordinates from stored channel geometry; color in {amplitude_unit}. "
            "Each marker is one channel summary aggregated across recordings in this task (Global QA). "
            "Typical appearance: smooth spatial gradients. Suspicious appearance: localized hotspots. "
            "Limitation: Global QA does not identify subjects."
        ),
    )

    ptp_violin = plot_violin_channel_distribution(
        acc.ptp_dist_by_condition,
        f"PtP channel distribution by task{suffix}",
        f"PtP summary ({amplitude_unit})",
    )
    ptp_profile = plot_quantile_band_timecourse(
        _aggregate_window_profiles(acc.ptp_window_profiles),
        f"PtP epoch profile (channel quantile bands){suffix}",
        "Global pooled epoch index",
        f"PtP amplitude ({amplitude_unit})",
    )
    ptp_heatmap_variants = _heatmap_variants_by_condition_from_acc(
        acc.ptp_heatmap_runs_by_condition,
        acc.ptp_heatmap_by_condition,
        acc.ptp_heatmap_sum_by_condition,
        acc.ptp_heatmap_count_by_condition,
        acc.ptp_heatmap_upper_by_condition,
    )
    ptp_epoch_notes = _epoch_consistency_notes(acc.ptp_heatmap_runs_by_condition)
    ptp_heatmaps = {
        cond: plot_heatmap_sorted_channels_windows(
            variants,
            title=f"PtP channel-by-epoch footprint ({cond}){suffix}",
            color_title=f"PtP ({amplitude_unit})",
            summary_mode="upper_tail",
            channel_names=(
                acc.ptp_topomap_by_condition[cond].layout.names
                if cond in acc.ptp_topomap_by_condition
                else None
            ),
        )
        for cond, variants in sorted(ptp_heatmap_variants.items())
    }
    ptp_topomaps = _topomap_blocks(
        payloads_by_condition=acc.ptp_topomap_by_condition,
        title_prefix=f"PtP spatial footprint (95th percentile over epochs){suffix}",
        color_title=f"PtP ({amplitude_unit})",
        interpretation=(
            f"Axes: spatial coordinates from stored channel geometry; color in {amplitude_unit}. "
            "Marker color shows per-channel upper-tail PtP amplitude aggregated across recordings in this task (Global QA). "
            "Typical appearance: moderate spatial spread. Suspicious appearance: persistent hotspots. "
            "Limitation: Global QA does not identify subjects."
        ),
    )

    psd_profile = plot_psd_median_band(
        _aggregate_psd_profiles(acc.psd_profiles),
        "PSD median profile with channel quantile bands",
    )
    psd_violin = plot_violin_channel_distribution(
        acc.psd_ratio_by_condition,
        "Mains relative power distribution by task",
        "Mains relative power (unitless)",
    )
    psd_topomaps = _topomap_blocks(
        payloads_by_condition=acc.psd_topomap_by_condition,
        title_prefix="Mains relative power spatial footprint",
        color_title="Mains ratio",
        interpretation=(
            "Axes: spatial coordinates from stored channel geometry; color is unitless relative power. "
            "Each marker is one channel aggregated across recordings in this task (Global QA). "
            "Typical appearance: mild spatial variation. Suspicious appearance: strong localized hotspots. "
            "Limitation: Global QA does not identify subjects."
        ),
    )

    ecg_violin = plot_violin_channel_distribution(
        acc.ecg_corr_by_condition,
        "ECG correlation magnitude distribution by task",
        "|r| (unitless)",
    )
    eog_violin = plot_violin_channel_distribution(
        acc.eog_corr_by_condition,
        "EOG correlation magnitude distribution by task",
        "|r| (unitless)",
    )
    muscle_profile = plot_quantile_band_timecourse(
        _aggregate_muscle_profiles(acc.muscle_profiles),
        "Muscle score epoch profile",
        "Global pooled epoch index",
        "Muscle score (z-score, unitless)",
    )
    muscle_violin = plot_violin_channel_distribution(
        acc.muscle_scalar_by_condition,
        "Muscle upper-tail distribution by task",
        "Muscle score (z-score, unitless)",
    )

    std_ecdf = _make_ecdf_figure(acc.std_dist_by_condition, f"STD ECDF{suffix}", f"STD ({amplitude_unit})")
    ptp_ecdf = _make_ecdf_figure(acc.ptp_dist_by_condition, f"PtP ECDF{suffix}", f"PtP ({amplitude_unit})")
    psd_ecdf = _make_ecdf_figure(acc.psd_ratio_by_condition, "Mains relative power ECDF", "Mains ratio")
    ecg_ecdf = _make_ecdf_figure(acc.ecg_corr_by_condition, "ECG |r| ECDF", "|r|")
    eog_ecdf = _make_ecdf_figure(acc.eog_corr_by_condition, "EOG |r| ECDF", "|r|")
    muscle_ecdf = _make_ecdf_figure(acc.muscle_scalar_by_condition, "Muscle score ECDF", "Muscle score")

    std_heatmap_blocks = _condition_figure_blocks(
        std_heatmaps,
        (
            f"Axes: x is epoch index, y is sorted channel index; color is STD in {amplitude_unit}. "
            "Each heatmap cell represents the STD of one channel in one epoch, aggregated across recordings. "
            "Typical appearance: moderate contrast without persistent vertical stripes. "
            "Suspicious appearance: repeated vertical stripes and heavy upper tails. "
            "Global pooled epoch index is not aligned in time across recordings."
        ),
        interpretation_by_condition=std_epoch_notes,
    )
    ptp_heatmap_blocks = _condition_figure_blocks(
        ptp_heatmaps,
        (
            f"Axes: x is epoch index, y is sorted channel index; color is PtP amplitude in {amplitude_unit}. "
            "Heatmap cell = PtP value for one channel in one epoch, aggregated across recordings. "
            "Typical appearance: moderate spread. Suspicious appearance: strong upper-tail bursts and vertical stripes. "
            "Global pooled epoch index is not aligned in time across recordings."
        ),
        interpretation_by_condition=ptp_epoch_notes,
    )

    return (
        "<section>"
        "<h2>Global QA</h2>"
        "<p>Aggregated distributions/footprints across all recordings in this dataset (all subjects x all runs).</p>"
        "<div class='metric-block'>"
        "<h3>STD</h3>"
        + _figure_block(
            std_violin,
            (
                f"Y-axis is STD in {amplitude_unit}; x-axis is task. "
                "Each observation represents one channel summarized across epochs for one recording. "
                "Typical appearance: compact distribution with moderate upper tail. "
                "Suspicious appearance: heavy upper tail where few channels dominate. Global QA does not identify subjects."
            ),
        )
        + _figure_block(
            std_profile,
            (
                f"Y-axis is STD in {amplitude_unit}; x-axis is global pooled epoch index. "
                "This profile is computed per epoch by pooling channels across recordings (global footprint). "
                "Typical appearance: stable median with moderate band width. Suspicious appearance: persistent wide bands or repeated peaks. "
                "Global pooled epoch index is not aligned in time across recordings."
            ),
        )
        + std_heatmap_blocks
        + std_topomaps
        + "</div>"
        "<div class='metric-block'>"
        "<h3>PtP</h3>"
        + _figure_block(
            ptp_violin,
            (
                f"Y-axis is PtP amplitude in {amplitude_unit}; x-axis is task. "
                "Each observation represents one channel summarized across epochs for one recording. "
                "Typical appearance: moderate spread. Suspicious appearance: heavy upper tail and task-dependent broadening."
            ),
        )
        + _figure_block(
            ptp_profile,
            (
                f"Y-axis is PtP amplitude in {amplitude_unit}; x-axis is global pooled epoch index. "
                "Line = median across channels per epoch; bands = IQR and 5-95% across channels. "
                "Typical appearance: stable median and moderate band width. Suspicious appearance: bursty upper tails."
            ),
        )
        + ptp_heatmap_blocks
        + ptp_topomaps
        + "</div>"
        "<div class='metric-block'>"
        "<h3>PSD</h3>"
        + _figure_block(
            psd_profile,
            (
                "X-axis is frequency in Hz; y-axis is power spectral density (relative power). "
                "Each profile summarizes channels pooled across recordings (Global QA). "
                "Typical appearance: smooth broad-band shape. Suspicious appearance: strong mains peaks and harmonics."
            ),
        )
        + _figure_block(
            psd_violin,
            (
                "Y-axis is mains relative power (unitless). "
                "Each observation represents one channel summarized for one recording. "
                "Typical appearance: moderate distribution. Suspicious appearance: heavy upper tail with strong mains contribution."
            ),
        )
        + psd_topomaps
        + "</div>"
        "<div class='metric-block'>"
        "<h3>ECG / EOG</h3>"
        + _figure_block(
            ecg_violin,
            (
                "Y-axis is |r| (unitless). Each observation represents one channel summary for one recording. "
                "Typical appearance: moderate correlation magnitude distribution. Suspicious appearance: broad upper tails."
            ),
        )
        + _figure_block(
            eog_violin,
            (
                "Y-axis is |r| (unitless). Each observation represents one channel summary for one recording. "
                "Typical appearance: moderate correlation magnitude distribution. Suspicious appearance: pronounced upper tails and spatial hotspots."
            ),
        )
        + "</div>"
        "<div class='metric-block'>"
        "<h3>Muscle</h3>"
        + _figure_block(
            muscle_profile,
            (
                "Y-axis is muscle score (z-score, unitless); x-axis is global pooled epoch index. "
                "Line = median across channels per epoch; bands = IQR and 5-95% across channels. "
                "Typical appearance: stable profile. Suspicious appearance: sustained high upper tail and bursts."
            ),
        )
        + _figure_block(
            muscle_violin,
            (
                "Y-axis is muscle score (z-score, unitless). "
                "Each observation is one recording-level upper-tail summary. "
                "Typical appearance: compact distribution. Suspicious appearance: heavy upper tail."
            ),
        )
        + "</div>"
        "<div class='metric-block'>"
        "<h3>QA Distributions (Statistical)</h3>"
        + _figure_block(std_ecdf, "ECDF of global channel summaries. Right-shifted curves indicate larger values.")
        + _figure_block(ptp_ecdf, "ECDF of global channel summaries. Right-shifted curves indicate larger values.")
        + _figure_block(psd_ecdf, "ECDF of global mains relative power by channel.")
        + _figure_block(ecg_ecdf, "ECDF of global ECG correlation magnitude by channel.")
        + _figure_block(eog_ecdf, "ECDF of global EOG correlation magnitude by channel.")
        + _figure_block(muscle_ecdf, "ECDF of global recording-level muscle summaries.")
        + "</div>"
        "</section>"
    )


def _subject_scores_from_df(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    if df.empty or metric_col not in df.columns:
        return {}
    out = {}
    for subject, group in df.groupby("subject"):
        vals = _finite_array(group[metric_col].to_numpy(dtype=float))
        out[str(subject)] = float(np.nanmedian(vals)) if vals.size else float("nan")
    return out


def _build_subject_qa_section(acc: ChTypeAccumulator, tab_name: str, amplitude_unit: str) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>Subject QA</h2><p>No recording-level summaries are available.</p></section>"

    std_violin = plot_subject_condition_violin(
        df, "std_median",
        "STD by task with subject-aware points",
        f"STD ({amplitude_unit})",
    )
    ptp_violin = plot_subject_condition_violin(
        df, "ptp_upper_tail",
        "PtP by task with subject-aware points",
        f"PtP ({amplitude_unit})",
    )
    ecg_violin = plot_subject_condition_violin(
        df, "ecg_mean_abs_corr",
        "ECG correlation magnitude by task with subject-aware points",
        "|r| (unitless)",
    )
    eog_violin = plot_subject_condition_violin(
        df, "eog_mean_abs_corr",
        "EOG correlation magnitude by task with subject-aware points",
        "|r| (unitless)",
    )
    mains_violin = plot_subject_condition_violin(
        df, "mains_ratio",
        "Mains relative power by task with subject-aware points",
        "Mains ratio (unitless)",
    )

    std_scatter = plot_run_fingerprint_scatter(
        df,
        "std_median",
        "std_upper_tail",
        "Recording fingerprint scatter (STD)",
        f"Median channel STD per recording ({amplitude_unit})",
        f"Upper-tail channel STD per recording ({amplitude_unit})",
    )
    ptp_scatter = plot_run_fingerprint_scatter(
        df,
        "ptp_median",
        "ptp_upper_tail",
        "Recording fingerprint scatter (PtP)",
        f"Median channel PtP per recording ({amplitude_unit})",
        f"Upper-tail channel PtP per recording ({amplitude_unit})",
    )
    mains_scatter = plot_run_fingerprint_scatter(
        df,
        "mains_ratio",
        "mains_harmonics_ratio",
        "Recording fingerprint scatter (PSD mains)",
        "Mains relative power (unitless)",
        "Harmonics relative power (unitless)",
    )
    ecg_scatter = plot_run_fingerprint_scatter(
        df,
        "ecg_mean_abs_corr",
        "ecg_p95_abs_corr",
        "Recording fingerprint scatter (ECG)",
        "Mean |ECG correlation| per recording",
        "p95 |ECG correlation| per recording",
    )
    eog_scatter = plot_run_fingerprint_scatter(
        df,
        "eog_mean_abs_corr",
        "eog_p95_abs_corr",
        "Recording fingerprint scatter (EOG)",
        "Mean |EOG correlation| per recording",
        "p95 |EOG correlation| per recording",
    )
    muscle_scatter = plot_run_fingerprint_scatter(
        df,
        "muscle_median",
        "muscle_p95",
        "Recording fingerprint scatter (Muscle)",
        "Median muscle score per recording",
        "p95 muscle score per recording",
    )

    ranking = plot_subject_ranking_table(df, "Subject ranking by robust QA summaries")
    metric_heatmap = plot_subject_metric_heatmap(df, "Subject-by-metric normalized summary heatmap")
    condition_effect = plot_subject_condition_effect(
        df,
        metric_col="std_median",
        title="Task effect per subject (STD median)",
        y_label=f"STD ({amplitude_unit})",
    )

    std_subject_profiles = plot_subject_epoch_small_multiples(
        acc.std_subject_profiles,
        _subject_scores_from_df(df, "std_upper_tail"),
        "Per-subject STD epoch profile small multiples (top 12 by upper-tail summary)",
        y_label=f"STD ({amplitude_unit})",
        top_n=12,
    )
    ptp_subject_profiles = plot_subject_epoch_small_multiples(
        acc.ptp_subject_profiles,
        _subject_scores_from_df(df, "ptp_upper_tail"),
        "Per-subject PtP epoch profile small multiples (top 12 by upper-tail summary)",
        y_label=f"PtP ({amplitude_unit})",
        top_n=12,
    )

    return (
        "<section>"
        "<h2>Subject QA</h2>"
        "<p>Subject-aware summaries highlight recordings/subjects with atypical distributions without applying thresholds.</p>"
        "<div class='metric-block'>"
        "<h3>Subject-aware distributions</h3>"
        + _figure_block(
            std_violin,
            (
                f"Y-axis is STD in {amplitude_unit}; x-axis is task. "
                "Each point represents one recording summary and hover includes BIDS entities. "
                "Typical appearance: moderate spread per task. Suspicious appearance: heavy upper tails or subject clusters away from peers."
            ),
        )
        + _figure_block(
            ptp_violin,
            (
                f"Y-axis is PtP in {amplitude_unit}; x-axis is task. "
                "Each point represents one recording summary and is colored by subject. "
                "Typical appearance: compact subject overlap. Suspicious appearance: repeated subject-specific upper tails."
            ),
        )
        + _figure_block(ecg_violin, "Y-axis is |r| (unitless); each point is one recording summary. Look for consistent subject-level shifts.")
        + _figure_block(eog_violin, "Y-axis is |r| (unitless); each point is one recording summary. Look for task-specific and subject-specific shifts.")
        + _figure_block(mains_violin, "Y-axis is mains ratio (unitless); each point is one recording summary. Heavy upper tails indicate stronger mains burden.")
        + "</div>"
        "<div class='metric-block'>"
        "<h3>Recording fingerprint scatter</h3>"
        + _figure_block(
            std_scatter,
            (
                "Each point = one recording (sub x task x run), colored by subject and symbolized by task. "
                "Axes represent central and upper-tail STD summaries. "
                "Typical appearance: compact cloud with moderate spread. Suspicious appearance: isolated points with strong upper-tail displacement."
            ),
        )
        + _figure_block(ptp_scatter, "Each point = one recording. Axes represent central and upper-tail PtP summaries.")
        + _figure_block(mains_scatter, "Each point = one recording. Axes summarize mains and harmonics relative power.")
        + _figure_block(ecg_scatter, "Each point = one recording. Axes summarize mean and upper-tail ECG correlation magnitude.")
        + _figure_block(eog_scatter, "Each point = one recording. Axes summarize mean and upper-tail EOG correlation magnitude.")
        + _figure_block(muscle_scatter, "Each point = one recording. Axes summarize median and upper-tail muscle scores.")
        + "</div>"
        "<div class='metric-block'>"
        "<h3>Subject ranking and footprints</h3>"
        + _figure_block(
            ranking,
            (
                "Table values are robust per-subject summaries with recording counts. "
                "This table ranks distribution footprints only and does not imply binary decisions."
            ),
        )
        + _figure_block(
            std_subject_profiles,
            (
                f"Each panel shows one subject: median and quantile bands across channels per epoch (STD in {amplitude_unit}). "
                "Panels are limited to top 12 subjects by upper-tail summary for scalability."
            ),
        )
        + _figure_block(
            ptp_subject_profiles,
            (
                f"Each panel shows one subject: median and quantile bands across channels per epoch (PtP in {amplitude_unit}). "
                "Panels are limited to top 12 subjects by upper-tail summary for scalability."
            ),
        )
        + _figure_block(metric_heatmap, "Rows are subjects and columns are metrics. Values are normalized summaries (z-score) for quick multi-metric comparison.")
        + _figure_block(condition_effect, "Paired condition lines per subject help visualize task-related shifts in summary metrics.")
        + "</div>"
        "</section>"
    )


def _summary_shared_controls_panel_html(summary_group_id: str) -> str:
    """Single shared controller panel for the six summary-strip plots."""

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


def _build_summary_distributions_section(
    acc: ChTypeAccumulator,
    amplitude_unit: str,
    is_combined: bool,
    *,
    tab_token: str = "tab",
) -> str:
    """Quick pooled distributions and pooled topomaps across all recordings."""
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return (
            "<section>"
            "<h2>Summary distributions</h2>"
            "<p>No run-level summaries are available for this tab.</p>"
            "</section>"
        )

    summary_group_id = f"summary-dist-{tab_token}"
    def _finite_count_map(values_by_condition: Dict[str, List[float]]) -> int:
        total = 0
        for vals in values_by_condition.values():
            arr = np.asarray(vals, dtype=float).reshape(-1)
            total += int(np.isfinite(arr).sum())
        return int(total)

    std_nch_total = _finite_count_map(acc.std_dist_by_condition)
    ptp_nch_total = _finite_count_map(acc.ptp_dist_by_condition)
    psd_nch_total = _finite_count_map(acc.psd_ratio_by_condition)
    ecg_nch_total = _finite_count_map(acc.ecg_corr_by_condition)
    eog_nch_total = _finite_count_map(acc.eog_corr_by_condition)
    muscle_nch_total = _finite_count_map(acc.muscle_scalar_by_condition)
    metric_defs = [
        ("STD", "std_mean", f"STD ({amplitude_unit})", "#2A6FBB", std_nch_total),
        ("PtP", "ptp_mean", f"PtP ({amplitude_unit})", "#2B9348", ptp_nch_total),
        ("PSD", "mains_ratio", "Mains relative power", "#E07A00", psd_nch_total),
        ("ECG", "ecg_mean_abs_corr", "ECG |r| mean", "#C23B22", ecg_nch_total),
        ("EOG", "eog_mean_abs_corr", "EOG |r| mean", "#7B5CC4", eog_nch_total),
        ("Muscle", "muscle_mean", "Muscle mean score", "#0B8F8C", muscle_nch_total),
    ]

    def _metric_cards(defs: Sequence[Tuple[str, str, str, str, Optional[int]]]) -> str:
        cards: List[str] = []
        for metric_name, col, y_label, color, nch_total in defs:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) if col in df.columns else np.array([])
            n_rec = int(np.isfinite(vals).sum())
            n_ch_label = str(int(nch_total)) if int(nch_total) > 0 else "n/a"
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
                + f"<div class='summary-card-meta'>N recordings={n_rec}; N channels={n_ch_label}</div>"
                + "<div class='fig summary-strip-fig'>"
                + _figure_to_div(fig, include_axis_size_control=False, include_plot_controls=False)
                + "</div>"
                + "</div>"
            )
        return f"<div class='summary-dist-grid' data-summary-group='{summary_group_id}'>" + "".join(cards) + "</div>"

    topo_specs = [
        ("STD", acc.std_topomap_by_condition, acc.std_topomap_count_by_condition, f"STD ({amplitude_unit})"),
        ("PtP", acc.ptp_topomap_by_condition, acc.ptp_topomap_count_by_condition, f"PtP ({amplitude_unit})"),
        ("PSD", acc.psd_topomap_by_condition, acc.psd_topomap_count_by_condition, "Mains relative power"),
        ("ECG", acc.ecg_topomap_by_condition, acc.ecg_topomap_count_by_condition, "ECG |r| mean"),
        ("EOG", acc.eog_topomap_by_condition, acc.eog_topomap_count_by_condition, "EOG |r| mean"),
        ("Muscle", {}, None, "Muscle mean score"),
    ]
    topo_tabs: List[Tuple[str, str]] = []
    for idx, (metric_name, payloads, counts, color_title) in enumerate(topo_specs):
        pooled = _pooled_topomap_payload(payloads, counts)
        if pooled is None:
            topo_tabs.append((metric_name, "<p>Topographic maps are not available for this metric in stored outputs.</p>"))
            continue
        fig2d = plot_topomap_if_available(
            pooled,
            title=f"{metric_name} pooled channel footprint",
            color_title=color_title,
        )
        fig3d = plot_topomap_3d_if_available(
            pooled,
            title=f"{metric_name} pooled channel footprint (3D)",
            color_title=color_title,
        )
        views: List[Tuple[str, str]] = []
        if fig2d is not None:
            views.append(
                (
                    "2D",
                    _figure_block(
                        fig2d,
                        "Each channel value is pooled across all recordings and tasks in this tab.",
                    ),
                )
            )
        if fig3d is not None:
            views.append(
                (
                    "3D",
                    _figure_block(
                        fig3d,
                        "3D view of pooled channel values across all recordings and tasks in this tab.",
                    ),
                )
            )
        topo_tabs.append(
            (
                metric_name,
                _build_subtabs_html(
                    f"summary-topo-{tab_token}-{re.sub(r'[^a-z0-9]+', '-', metric_name.lower())}-{idx}",
                    views,
                    level=3,
                ) if views else "<p>Topographic maps are not available for this metric.</p>",
            )
        )

    env_defs = metric_defs[:3]
    phys_defs = metric_defs[3:]
    return (
        "<section>"
        "<h2>Summary distributions</h2>"
        "<p>Task-agnostic compact summary strip.</p>"
        f"<div data-summary-group='{summary_group_id}'>"
        "<div class='summary-category-block'>"
        "<h3>Environmental/Hardware noise</h3>"
        + _metric_cards(env_defs)
        + "</div>"
        "<div class='summary-category-block'>"
        "<h3>Physiological noise</h3>"
        + _metric_cards(phys_defs)
        + "</div>"
        + "</div>"
        + _summary_shared_controls_panel_html(summary_group_id)
        + (
            "<div class='summary-shared-note'><strong>How to interpret:</strong> "
            "Each panel is one metric. One dot is one recording summary (subject x task x run); hover shows recording entities. "
            "Violin + box represent pooled recording distribution. "
            "Units: STD/PtP use amplitude units shown on each y-axis, PSD is mains relative power, ECG/EOG are |r| magnitudes, and Muscle is score."
            "</div>"
        )
        + "<h3>Pooled channel topomaps</h3>"
        + _build_subtabs_html(f"summary-topo-main-{tab_token}", topo_tabs, level=2)
        + "</section>"
    )


def _build_cohort_overview_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    token = "combined" if is_combined else ("grad" if "/m" in amplitude_unit else "mag")

    suffix = " (all channels)" if is_combined else ""
    amp_label = "all channels" if is_combined else amplitude_unit
    metric_specs = [
        ("std_upper_tail", f"STD upper tail ({amp_label})"),
        ("ptp_upper_tail", f"PtP upper tail ({amp_label})"),
        ("mains_ratio", "Mains relative power"),
        ("ecg_p95_abs_corr", "ECG |r| upper tail"),
        ("eog_p95_abs_corr", "EOG |r| upper tail"),
        ("muscle_p95", "Muscle upper tail"),
    ]
    if df.empty:
        return (
            "<section><h2>Cohort QA overview</h2>"
            + _summary_table_html(acc)
            + "<h3>Machine-readable derivatives used</h3>"
            + _paths_html(acc.source_paths)
            + "</section>"
        )

    overview_heatmap = plot_recording_metric_heatmap(df, metric_specs=metric_specs, title=f"Recording-by-metric cohort overview{suffix}")
    subject_matrix = plot_subject_metric_heatmap(df, f"Subject-by-metric normalized summary matrix{suffix} (raw values on hover)")
    ranking = plot_subject_ranking_table(df, "Subject ranking by robust summary footprint")
    std_profiles = plot_subject_epoch_small_multiples(
        acc.std_subject_profiles,
        _subject_scores_from_df(df, "std_upper_tail"),
        f"Top subject epoch profiles (STD){suffix}",
        y_label=f"STD ({amplitude_unit})",
        top_n=12,
    )
    ptp_profiles = plot_subject_epoch_small_multiples(
        acc.ptp_subject_profiles,
        _subject_scores_from_df(df, "ptp_upper_tail"),
        f"Top subject epoch profiles (PtP){suffix}",
        y_label=f"PtP ({amplitude_unit})",
        top_n=12,
    )

    heatmap_tabs = _build_subtabs_html(
        f"cohort-heatmaps-{token}",
        [
            (
                "Recording-by-metric",
                _figure_block(
                    overview_heatmap,
                    (
                        "Rows are recordings and columns are metric summaries. "
                        "Color is robustly normalized for readability across mixed scales; hover reports the raw value in metric units."
                    ),
                ),
            ),
            (
                "Subject-by-metric",
                _figure_block(
                    subject_matrix,
                    (
                        "Rows are subjects and columns are metrics. "
                        "Color is normalized per metric to support cross-metric contrast; hover keeps raw summaries."
                    ),
                ),
            ),
        ],
        level=2,
    )
    profile_tabs = _build_subtabs_html(
        f"cohort-profiles-{token}",
        [
            (
                "STD",
                _figure_block(
                    std_profiles,
                    "Each panel is one subject, showing epoch-wise channel quantile bands for STD. Panels are ordered by subject upper-tail summary.",
                ),
            ),
            (
                "PtP",
                _figure_block(
                    ptp_profiles,
                    "Each panel is one subject, showing epoch-wise channel quantile bands for PtP. Panels are ordered by subject upper-tail summary.",
                ),
            ),
        ],
        level=2,
    )

    return (
        "<section>"
        "<h2>Cohort QA overview</h2>"
        "<p>This section merges cohort metadata, ranking, normalized cohort maps, and top subject epoch footprints for integrated QA triage.</p>"
        + _summary_table_html(acc)
        + _figure_block(
            ranking,
            (
                "Rows are subjects ranked by robust aggregated summaries and recording count. "
                "Higher rank indicates larger relative burden across one or more metrics."
            ),
        )
        + "<h3>Cohort matrices</h3>"
        + heatmap_tabs
        + "<h3>Top subject epoch profiles</h3>"
        + profile_tabs
        + "<h3>Machine-readable derivatives used</h3>"
        + _paths_html(acc.source_paths)
        + "</section>"
    )


def _build_condition_effect_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>QA metrics across tasks</h2><p>No task comparison is available.</p></section>"

    if "condition_label" not in df.columns or df["condition_label"].nunique() < 2:
        return "<section><h2>QA metrics across tasks</h2><p>Task comparison requires at least two task labels.</p></section>"

    suffix = " (all channels)" if is_combined else ""
    token = "combined" if is_combined else ("grad" if "/m" in amplitude_unit else "mag")

    def _variant_order(default_label: str) -> List[str]:
        base = ["Median", "Mean", "Upper tail"]
        return [default_label] + [lbl for lbl in base if lbl != default_label]

    metric_defs = [
        {
            "name": "STD",
            "default": "Median",
            "variants": {
                "Median": ("std_median", f"STD median ({amplitude_unit})"),
                "Mean": ("std_mean", f"STD mean ({amplitude_unit})"),
                "Upper tail": ("std_upper_tail", f"STD upper tail ({amplitude_unit})"),
            },
        },
        {
            "name": "PtP",
            "default": "Upper tail",
            "variants": {
                "Median": ("ptp_median", f"PtP median ({amplitude_unit})"),
                "Mean": ("ptp_mean", f"PtP mean ({amplitude_unit})"),
                "Upper tail": ("ptp_upper_tail", f"PtP upper tail ({amplitude_unit})"),
            },
        },
        {
            "name": "PSD mains ratio",
            "default": "Mean",
            "variants": {
                "Median": ("mains_ratio", "Mains relative power"),
                "Mean": ("mains_ratio", "Mains relative power"),
                "Upper tail": ("mains_harmonics_ratio", "Harmonics relative power"),
            },
        },
        {
            "name": "ECG correlation",
            "default": "Mean",
            "variants": {
                "Median": ("ecg_mean_abs_corr", "ECG |r| median proxy"),
                "Mean": ("ecg_mean_abs_corr", "ECG |r| mean"),
                "Upper tail": ("ecg_p95_abs_corr", "ECG |r| upper tail"),
            },
        },
        {
            "name": "EOG correlation",
            "default": "Mean",
            "variants": {
                "Median": ("eog_mean_abs_corr", "EOG |r| median proxy"),
                "Mean": ("eog_mean_abs_corr", "EOG |r| mean"),
                "Upper tail": ("eog_p95_abs_corr", "EOG |r| upper tail"),
            },
        },
        {
            "name": "Muscle score",
            "default": "Median",
            "variants": {
                "Median": ("muscle_median", "Muscle median"),
                "Mean": ("muscle_mean", "Muscle mean"),
                "Upper tail": ("muscle_p95", "Muscle upper tail"),
            },
        },
    ]

    metric_tabs: List[Tuple[str, str]] = []
    for metric in metric_defs:
        metric_name = str(metric["name"])
        variants = metric["variants"]
        default = str(metric["default"])
        variant_tabs: List[Tuple[str, str]] = []
        for variant_label in _variant_order(default):
            col, y_label = variants.get(variant_label, ("", ""))
            fig = plot_condition_effect_single(
                df,
                metric_col=col,
                title=f"{metric_name} across tasks{suffix} [{variant_label}]",
                y_label=y_label,
            )
            interpretation = (
                f"Each thin line is one subject profile across task labels for the {variant_label.lower()} summary. "
                "The dark line is the cohort median profile across subjects. "
                "Consistent separation between task labels indicates a task-linked shift in this QA summary."
            )
            variant_tabs.append((variant_label, _figure_block(fig, interpretation)))
        metric_tabs.append(
            (
                metric_name,
                _build_subtabs_html(
                    f"task-effect-{token}-{re.sub(r'[^a-z0-9]+', '-', metric_name.lower())}",
                    variant_tabs,
                    level=3,
                ),
            )
        )

    return (
        "<section>"
        "<h2>QA metrics across tasks</h2>"
        "<p>One tab per metric. Inside each metric, switch between median, mean, and upper-tail summaries to inspect how subject trajectories change across task labels.</p>"
        + _build_subtabs_html(f"task-effects-main-{token}", metric_tabs, level=2)
        + "</section>"
    )


def _build_metric_details_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    suffix = " (all channels)" if is_combined else ""
    df = _run_rows_dataframe(acc.run_rows)
    tab_token = "combined" if is_combined else ("grad" if "/m" in amplitude_unit else "mag")

    VariantSpec = Tuple[str, Dict[str, List[float]], str]

    def _count_map_values(values_map: Dict[str, List[float]]) -> int:
        return int(sum(_finite_array(vals).size for vals in values_map.values()))

    def _panel_counts_html(
        point_col: str,
        rec_values: Dict[str, List[float]],
        ch_values: Dict[str, List[float]],
        epoch_values: Dict[str, List[float]],
    ) -> str:
        if (point_col in df.columns) and (not df.empty):
            finite_mask = np.isfinite(pd.to_numeric(df[point_col], errors="coerce").to_numpy(dtype=float))
            dff = df.loc[finite_mask].copy()
        else:
            dff = df.copy()
        n_subjects = int(dff["subject"].nunique()) if ("subject" in dff.columns and not dff.empty) else 0
        n_runs = int(dff["run_key"].nunique()) if ("run_key" in dff.columns and not dff.empty) else 0
        n_channels = _count_map_values(ch_values)
        n_epochs = _count_map_values(epoch_values)
        n_recording_values = _count_map_values(rec_values)
        return (
            f"<strong>Counts:</strong> N subjects={n_subjects}, N recordings={n_runs}, "
            f"N recording values={n_recording_values}, N channels={n_channels}, N epochs={n_epochs}."
        )

    def _panel_subtitle(panel_label: str, formula_text: str, counts_html: str) -> str:
        return (
            "<div class='fig-note'>"
            f"<strong>{panel_label} summary definition:</strong> {formula_text}<br>"
            f"{counts_html}"
            "</div>"
        )

    def _variant_tabs(
        group_id: str,
        values_map: Dict[str, List[float]],
        point_col: str,
        variant_label: str,
        title_prefix: str,
        value_label: str,
    ) -> str:
        vals = _values_with_task_agnostic(values_map)
        if not vals:
            return "<p>No values are available for this variant.</p>"
        point_col_safe = point_col if point_col in df.columns else "__none__"
        box_raw = plot_box_with_subject_jitter(
            vals,
            df,
            point_col=point_col_safe,
            title=f"{title_prefix} - boxplot",
            y_title=value_label,
        )
        violin_density = plot_violin_with_subject_jitter(
            vals,
            df,
            point_col=point_col_safe,
            title=f"{title_prefix} - violin densities",
            y_title=value_label,
            show_density_ridge=True,
            spanmode="soft",
        )
        hist = plot_histogram_distribution(
            vals,
            title=f"{title_prefix} - histogram",
            x_title=value_label,
        )
        dens = plot_density_distribution(
            vals,
            title=f"{title_prefix} - density",
            x_title=value_label,
        )
        return _build_subtabs_html(
            group_id,
            [
                (
                    "Boxplot",
                    _figure_block(
                        box_raw,
                        (
                            "Each box summarizes the pooled empirical spread for the selected group (median, quartiles, and whiskers). "
                            f"Jittered dots show one robust value per subject for the selected {variant_label.lower()} summary "
                            "(median over available runs). "
                            f"Units: {value_label}."
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
                            "Density-smoothed violin variant of the same values. "
                            f"Jittered dots show one robust value per subject for the selected {variant_label.lower()} summary. "
                            f"Units: {value_label}."
                        ),
                        normalized_variant=True,
                        norm_mode="y",
                    ),
                ),
                (
                    "Histogram",
                    _figure_block(
                        hist,
                        f"Histogram variant of the same summary values. Units: {value_label}.",
                        normalized_variant=True,
                        norm_mode="x",
                    ),
                ),
                (
                    "Density",
                    _figure_block(
                        dens,
                        f"Kernel-density variant of the same summary values. Units: {value_label}.",
                        normalized_variant=True,
                        norm_mode="x",
                    ),
                ),
            ],
            level=4,
        )

    def _ordered_variant_specs(
        default_label: str,
        values_by_label: Dict[str, Dict[str, List[float]]],
        point_col_by_label: Dict[str, str],
    ) -> List[VariantSpec]:
        ordered = [default_label] + [label for label in ("Median", "Mean", "Upper tail") if label != default_label]
        specs: List[VariantSpec] = []
        for label in ordered:
            specs.append((label, values_by_label.get(label, {}), point_col_by_label.get(label, "")))
        return specs

    def _summary_variant_tabs(
        group_id: str,
        variants: List[VariantSpec],
        title_prefix: str,
        value_label: str,
    ) -> str:
        items: List[Tuple[str, str]] = []
        for idx, (variant_label, values_map, variant_point_col) in enumerate(variants):
            items.append(
                (
                    variant_label,
                    _variant_tabs(
                        f"{group_id}-variant-{idx}",
                        values_map,
                        point_col=variant_point_col,
                        variant_label=variant_label,
                        title_prefix=f"{title_prefix} [{variant_label}]",
                        value_label=value_label,
                    ),
                )
            )
        return _build_subtabs_html(group_id, items, level=4)

    def _metric_panel(
        metric_name: str,
        rec_variants: List[VariantSpec],
        ch_variants: List[VariantSpec],
        epoch_variants: List[VariantSpec],
        unit_label: str,
        formula_a: str,
        formula_b: str,
        formula_c: str,
        heatmap_matrices_by_condition: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        heatmap_epoch_notes: Optional[Dict[str, str]] = None,
        topomap_payloads: Optional[Dict[str, TopomapPayload]] = None,
        fingerprint_spec: Optional[Tuple[str, str, str, str]] = None,
    ) -> str:
        rec_default = rec_variants[0][1] if rec_variants else {}
        ch_default = ch_variants[0][1] if ch_variants else {}
        epoch_default = epoch_variants[0][1] if epoch_variants else {}
        point_col_default = rec_variants[0][2] if rec_variants else ""
        counts_html = _panel_counts_html(point_col_default, rec_default, ch_default, epoch_default)

        dist_tabs = _build_subtabs_html(
            f"dist-{metric_name.lower()}-{tab_token}",
            [
                (
                    "A: Recording distributions",
                    _panel_subtitle("A", formula_a, counts_html)
                    + _summary_variant_tabs(
                        f"var-{metric_name.lower()}-a-{tab_token}",
                        rec_variants,
                        title_prefix=f"{metric_name} recording distribution{suffix}",
                        value_label=unit_label,
                    ),
                ),
                (
                    "B: Epochs per channel",
                    _panel_subtitle("B", formula_b, counts_html)
                    + _summary_variant_tabs(
                        f"var-{metric_name.lower()}-b-{tab_token}",
                        ch_variants,
                        title_prefix=f"{metric_name} epochs-per-channel distribution{suffix}",
                        value_label=unit_label,
                    ),
                ),
                (
                    "C: Channels per epoch",
                    _panel_subtitle("C", formula_c, counts_html)
                    + _summary_variant_tabs(
                        f"var-{metric_name.lower()}-c-{tab_token}",
                        epoch_variants,
                        title_prefix=f"{metric_name} channels-per-epoch distribution{suffix}",
                        value_label=unit_label,
                    ),
                ),
            ],
            level=3,
        )

        blocks = [f"<div class='metric-block'><h3>{metric_name}</h3>{dist_tabs}</div>"]

        if heatmap_matrices_by_condition:
            heatmap_figures = {
                cond: plot_heatmap_sorted_channels_windows(
                    variants,
                    title=f"{metric_name} channel-epoch map ({cond}){suffix}",
                    color_title=unit_label,
                    summary_mode=("upper_tail" if metric_name.lower().startswith("ptp") else "median"),
                    channel_names=(
                        topomap_payloads[cond].layout.names
                        if (topomap_payloads is not None and cond in topomap_payloads)
                        else None
                    ),
                )
                for cond, variants in sorted(heatmap_matrices_by_condition.items())
            }
            blocks.append(
                "<h4>Channel x epoch maps</h4>"
                + _condition_figure_blocks(
                    heatmap_figures,
                    (
                        "Heatmap cell is one channel-by-epoch value. "
                        "Bottom buttons switch heatmap variant (median/mean/upper tail) and "
                        "independently switch top/right profile central summaries."
                    ),
                    interpretation_by_condition=heatmap_epoch_notes,
                    normalized_variant=True,
                    norm_mode="z",
                )
            )

        if topomap_payloads is not None:
            blocks.append(
                _topomap_blocks(
                    payloads_by_condition=topomap_payloads,
                    title_prefix=f"{metric_name} topographic footprint{suffix}",
                    color_title=unit_label,
                    interpretation=(
                        "Each point is a channel-level summary value. "
                        "For Elekta systems, overlapping triplets are visually spread so 1 magnetometer and 2 gradiometer values can be inspected."
                    ),
                    normalized_variant=True,
                )
            )

        if fingerprint_spec is not None:
            x_col, y_col, x_label, y_label = fingerprint_spec
            fingerprint = plot_run_fingerprint_scatter(
                df,
                x_col=x_col,
                y_col=y_col,
                title=f"Recording fingerprint ({metric_name}){suffix}",
                x_label=x_label,
                y_label=y_label,
            )
            blocks.append(
                _figure_block(
                    fingerprint,
                    (
                        "Each point is one recording (subject x task x run). "
                        "X and Y summarize central and upper-tail dimensions for this metric to expose recording-level spread."
                    ),
                )
            )

        return "".join(blocks)

    std_heatmap_by_condition = _heatmap_variants_by_condition_from_acc(
        acc.std_heatmap_runs_by_condition,
        acc.std_heatmap_by_condition,
        acc.std_heatmap_sum_by_condition,
        acc.std_heatmap_count_by_condition,
        acc.std_heatmap_upper_by_condition,
    )
    std_heatmap_epoch_notes = _epoch_consistency_notes(acc.std_heatmap_runs_by_condition)
    std_panel = _metric_panel(
        metric_name="STD",
        rec_variants=_ordered_variant_specs(
            "Median",
            {
                "Median": _run_values_by_condition(df, "std_median"),
                "Mean": _run_values_by_condition(df, "std_mean"),
                "Upper tail": _run_values_by_condition(df, "std_upper_tail"),
            },
            {"Median": "std_median", "Mean": "std_mean", "Upper tail": "std_upper_tail"},
        ),
        ch_variants=_ordered_variant_specs(
            "Median",
            {
                "Median": acc.std_dist_by_condition,
                "Mean": acc.std_dist_mean_by_condition,
                "Upper tail": acc.std_dist_upper_by_condition,
            },
            {"Median": "std_median", "Mean": "std_mean", "Upper tail": "std_upper_tail"},
        ),
        epoch_variants=_ordered_variant_specs(
            "Median",
            {
                "Median": _epoch_values_from_profiles(acc.std_window_profiles_by_condition, field="q50"),
                "Mean": _epoch_values_from_profiles(acc.std_window_profiles_by_condition, field="mean"),
                "Upper tail": _epoch_values_from_profiles(acc.std_window_profiles_by_condition, field="q95"),
            },
            {"Median": "std_median", "Mean": "std_mean", "Upper tail": "std_upper_tail"},
        ),
        unit_label=f"STD ({amplitude_unit})",
        formula_a="Median: median_c(median_t STD[c,t]); Mean: mean_c(mean_t STD[c,t]); Upper tail: q95_c(q95_t STD[c,t]).",
        formula_b="Median: median_t STD[c,t] per channel c; Mean: mean_t STD[c,t] per channel c; Upper tail: q95_t STD[c,t] per channel c.",
        formula_c="Median: median_c STD[c,t] per epoch t; Mean: mean_c STD[c,t] per epoch t; Upper tail: q95_c STD[c,t] per epoch t.",
        heatmap_matrices_by_condition=std_heatmap_by_condition,
        heatmap_epoch_notes=std_heatmap_epoch_notes,
        topomap_payloads=acc.std_topomap_by_condition,
        fingerprint_spec=(
            "std_median",
            "std_upper_tail",
            f"Median channel STD per recording ({amplitude_unit})",
            f"Upper-tail channel STD per recording ({amplitude_unit})",
        ),
    )

    ptp_heatmap_by_condition = _heatmap_variants_by_condition_from_acc(
        acc.ptp_heatmap_runs_by_condition,
        acc.ptp_heatmap_by_condition,
        acc.ptp_heatmap_sum_by_condition,
        acc.ptp_heatmap_count_by_condition,
        acc.ptp_heatmap_upper_by_condition,
    )
    ptp_heatmap_epoch_notes = _epoch_consistency_notes(acc.ptp_heatmap_runs_by_condition)
    ptp_panel = _metric_panel(
        metric_name="PtP",
        rec_variants=_ordered_variant_specs(
            "Upper tail",
            {
                "Median": _run_values_by_condition(df, "ptp_median"),
                "Mean": _run_values_by_condition(df, "ptp_mean"),
                "Upper tail": _run_values_by_condition(df, "ptp_upper_tail"),
            },
            {"Median": "ptp_median", "Mean": "ptp_mean", "Upper tail": "ptp_upper_tail"},
        ),
        ch_variants=_ordered_variant_specs(
            "Upper tail",
            {
                "Median": acc.ptp_dist_by_condition,
                "Mean": acc.ptp_dist_mean_by_condition,
                "Upper tail": acc.ptp_dist_upper_by_condition,
            },
            {"Median": "ptp_median", "Mean": "ptp_mean", "Upper tail": "ptp_upper_tail"},
        ),
        epoch_variants=_ordered_variant_specs(
            "Upper tail",
            {
                "Median": _epoch_values_from_profiles(acc.ptp_window_profiles_by_condition, field="q50"),
                "Mean": _epoch_values_from_profiles(acc.ptp_window_profiles_by_condition, field="mean"),
                "Upper tail": _epoch_values_from_profiles(acc.ptp_window_profiles_by_condition, field="q95"),
            },
            {"Median": "ptp_median", "Mean": "ptp_mean", "Upper tail": "ptp_upper_tail"},
        ),
        unit_label=f"PtP ({amplitude_unit})",
        formula_a="Upper tail (default): q95_c(q99_t PtP[c,t]); Mean: mean_c(mean_t PtP[c,t]); Median: median_c(q95_t PtP[c,t]).",
        formula_b="Upper tail (default): q99_t PtP[c,t] per channel c; Mean: mean_t PtP[c,t] per channel c; Median: stored central channel summary per c.",
        formula_c="Upper tail (default): q95_c PtP[c,t] per epoch t; Mean: mean_c PtP[c,t] per epoch t; Median: median_c PtP[c,t] per epoch t.",
        heatmap_matrices_by_condition=ptp_heatmap_by_condition,
        heatmap_epoch_notes=ptp_heatmap_epoch_notes,
        topomap_payloads=acc.ptp_topomap_by_condition,
        fingerprint_spec=(
            "ptp_median",
            "ptp_upper_tail",
            f"Median channel PtP per recording ({amplitude_unit})",
            f"Upper-tail channel PtP per recording ({amplitude_unit})",
        ),
    )

    psd_panel = _metric_panel(
        metric_name="PSD mains ratio",
        rec_variants=_ordered_variant_specs(
            "Mean",
            {
                "Median": _run_values_by_condition(df, "mains_ratio"),
                "Mean": _run_values_by_condition(df, "mains_ratio"),
                "Upper tail": _run_values_by_condition(df, "mains_harmonics_ratio"),
            },
            {"Median": "mains_ratio", "Mean": "mains_ratio", "Upper tail": "mains_harmonics_ratio"},
        ),
        ch_variants=_ordered_variant_specs(
            "Mean",
            {
                "Median": acc.psd_ratio_by_condition,
                "Mean": acc.psd_ratio_by_condition,
                "Upper tail": acc.psd_harmonics_ratio_by_condition,
            },
            {"Median": "mains_ratio", "Mean": "mains_ratio", "Upper tail": "mains_harmonics_ratio"},
        ),
        epoch_variants=_ordered_variant_specs("Mean", {"Median": {}, "Mean": {}, "Upper tail": {}}, {"Median": "mains_ratio", "Mean": "mains_ratio", "Upper tail": "mains_harmonics_ratio"}),
        unit_label="Relative power (unitless)",
        formula_a="Mean/Median: mean_c mains_ratio[c]; Upper tail: mean_c harmonics_ratio[c].",
        formula_b="Mean/Median: mains_ratio[c] per channel c; Upper tail: harmonics_ratio[c] per channel c.",
        formula_c="Not available: epoch-wise PSD summaries are not stored in run derivatives.",
        topomap_payloads=acc.psd_topomap_by_condition,
        fingerprint_spec=(
            "mains_ratio",
            "mains_harmonics_ratio",
            "Mains relative power per recording",
            "Harmonics relative power per recording",
        ),
    ) + _condition_figure_blocks(
        {
            cond: plot_psd_median_band(_aggregate_psd_profiles(profiles), title=f"PSD profile ({cond})")
            for cond, profiles in sorted(acc.psd_profiles_by_condition.items())
        },
        "Line and quantile bands summarize channel PSD profiles by task (frequency in Hz).",
        normalized_variant=True,
        norm_mode="y",
    )

    ecg_panel = _metric_panel(
        metric_name="ECG correlation",
        rec_variants=_ordered_variant_specs(
            "Mean",
            {
                "Median": _run_values_by_condition(df, "ecg_mean_abs_corr"),
                "Mean": _run_values_by_condition(df, "ecg_mean_abs_corr"),
                "Upper tail": _run_values_by_condition(df, "ecg_p95_abs_corr"),
            },
            {"Median": "ecg_mean_abs_corr", "Mean": "ecg_mean_abs_corr", "Upper tail": "ecg_p95_abs_corr"},
        ),
        ch_variants=_ordered_variant_specs(
            "Mean",
            {
                "Median": acc.ecg_corr_by_condition,
                "Mean": acc.ecg_corr_by_condition,
                "Upper tail": acc.ecg_corr_by_condition,
            },
            {"Median": "ecg_mean_abs_corr", "Mean": "ecg_mean_abs_corr", "Upper tail": "ecg_p95_abs_corr"},
        ),
        epoch_variants=_ordered_variant_specs("Mean", {"Median": {}, "Mean": {}, "Upper tail": {}}, {"Median": "ecg_mean_abs_corr", "Mean": "ecg_mean_abs_corr", "Upper tail": "ecg_p95_abs_corr"}),
        unit_label="|r| (unitless)",
        formula_a="Mean/Median: mean_c |r[c]|; Upper tail: q95_c |r[c]|.",
        formula_b="Channel values are |r[c]|. Median and mean variants coincide with stored channel-level magnitudes.",
        formula_c="Not available: epoch-wise ECG channel summaries are not stored in run derivatives.",
        topomap_payloads=acc.ecg_topomap_by_condition,
        fingerprint_spec=(
            "ecg_mean_abs_corr",
            "ecg_p95_abs_corr",
            "Mean |ECG correlation| per recording",
            "Upper-tail |ECG correlation| per recording",
        ),
    )

    eog_panel = _metric_panel(
        metric_name="EOG correlation",
        rec_variants=_ordered_variant_specs(
            "Mean",
            {
                "Median": _run_values_by_condition(df, "eog_mean_abs_corr"),
                "Mean": _run_values_by_condition(df, "eog_mean_abs_corr"),
                "Upper tail": _run_values_by_condition(df, "eog_p95_abs_corr"),
            },
            {"Median": "eog_mean_abs_corr", "Mean": "eog_mean_abs_corr", "Upper tail": "eog_p95_abs_corr"},
        ),
        ch_variants=_ordered_variant_specs(
            "Mean",
            {
                "Median": acc.eog_corr_by_condition,
                "Mean": acc.eog_corr_by_condition,
                "Upper tail": acc.eog_corr_by_condition,
            },
            {"Median": "eog_mean_abs_corr", "Mean": "eog_mean_abs_corr", "Upper tail": "eog_p95_abs_corr"},
        ),
        epoch_variants=_ordered_variant_specs("Mean", {"Median": {}, "Mean": {}, "Upper tail": {}}, {"Median": "eog_mean_abs_corr", "Mean": "eog_mean_abs_corr", "Upper tail": "eog_p95_abs_corr"}),
        unit_label="|r| (unitless)",
        formula_a="Mean/Median: mean_c |r[c]|; Upper tail: q95_c |r[c]|.",
        formula_b="Channel values are |r[c]|. Median and mean variants coincide with stored channel-level magnitudes.",
        formula_c="Not available: epoch-wise EOG channel summaries are not stored in run derivatives.",
        topomap_payloads=acc.eog_topomap_by_condition,
        fingerprint_spec=(
            "eog_mean_abs_corr",
            "eog_p95_abs_corr",
            "Mean |EOG correlation| per recording",
            "Upper-tail |EOG correlation| per recording",
        ),
    )

    muscle_epoch_profiles = {
        cond: [{"q50": np.asarray(v, dtype=float), "mean": np.asarray(v, dtype=float), "q95": np.asarray(v, dtype=float)} for v in vals]
        for cond, vals in acc.muscle_profiles_by_condition.items()
    }
    muscle_panel = _metric_panel(
        metric_name="Muscle score",
        rec_variants=_ordered_variant_specs(
            "Median",
            {
                "Median": _run_values_by_condition(df, "muscle_median"),
                "Mean": _run_values_by_condition(df, "muscle_mean"),
                "Upper tail": _run_values_by_condition(df, "muscle_p95"),
            },
            {"Median": "muscle_median", "Mean": "muscle_mean", "Upper tail": "muscle_p95"},
        ),
        ch_variants=_ordered_variant_specs(
            "Median",
            {
                "Median": _run_values_by_condition(df, "muscle_median"),
                "Mean": _run_values_by_condition(df, "muscle_mean"),
                "Upper tail": _run_values_by_condition(df, "muscle_p95"),
            },
            {"Median": "muscle_median", "Mean": "muscle_mean", "Upper tail": "muscle_p95"},
        ),
        epoch_variants=_ordered_variant_specs(
            "Median",
            {
                "Median": _epoch_values_from_profiles(muscle_epoch_profiles, field="q50"),
                "Mean": _epoch_values_from_profiles(muscle_epoch_profiles, field="mean"),
                "Upper tail": _epoch_values_from_profiles(muscle_epoch_profiles, field="q95"),
            },
            {"Median": "muscle_median", "Mean": "muscle_mean", "Upper tail": "muscle_p95"},
        ),
        unit_label="Muscle score (z-score)",
        formula_a="Median: median_t score[t] per run; Mean: mean_t score[t] per run; Upper tail: q95_t score[t] per run.",
        formula_b="Stored run-level summaries are reused (single channel-agnostic muscle score sequence per run).",
        formula_c="Epoch values are score[t] per run; Median/Mean/Upper tail are computed across pooled epoch values.",
        fingerprint_spec=(
            "muscle_median",
            "muscle_p95",
            "Median muscle score per recording",
            "Upper-tail muscle score per recording",
        ),
    ) + _condition_figure_blocks(
        {
            cond: plot_quantile_band_timecourse(
                _aggregate_muscle_profiles(profiles),
                title=f"Muscle epoch envelope ({cond})",
                x_title="Epoch index",
                y_title="Muscle score (z-score)",
            )
            for cond, profiles in sorted(acc.muscle_profiles_by_condition.items())
        },
        "Muscle epoch envelope by task (median and quantile bands).",
        normalized_variant=True,
        norm_mode="y",
    )

    metric_tabs = _build_subtabs_html(
        f"metric-main-{tab_token}",
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
        "<p>Each metric has three distribution panels (A/B/C), summary variants (default metric summary first, then the two alternatives), and recording fingerprint scatter for recording-level spread. STD and PtP also include channel-epoch heatmap variants.</p>"
        + metric_tabs
        + "</section>"
    )


def _build_subject_drilldown_section(acc: ChTypeAccumulator, amplitude_unit: str) -> str:
    df = _run_rows_dataframe(acc.run_rows)
    if df.empty:
        return "<section><h2>Subject Drill-down</h2><p>No subject-aware details are available.</p></section>"

    std_scatter = plot_run_fingerprint_scatter(
        df,
        "std_median",
        "std_upper_tail",
        "Recording fingerprint (STD)",
        f"Median channel STD per recording ({amplitude_unit})",
        f"Upper-tail channel STD per recording ({amplitude_unit})",
    )
    ptp_scatter = plot_run_fingerprint_scatter(
        df,
        "ptp_median",
        "ptp_upper_tail",
        "Recording fingerprint (PtP)",
        f"Median channel PtP per recording ({amplitude_unit})",
        f"Upper-tail channel PtP per recording ({amplitude_unit})",
    )
    mains_scatter = plot_run_fingerprint_scatter(
        df,
        "mains_ratio",
        "mains_harmonics_ratio",
        "Recording fingerprint (PSD mains/harmonics)",
        "Mains relative power",
        "Harmonics relative power",
    )
    ecg_scatter = plot_run_fingerprint_scatter(
        df,
        "ecg_mean_abs_corr",
        "ecg_p95_abs_corr",
        "Recording fingerprint (ECG)",
        "Mean |ECG correlation| per recording",
        "p95 |ECG correlation| per recording",
    )
    eog_scatter = plot_run_fingerprint_scatter(
        df,
        "eog_mean_abs_corr",
        "eog_p95_abs_corr",
        "Recording fingerprint (EOG)",
        "Mean |EOG correlation| per recording",
        "p95 |EOG correlation| per recording",
    )
    muscle_scatter = plot_run_fingerprint_scatter(
        df,
        "muscle_median",
        "muscle_p95",
        "Recording fingerprint (Muscle)",
        "Median muscle score per recording",
        "p95 muscle score per recording",
    )
    metric_heatmap = plot_subject_metric_heatmap(df, "Subject-by-metric normalized summary matrix (raw values on hover)")
    std_profiles = plot_subject_epoch_small_multiples(
        acc.std_subject_profiles,
        _subject_scores_from_df(df, "std_upper_tail"),
        "Subject epoch profiles (STD, top 12 by upper-tail summary)",
        y_label=f"STD ({amplitude_unit})",
        top_n=12,
    )
    ptp_profiles = plot_subject_epoch_small_multiples(
        acc.ptp_subject_profiles,
        _subject_scores_from_df(df, "ptp_upper_tail"),
        "Subject epoch profiles (PtP, top 12 by upper-tail summary)",
        y_label=f"PtP ({amplitude_unit})",
        top_n=12,
    )

    return (
        "<section>"
        "<h2>Subject Drill-down</h2>"
        "<p>These plots isolate recording and subject fingerprints after cohort-level triage.</p>"
        + _figure_block(
            std_scatter,
            "Each point is one recording. X is central summary and Y is upper-tail summary for STD.",
        )
        + _figure_block(
            ptp_scatter,
            "Each point is one recording. X is central summary and Y is upper-tail summary for PtP.",
        )
        + _figure_block(
            mains_scatter,
            "Each point is one recording. Axes summarize mains and harmonics relative power.",
        )
        + _figure_block(
            ecg_scatter,
            "Each point is one recording. Axes summarize central and upper-tail ECG correlation magnitude.",
        )
        + _figure_block(
            eog_scatter,
            "Each point is one recording. Axes summarize central and upper-tail EOG correlation magnitude.",
        )
        + _figure_block(
            muscle_scatter,
            "Each point is one recording. Axes summarize central and upper-tail muscle score burden.",
        )
        + _figure_block(
            metric_heatmap,
            "Rows are subjects; columns are metrics; color is normalized for readability and hover shows raw metric summaries.",
        )
        + _figure_block(
            std_profiles,
            "Each panel shows one subject. Line and bands summarize channel quantiles across epochs (STD).",
        )
        + _figure_block(
            ptp_profiles,
            "Each panel shows one subject. Line and bands summarize channel quantiles across epochs (PtP).",
        )
        + "</section>"
    )


def _build_statistical_appendix_section(acc: ChTypeAccumulator, amplitude_unit: str, is_combined: bool) -> str:
    suffix = " (all channels)" if is_combined else ""
    std_ecdf = _make_ecdf_figure(acc.std_dist_by_condition, f"STD ECDF{suffix}", f"STD ({amplitude_unit})")
    ptp_ecdf = _make_ecdf_figure(acc.ptp_dist_by_condition, f"PtP ECDF{suffix}", f"PtP ({amplitude_unit})")
    psd_ecdf = _make_ecdf_figure(acc.psd_ratio_by_condition, "Mains relative power ECDF", "Mains ratio")
    ecg_ecdf = _make_ecdf_figure(acc.ecg_corr_by_condition, "ECG |r| ECDF", "|r|")
    eog_ecdf = _make_ecdf_figure(acc.eog_corr_by_condition, "EOG |r| ECDF", "|r|")
    muscle_ecdf = _make_ecdf_figure(acc.muscle_scalar_by_condition, "Muscle score ECDF", "Muscle score")

    return (
        "<section>"
        "<h2>Cummulative distributions</h2>"
        "<p>Supplementary distribution views for readers who prefer cumulative representations.</p>"
        + _figure_block(std_ecdf, "Cumulative distribution of channel-level STD summaries.")
        + _figure_block(ptp_ecdf, "Cumulative distribution of channel-level PtP summaries.")
        + _figure_block(psd_ecdf, "Cumulative distribution of mains relative power summaries.")
        + _figure_block(ecg_ecdf, "Cumulative distribution of ECG correlation magnitude summaries.")
        + _figure_block(eog_ecdf, "Cumulative distribution of EOG correlation magnitude summaries.")
        + _figure_block(muscle_ecdf, "Cumulative distribution of recording-level muscle summaries.")
        + "</section>"
    )


def _build_tab_content(tab_name: str, acc: ChTypeAccumulator, is_combined: bool) -> str:
    tab_token = re.sub(r"[^a-z0-9]+", "-", tab_name.lower())
    if is_combined:
        amplitude_unit = "mixed pT-based MEG units (all channels)"
    elif tab_name.upper() == "MAG":
        amplitude_unit = "picoTesla (pT)"
    elif tab_name.upper() == "EEG":
        amplitude_unit = "microVolts (µV)"
    else:
        amplitude_unit = "picoTesla/m (pT/m)"

    combined_notice = ""
    if is_combined:
        df = _run_rows_dataframe(acc.run_rows)
        mag_rows = int(df.loc[df["channel_type"] == "mag", "run_key"].nunique()) if (not df.empty and "channel_type" in df.columns and "run_key" in df.columns) else 0
        grad_rows = int(df.loc[df["channel_type"] == "grad", "run_key"].nunique()) if (not df.empty and "channel_type" in df.columns and "run_key" in df.columns) else 0
        n_subjects = int(df["subject"].nunique()) if (not df.empty and "subject" in df.columns) else 0
        combined_notice = (
            "<section>"
            "<h2>Combined Channel-Type Context</h2>"
            f"<p><strong>Per-type run counts:</strong> MAG={mag_rows}, GRAD={grad_rows}; <strong>N subjects:</strong> {n_subjects}.</p>"
            "<p><strong>Unit warning:</strong> The combined tab is cumulative across channel types (MAG + GRAD). "
            "Amplitude metrics therefore mix pT and pT/m footprints. Use MAG and GRAD tabs for strict unit-specific interpretation.</p>"
            "</section>"
        )

    sections = [
        (
            "Summary distributions",
            _build_summary_distributions_section(
                acc,
                amplitude_unit=amplitude_unit,
                is_combined=is_combined,
                tab_token=tab_token,
            ),
        ),
        ("Cohort QA overview", _build_cohort_overview_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        ("QA metrics across tasks", _build_condition_effect_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        ("QA metrics details", _build_metric_details_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
        ("Cummulative distributions", _build_statistical_appendix_section(acc, amplitude_unit=amplitude_unit, is_combined=is_combined)),
    ]
    group_id = f"main-{re.sub(r'[^a-z0-9]+', '-', tab_name.lower())}"
    return combined_notice + _build_subtabs_html(group_id, sections, level=1)


def _build_report_html(
    dataset_name: str,
    tab_accumulators: Dict[str, ChTypeAccumulator],
    settings_snapshot: str,
) -> str:
    """Compose the self-contained HTML report (tabs + plots + client JS)."""
    _reset_lazy_figure_store()
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version = getattr(meg_qc, "__version__", "unknown")
    tab_order = ["Combined (mag+grad)", "MAG", "GRAD", "EEG"]
    available_tabs = [tab for tab in tab_order if tab in tab_accumulators and tab_accumulators[tab].run_count > 0]
    if not available_tabs:
        available_tabs = [tab for tab in tab_order if tab in tab_accumulators]

    tab_buttons = []
    tab_divs = []
    for idx, tab in enumerate(available_tabs):
        tab_id = f"tab-{idx}"
        active_class = " active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn{active_class}' data-target='{tab_id}'>{tab}</button>")
        content = _build_tab_content(tab, tab_accumulators[tab], is_combined=(tab == "Combined (mag+grad)"))
        tab_divs.append(f"<div id='{tab_id}' class='tab-content{active_class}'>{content}</div>")
    lazy_payload_scripts = _lazy_payload_script_tags_html()

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QA group report {dataset_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
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
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 12px;
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
    .tile {{
      border: 1px solid #d9e7f8;
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
    }}
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
  <div id="report-loading-overlay" class="loading-overlay">
    <div class="loading-card">
      <div class="loading-spinner"></div>
      <div class="loading-title">Loading QA report</div>
      <div class="loading-subtitle">Rendering visible figures...</div>
    </div>
  </div>
  <main>
    <section>
      <div class="report-header">
        <div>
          <h1>QA group report: {dataset_name}</h1>
          <p><strong>Generated:</strong> {generated}</p>
          <p><strong>MEGqc version:</strong> {version}</p>
          <p><strong>Epoch label:</strong> epochs</p>
        </div>
      </div>
      <h3>Settings snapshot</h3>
      <pre>{settings_snapshot}</pre>
      <p><strong>Important:</strong> Cohort QA overview combines global cohort footprints and subject-aware summaries; metric-level panels preserve recording identity in hover text.</p>
      <div class="report-tools">
        <button id="grid-toggle-btn" class="tool-btn active" type="button">Hide grids</button>
      </div>
      <div class="tab-row">
        {"".join(tab_buttons)}
      </div>
      {"".join(tab_divs)}
    </section>
  </main>
  {lazy_payload_scripts}
  <script>
    (function() {{
      // Top-level tab activation and responsive plot resizing.
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
          // Release parsed JSON text from DOM memory after first use.
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
        if (!plotEl.__summaryEverDisplaced) plotEl.__summaryEverDisplaced = false;
        const idxs = _summaryScatterIdx(plotEl);
        const xUpdate = [];
        const traceIdx = [];
        idxs.forEach((i) => {{
          if (!plotEl.__summaryBaseX.hasOwnProperty(i)) {{
            const rawX = plotEl.data[i] && plotEl.data[i].x;
            const base = (rawX != null) ? Array.from(rawX) : [];
            plotEl.__summaryBaseX[i] = base;
          }}
          const base = plotEl.__summaryBaseX[i] || [];
          xUpdate.push(base.map((v) => Number(v) + shift));
          traceIdx.push(i);
        }});
        if (traceIdx.length === 0) return;
        if (shift === 0.0 && !plotEl.__summaryEverDisplaced) return;
        plotEl.__summaryEverDisplaced = true;
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

      function applySummaryState(groupId, scopeRoot) {{
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
              applySummaryState(groupId, scope);
            }});
          }});
          applySummaryState(groupId, scope);
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
      // Per-figure raw/normalized toggles.
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


def _update_accumulator_for_loaded_run(
    acc_by_type: Dict[str, ChTypeAccumulator],
    record: RunRecord,
    loaded: LoadedRunData,
) -> None:
    """Update accumulators for a run using preloaded derivative arrays."""
    files = record.files
    condition_label = _condition_label(record.meta)

    std_data = loaded.std_data
    ptp_data = loaded.ptp_data
    psd_data = loaded.psd_data
    ecg_raw_data = loaded.ecg_raw_data
    eog_raw_data = loaded.eog_raw_data
    muscle_data = loaded.muscle_data
    ptp_desc = loaded.ptp_desc
    layout_cache = loaded.layout_cache

    def _layout_by_desc(desc: str) -> Dict[str, SensorLayout]:
        if desc not in files:
            return {}
        return layout_cache.get(desc, {})

    run_ch_types = set()
    run_ch_types.update(std_data.keys())
    run_ch_types.update(ptp_data.keys())
    run_ch_types.update(psd_data.keys())
    run_ch_types.update(ecg_raw_data.keys())
    run_ch_types.update(eog_raw_data.keys())
    run_ch_types.update(muscle_data.keys())

    for ch_type in run_ch_types:
        if ch_type not in CH_TYPES:
            continue

        # ── BIDS modality guard ───────────────────────────────────────────────
        # The BIDS suffix of the derivative file (stored in record.meta.bids_suffix)
        # is the authoritative signal for which modality this recording belongs to.
        # A MEG recording (_meg.tsv) may contain embedded EEG channels, but those
        # must NOT be counted as EEG-modality data.  Conversely, an EEG recording
        # (_eeg.tsv) should never produce mag/grad channel rows.
        _bids_sfx = record.meta.bids_suffix
        if _bids_sfx == "meg" and ch_type == "eeg":
            continue  # EEG channels inside a MEG recording → skip
        if _bids_sfx == "eeg" and ch_type in ("mag", "grad"):
            continue  # MEG channel types inside an EEG recording → skip

        acc = acc_by_type[ch_type]
        acc.run_count += 1
        if record.meta.subject != "n/a":
            acc.subjects.add(record.meta.subject)
        acc.runs_by_condition[condition_label] += 1

        module_seen = {module: False for module in MODULES}
        row = RunMetricRow(
            run_key=record.meta.run_key,
            subject=record.meta.subject,
            session=record.meta.session,
            task=record.meta.task,
            run=record.meta.run,
            condition=record.meta.condition,
            acquisition=record.meta.acquisition,
            recording=record.meta.recording,
            processing=record.meta.processing,
            channel_type=ch_type,
        )

        if ch_type in std_data:
            matrix = np.asarray(std_data[ch_type], dtype=float) * TESLA_TO_PICO
            # Channel scalar summaries are used for distribution views.
            ch_summary_default_all = np.nanmedian(matrix, axis=1)
            ch_summary_mean_all = np.nanmean(matrix, axis=1)
            ch_summary_upper_all = np.nanquantile(matrix, 0.95, axis=1)
            ch_summary_norm = _robust_normalize_array(ch_summary_default_all)
            ch_summary_default = _finite_array(ch_summary_default_all)
            ch_summary_mean = _finite_array(ch_summary_mean_all)
            ch_summary_upper = _finite_array(ch_summary_upper_all)
            if ch_summary_default.size:
                acc.std_dist_by_condition[condition_label].extend(ch_summary_default.tolist())
                std_central_score = float(np.nanmedian(ch_summary_default))
                std_upper_score = float(np.nanquantile(ch_summary_upper if ch_summary_upper.size else ch_summary_default, 0.95))
                _update_representative_matrix(
                    acc.std_heatmap_by_condition,
                    acc.std_heatmap_score_by_condition,
                    acc.std_heatmap_score_history,
                    condition_label,
                    matrix,
                    std_central_score,
                )
                _update_upper_tail_matrix(
                    acc.std_heatmap_upper_by_condition,
                    acc.std_heatmap_upper_score_by_condition,
                    condition_label,
                    matrix,
                    std_upper_score,
                )
                row.std_median = _as_float(std_central_score)
                row.std_upper_tail = _as_float(std_upper_score)
            if ch_summary_mean.size:
                acc.std_dist_mean_by_condition[condition_label].extend(ch_summary_mean.tolist())
                row.std_mean = _as_float(np.nanmean(ch_summary_mean))
            if ch_summary_upper.size:
                acc.std_dist_upper_by_condition[condition_label].extend(ch_summary_upper.tolist())
            ch_summary_norm_finite = _finite_array(ch_summary_norm)
            if ch_summary_norm_finite.size:
                row.std_median_norm = _as_float(np.nanmedian(ch_summary_norm_finite))
                row.std_upper_tail_norm = _as_float(np.nanquantile(ch_summary_norm_finite, 0.95))
            std_profile = _profile_quantiles(matrix)
            # Profile quantiles preserve epoch structure with channel collapse.
            acc.std_window_profiles.append(std_profile)
            acc.std_window_profiles_by_condition[condition_label].append(std_profile)
            _accumulate_matrix_mean(
                acc.std_heatmap_sum_by_condition,
                acc.std_heatmap_count_by_condition,
                condition_label,
                matrix,
            )
            acc.std_heatmap_runs_by_condition[condition_label].append(np.asarray(matrix, dtype=float))
            acc.std_epoch_counts_by_condition[condition_label].append(int(matrix.shape[1]))
            if record.meta.subject != "n/a":
                acc.std_subject_profiles[record.meta.subject].append(std_profile)
            _update_topomap_payload_mean(
                acc.std_topomap_by_condition,
                acc.std_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("STDs").get(ch_type),
                ch_summary_default_all,
            )
            acc.source_paths.add(str(files["STDs"]))
            module_seen["STD"] = True

        if ch_type in ptp_data:
            matrix = np.asarray(ptp_data[ch_type], dtype=float) * TESLA_TO_PICO
            ch_summary_default_all = np.nanquantile(matrix, 0.95, axis=1)
            ch_summary_mean_all = np.nanmean(matrix, axis=1)
            ch_summary_upper_all = np.nanquantile(matrix, 0.99, axis=1)
            ch_summary_norm = _robust_normalize_array(ch_summary_default_all)
            ch_summary_default = _finite_array(ch_summary_default_all)
            ch_summary_mean = _finite_array(ch_summary_mean_all)
            ch_summary_upper = _finite_array(ch_summary_upper_all)
            if ch_summary_default.size:
                acc.ptp_dist_by_condition[condition_label].extend(ch_summary_default.tolist())
                ptp_central_score = float(np.nanmedian(ch_summary_default))
                ptp_upper_score = float(np.nanquantile(ch_summary_upper if ch_summary_upper.size else ch_summary_default, 0.95))
                _update_representative_matrix(
                    acc.ptp_heatmap_by_condition,
                    acc.ptp_heatmap_score_by_condition,
                    acc.ptp_heatmap_score_history,
                    condition_label,
                    matrix,
                    ptp_central_score,
                )
                _update_upper_tail_matrix(
                    acc.ptp_heatmap_upper_by_condition,
                    acc.ptp_heatmap_upper_score_by_condition,
                    condition_label,
                    matrix,
                    ptp_upper_score,
                )
                row.ptp_median = _as_float(ptp_central_score)
                row.ptp_upper_tail = _as_float(ptp_upper_score)
            if ch_summary_mean.size:
                acc.ptp_dist_mean_by_condition[condition_label].extend(ch_summary_mean.tolist())
                row.ptp_mean = _as_float(np.nanmean(ch_summary_mean))
            if ch_summary_upper.size:
                acc.ptp_dist_upper_by_condition[condition_label].extend(ch_summary_upper.tolist())
            ch_summary_norm_finite = _finite_array(ch_summary_norm)
            if ch_summary_norm_finite.size:
                row.ptp_median_norm = _as_float(np.nanmedian(ch_summary_norm_finite))
                row.ptp_upper_tail_norm = _as_float(np.nanquantile(ch_summary_norm_finite, 0.95))
            ptp_profile = _profile_quantiles(matrix)
            acc.ptp_window_profiles.append(ptp_profile)
            acc.ptp_window_profiles_by_condition[condition_label].append(ptp_profile)
            _accumulate_matrix_mean(
                acc.ptp_heatmap_sum_by_condition,
                acc.ptp_heatmap_count_by_condition,
                condition_label,
                matrix,
            )
            acc.ptp_heatmap_runs_by_condition[condition_label].append(np.asarray(matrix, dtype=float))
            acc.ptp_epoch_counts_by_condition[condition_label].append(int(matrix.shape[1]))
            if record.meta.subject != "n/a":
                acc.ptp_subject_profiles[record.meta.subject].append(ptp_profile)
            if ptp_desc is not None:
                _update_topomap_payload_mean(
                    acc.ptp_topomap_by_condition,
                    acc.ptp_topomap_count_by_condition,
                    condition_label,
                    _layout_by_desc(ptp_desc).get(ch_type),
                    ch_summary_default_all,
                )
            if "PtPsManual" in files:
                acc.source_paths.add(str(files["PtPsManual"]))
            elif "PtPsAuto" in files:
                acc.source_paths.add(str(files["PtPsAuto"]))
            module_seen["PTP"] = True

        if ch_type in psd_data:
            freqs, matrix = psd_data[ch_type]
            ratios_all, harmonics_all = _compute_mains_and_harmonics_ratio(matrix, freqs)
            ratios = _finite_array(ratios_all)
            if ratios.size:
                acc.psd_ratio_by_condition[condition_label].extend(ratios.tolist())
                row.mains_ratio = _as_float(np.nanmean(ratios))
            harmonics = _finite_array(harmonics_all)
            if harmonics.size:
                acc.psd_harmonics_ratio_by_condition[condition_label].extend(harmonics.tolist())
                row.mains_harmonics_ratio = _as_float(np.nanmean(harmonics))
            quant = _profile_quantiles(matrix)
            quant["freqs"] = freqs
            acc.psd_profiles.append(quant)
            acc.psd_profiles_by_condition[condition_label].append(quant)
            _update_topomap_payload_mean(
                acc.psd_topomap_by_condition,
                acc.psd_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("PSDs").get(ch_type),
                ratios_all,
            )
            acc.source_paths.add(str(files["PSDs"]))
            module_seen["PSD"] = True

        if ch_type in ecg_raw_data:
            vals_all = np.asarray(ecg_raw_data[ch_type], dtype=float)
            vals = _finite_array(vals_all)
            if vals.size:
                acc.ecg_corr_by_condition[condition_label].extend(vals.tolist())
                row.ecg_mean_abs_corr = _as_float(np.nanmean(vals))
                row.ecg_p95_abs_corr = _as_float(np.nanquantile(vals, 0.95))
            _update_topomap_payload_mean(
                acc.ecg_topomap_by_condition,
                acc.ecg_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("ECGs").get(ch_type),
                vals_all,
            )
            acc.source_paths.add(str(files["ECGs"]))
            module_seen["ECG"] = True

        if ch_type in eog_raw_data:
            vals_all = np.asarray(eog_raw_data[ch_type], dtype=float)
            vals = _finite_array(vals_all)
            if vals.size:
                acc.eog_corr_by_condition[condition_label].extend(vals.tolist())
                row.eog_mean_abs_corr = _as_float(np.nanmean(vals))
                row.eog_p95_abs_corr = _as_float(np.nanquantile(vals, 0.95))
            _update_topomap_payload_mean(
                acc.eog_topomap_by_condition,
                acc.eog_topomap_count_by_condition,
                condition_label,
                _layout_by_desc("EOGs").get(ch_type),
                vals_all,
            )
            acc.source_paths.add(str(files["EOGs"]))
            module_seen["EOG"] = True

        if ch_type in muscle_data:
            scores = _finite_array(muscle_data[ch_type])
            if scores.size:
                acc.muscle_profiles.append(scores)
                acc.muscle_profiles_by_condition[condition_label].append(scores)
                acc.muscle_mean_by_condition[condition_label].append(float(np.nanmean(scores)))
                acc.muscle_scalar_by_condition[condition_label].append(
                    float(np.nanquantile(scores, 0.95))
                )
                row.muscle_mean = _as_float(np.nanmean(scores))
                row.muscle_median = _as_float(np.nanmedian(scores))
                row.muscle_p95 = _as_float(np.nanquantile(scores, 0.95))
            acc.source_paths.add(str(files["Muscle"]))
            module_seen["Muscle"] = True

        acc.run_rows.append(row)

        for module in MODULES:
            if module_seen[module]:
                acc.module_present[module] += 1
            else:
                acc.module_missing[module] += 1


def _load_run_data(record: RunRecord) -> LoadedRunData:
    """Load per-run machine-readable arrays once for optional parallel prefetch."""
    files = record.files

    std_data: Dict[str, np.ndarray] = {}
    ptp_data: Dict[str, np.ndarray] = {}
    psd_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    ecg_raw_data: Dict[str, np.ndarray] = {}
    eog_raw_data: Dict[str, np.ndarray] = {}
    muscle_data: Dict[str, np.ndarray] = {}
    ptp_desc: Optional[str] = None

    if "STDs" in files:
        std_data = _load_std_or_ptp_matrix(files["STDs"], "STD epoch_", "STD all")
    if "PtPsManual" in files:
        ptp_desc = "PtPsManual"
        ptp_data = _load_std_or_ptp_matrix(files["PtPsManual"], "PtP epoch_", "PtP all")
    elif "PtPsAuto" in files:
        ptp_desc = "PtPsAuto"
        ptp_data = _load_std_or_ptp_matrix(files["PtPsAuto"], "PtP epoch_", "PtP all")
    if "PSDs" in files:
        psd_data = _load_psd_matrix(files["PSDs"])
    if "ECGs" in files:
        ecg_raw_data = _load_correlation_values(files["ECGs"], "ecg")
    if "EOGs" in files:
        eog_raw_data = _load_correlation_values(files["EOGs"], "eog")
    if "Muscle" in files:
        muscle_data = _load_muscle_scores(files["Muscle"])

    layout_cache: Dict[str, Dict[str, SensorLayout]] = {}
    for desc in ("STDs", "PSDs", "ECGs", "EOGs"):
        if desc in files:
            layout_cache[desc] = _load_sensor_layout(files[desc])
    if ptp_desc is not None and ptp_desc in files:
        layout_cache[ptp_desc] = _load_sensor_layout(files[ptp_desc])

    return LoadedRunData(
        std_data=std_data,
        ptp_data=ptp_data,
        psd_data=psd_data,
        ecg_raw_data=ecg_raw_data,
        eog_raw_data=eog_raw_data,
        muscle_data=muscle_data,
        ptp_desc=ptp_desc,
        layout_cache=layout_cache,
    )


def _update_accumulator_for_run(acc_by_type: Dict[str, ChTypeAccumulator], record: RunRecord) -> None:
    """Sequential update entrypoint (single-run read + aggregation)."""
    loaded = _load_run_data(record)
    _update_accumulator_for_loaded_run(acc_by_type, record, loaded)


def _normalize_n_jobs(n_jobs: int) -> int:
    try:
        n_jobs = int(n_jobs)
    except Exception as exc:
        raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}") from exc
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use 1 for sequential or -1 for all cores.")
    return n_jobs


def _merge_list_map(dst: Dict[str, List], src: Dict[str, List]) -> None:
    for key, vals in src.items():
        if vals:
            dst[key].extend(vals)


def _merge_sum_count_dict(
    dst_sum: Dict[str, np.ndarray],
    dst_count: Dict[str, np.ndarray],
    src_sum: Dict[str, np.ndarray],
    src_count: Dict[str, np.ndarray],
) -> None:
    for cond, src_sum_matrix in src_sum.items():
        s = np.asarray(src_sum_matrix, dtype=float)
        if s.ndim != 2 or s.size == 0:
            continue
        c = np.asarray(src_count.get(cond, np.zeros_like(s, dtype=float)), dtype=float)
        if c.shape != s.shape:
            c = np.zeros_like(s, dtype=float)

        if cond not in dst_sum:
            dst_sum[cond] = s.copy()
            dst_count[cond] = c.copy()
            continue

        prev_s = np.asarray(dst_sum[cond], dtype=float)
        prev_c = np.asarray(dst_count.get(cond, np.zeros_like(prev_s, dtype=float)), dtype=float)
        n_rows = max(prev_s.shape[0], s.shape[0])
        n_cols = max(prev_s.shape[1], s.shape[1])

        def _pad(a: np.ndarray) -> np.ndarray:
            if a.shape == (n_rows, n_cols):
                return a
            return np.pad(
                a,
                ((0, n_rows - a.shape[0]), (0, n_cols - a.shape[1])),
                mode="constant",
                constant_values=0.0,
            )

        dst_sum[cond] = _pad(prev_s) + _pad(s)
        dst_count[cond] = _pad(prev_c) + _pad(c)


def _merge_topomap_mean_dict(
    dst_payloads: Dict[str, TopomapPayload],
    dst_counts: Dict[str, np.ndarray],
    src_payloads: Dict[str, TopomapPayload],
    src_counts: Dict[str, np.ndarray],
) -> None:
    for cond, src_payload in src_payloads.items():
        src_vals = np.asarray(src_payload.values, dtype=float).reshape(-1)
        src_cnt = np.asarray(
            src_counts.get(cond, np.isfinite(src_vals).astype(float)),
            dtype=float,
        ).reshape(-1)
        if src_vals.size == 0:
            continue

        if cond not in dst_payloads:
            dst_payloads[cond] = TopomapPayload(
                layout=SensorLayout(
                    x=np.asarray(src_payload.layout.x, dtype=float).copy(),
                    y=np.asarray(src_payload.layout.y, dtype=float).copy(),
                    names=list(src_payload.layout.names),
                    z=(np.asarray(src_payload.layout.z, dtype=float).copy() if src_payload.layout.z is not None else None),
                ),
                values=src_vals.copy(),
            )
            dst_counts[cond] = src_cnt.copy()
            continue

        dst_payload = dst_payloads[cond]
        dst_vals = np.asarray(dst_payload.values, dtype=float).reshape(-1)
        dst_cnt = np.asarray(
            dst_counts.get(cond, np.isfinite(dst_vals).astype(float)),
            dtype=float,
        ).reshape(-1)

        m = min(
            dst_vals.size,
            src_vals.size,
            dst_cnt.size,
            src_cnt.size,
            len(dst_payload.layout.names),
            dst_payload.layout.x.size,
            dst_payload.layout.y.size,
            (dst_payload.layout.z.size if dst_payload.layout.z is not None else dst_payload.layout.x.size),
            (src_payload.layout.z.size if src_payload.layout.z is not None else src_payload.layout.x.size),
        )
        if m < 1:
            continue

        a_vals = dst_vals[:m]
        b_vals = src_vals[:m]
        a_cnt = dst_cnt[:m]
        b_cnt = src_cnt[:m]
        num = np.where(np.isfinite(a_vals), a_vals, 0.0) * a_cnt + np.where(np.isfinite(b_vals), b_vals, 0.0) * b_cnt
        den = a_cnt + b_cnt
        out = np.full(m, np.nan, dtype=float)
        np.divide(num, np.maximum(den, np.finfo(float).eps), out=out, where=den > 0)

        dst_payloads[cond] = TopomapPayload(
            layout=SensorLayout(
                x=np.asarray(dst_payload.layout.x[:m], dtype=float),
                y=np.asarray(dst_payload.layout.y[:m], dtype=float),
                names=list(dst_payload.layout.names[:m]),
                z=(
                    np.asarray(dst_payload.layout.z[:m], dtype=float)
                    if dst_payload.layout.z is not None
                    else None
                ),
            ),
            values=out,
        )
        dst_counts[cond] = den


def _merge_representative_state(
    dst_matrix: Dict[str, np.ndarray],
    dst_score: Dict[str, float],
    dst_history: Dict[str, List[float]],
    src_matrix: Dict[str, np.ndarray],
    src_score: Dict[str, float],
    src_history: Dict[str, List[float]],
) -> None:
    for cond, hist in src_history.items():
        finite_hist = _finite_array(hist)
        if finite_hist.size:
            dst_history[cond].extend(finite_hist.tolist())

    for cond, matrix in src_matrix.items():
        s = float(src_score.get(cond, np.nan))
        if cond not in dst_matrix:
            dst_matrix[cond] = np.asarray(matrix, dtype=float)
            dst_score[cond] = s
            continue

        history = _finite_array(dst_history.get(cond, []))
        if history.size == 0:
            prev = float(dst_score.get(cond, np.nan))
            if (not np.isfinite(prev)) and np.isfinite(s):
                dst_matrix[cond] = np.asarray(matrix, dtype=float)
                dst_score[cond] = s
            continue

        target = float(np.nanmedian(history))
        prev = float(dst_score.get(cond, np.nan))
        if np.isfinite(s) and ((not np.isfinite(prev)) or abs(s - target) <= abs(prev - target)):
            dst_matrix[cond] = np.asarray(matrix, dtype=float)
            dst_score[cond] = s


def _merge_upper_tail_state(
    dst_matrix: Dict[str, np.ndarray],
    dst_score: Dict[str, float],
    src_matrix: Dict[str, np.ndarray],
    src_score: Dict[str, float],
) -> None:
    for cond, matrix in src_matrix.items():
        s = float(src_score.get(cond, np.nan))
        prev = float(dst_score.get(cond, float("-inf")))
        if cond not in dst_matrix or (np.isfinite(s) and ((not np.isfinite(prev)) or s >= prev)):
            dst_matrix[cond] = np.asarray(matrix, dtype=float)
            dst_score[cond] = s


def _merge_single_ch_accumulator(dst: ChTypeAccumulator, src: ChTypeAccumulator) -> None:
    dst.subjects.update(src.subjects)
    dst.run_count += int(src.run_count)
    dst.runs_by_condition.update(src.runs_by_condition)
    dst.module_present.update(src.module_present)
    dst.module_missing.update(src.module_missing)

    _merge_list_map(dst.std_dist_by_condition, src.std_dist_by_condition)
    _merge_list_map(dst.std_dist_mean_by_condition, src.std_dist_mean_by_condition)
    _merge_list_map(dst.std_dist_upper_by_condition, src.std_dist_upper_by_condition)
    _merge_list_map(dst.std_heatmap_runs_by_condition, src.std_heatmap_runs_by_condition)
    _merge_list_map(dst.std_epoch_counts_by_condition, src.std_epoch_counts_by_condition)
    dst.std_window_profiles.extend(src.std_window_profiles)
    for cond, profiles in src.std_window_profiles_by_condition.items():
        dst.std_window_profiles_by_condition[cond].extend(profiles)

    _merge_list_map(dst.ptp_dist_by_condition, src.ptp_dist_by_condition)
    _merge_list_map(dst.ptp_dist_mean_by_condition, src.ptp_dist_mean_by_condition)
    _merge_list_map(dst.ptp_dist_upper_by_condition, src.ptp_dist_upper_by_condition)
    _merge_list_map(dst.ptp_heatmap_runs_by_condition, src.ptp_heatmap_runs_by_condition)
    _merge_list_map(dst.ptp_epoch_counts_by_condition, src.ptp_epoch_counts_by_condition)
    dst.ptp_window_profiles.extend(src.ptp_window_profiles)
    for cond, profiles in src.ptp_window_profiles_by_condition.items():
        dst.ptp_window_profiles_by_condition[cond].extend(profiles)

    _merge_list_map(dst.psd_ratio_by_condition, src.psd_ratio_by_condition)
    _merge_list_map(dst.psd_harmonics_ratio_by_condition, src.psd_harmonics_ratio_by_condition)
    dst.psd_profiles.extend(src.psd_profiles)
    for cond, profiles in src.psd_profiles_by_condition.items():
        dst.psd_profiles_by_condition[cond].extend(profiles)

    _merge_list_map(dst.ecg_corr_by_condition, src.ecg_corr_by_condition)
    _merge_list_map(dst.eog_corr_by_condition, src.eog_corr_by_condition)

    _merge_list_map(dst.muscle_scalar_by_condition, src.muscle_scalar_by_condition)
    _merge_list_map(dst.muscle_mean_by_condition, src.muscle_mean_by_condition)
    dst.muscle_profiles.extend(src.muscle_profiles)
    for cond, profiles in src.muscle_profiles_by_condition.items():
        dst.muscle_profiles_by_condition[cond].extend(profiles)

    for subject, profiles in src.std_subject_profiles.items():
        dst.std_subject_profiles[subject].extend(profiles)
    for subject, profiles in src.ptp_subject_profiles.items():
        dst.ptp_subject_profiles[subject].extend(profiles)

    _merge_representative_state(
        dst.std_heatmap_by_condition,
        dst.std_heatmap_score_by_condition,
        dst.std_heatmap_score_history,
        src.std_heatmap_by_condition,
        src.std_heatmap_score_by_condition,
        src.std_heatmap_score_history,
    )
    _merge_representative_state(
        dst.ptp_heatmap_by_condition,
        dst.ptp_heatmap_score_by_condition,
        dst.ptp_heatmap_score_history,
        src.ptp_heatmap_by_condition,
        src.ptp_heatmap_score_by_condition,
        src.ptp_heatmap_score_history,
    )
    _merge_upper_tail_state(
        dst.std_heatmap_upper_by_condition,
        dst.std_heatmap_upper_score_by_condition,
        src.std_heatmap_upper_by_condition,
        src.std_heatmap_upper_score_by_condition,
    )
    _merge_upper_tail_state(
        dst.ptp_heatmap_upper_by_condition,
        dst.ptp_heatmap_upper_score_by_condition,
        src.ptp_heatmap_upper_by_condition,
        src.ptp_heatmap_upper_score_by_condition,
    )

    _merge_sum_count_dict(
        dst.std_heatmap_sum_by_condition,
        dst.std_heatmap_count_by_condition,
        src.std_heatmap_sum_by_condition,
        src.std_heatmap_count_by_condition,
    )
    _merge_sum_count_dict(
        dst.ptp_heatmap_sum_by_condition,
        dst.ptp_heatmap_count_by_condition,
        src.ptp_heatmap_sum_by_condition,
        src.ptp_heatmap_count_by_condition,
    )

    _merge_topomap_mean_dict(
        dst.std_topomap_by_condition,
        dst.std_topomap_count_by_condition,
        src.std_topomap_by_condition,
        src.std_topomap_count_by_condition,
    )
    _merge_topomap_mean_dict(
        dst.ptp_topomap_by_condition,
        dst.ptp_topomap_count_by_condition,
        src.ptp_topomap_by_condition,
        src.ptp_topomap_count_by_condition,
    )
    _merge_topomap_mean_dict(
        dst.psd_topomap_by_condition,
        dst.psd_topomap_count_by_condition,
        src.psd_topomap_by_condition,
        src.psd_topomap_count_by_condition,
    )
    _merge_topomap_mean_dict(
        dst.ecg_topomap_by_condition,
        dst.ecg_topomap_count_by_condition,
        src.ecg_topomap_by_condition,
        src.ecg_topomap_count_by_condition,
    )
    _merge_topomap_mean_dict(
        dst.eog_topomap_by_condition,
        dst.eog_topomap_count_by_condition,
        src.eog_topomap_by_condition,
        src.eog_topomap_count_by_condition,
    )

    dst.run_rows.extend(src.run_rows)
    dst.source_paths.update(src.source_paths)


def _group_run_records_by_subject(run_records: Dict[str, RunRecord]) -> List[List[RunRecord]]:
    groups: Dict[str, List[RunRecord]] = defaultdict(list)
    for run_key in sorted(run_records):
        record = run_records[run_key]
        if record.meta.subject != "n/a":
            group_key = record.meta.subject
        else:
            group_key = f"no_subject::{run_key}"
        groups[group_key].append(record)
    return [groups[key] for key in sorted(groups)]


def _process_subject_batch(records: Sequence[RunRecord]) -> Dict[str, ChTypeAccumulator]:
    """Process one subject batch fully inside a worker."""
    local_acc: Dict[str, ChTypeAccumulator] = {ch: ChTypeAccumulator() for ch in CH_TYPES}
    for record in sorted(records, key=lambda rec: rec.meta.run_key):
        _update_accumulator_for_run(local_acc, record)
    return local_acc


def _build_accumulators_for_runs(
    run_records: Dict[str, RunRecord],
    n_jobs: int = 1,
) -> Dict[str, ChTypeAccumulator]:
    """Build channel-type accumulators with optional subject-parallel workers."""
    n_jobs = _normalize_n_jobs(n_jobs)
    acc_by_type: Dict[str, ChTypeAccumulator] = {ch: ChTypeAccumulator() for ch in CH_TYPES}
    run_keys = sorted(run_records)

    if n_jobs == 1 or len(run_keys) <= 1:
        for run_key in run_keys:
            _update_accumulator_for_run(acc_by_type, run_records[run_key])
        return acc_by_type

    if Parallel is None or delayed is None:
        print("___MEGqc___: joblib is unavailable; falling back to sequential run processing.")
        for run_key in run_keys:
            _update_accumulator_for_run(acc_by_type, run_records[run_key])
        return acc_by_type

    subject_batches = _group_run_records_by_subject(run_records)
    print(
        "___MEGqc___: Parallel subject processing enabled: "
        f"n_jobs={n_jobs}, subject_batches={len(subject_batches)}."
    )
    try:
        partials = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_process_subject_batch)(batch) for batch in subject_batches
        )
        for part in partials:
            for ch in CH_TYPES:
                _merge_single_ch_accumulator(acc_by_type[ch], part[ch])
    except Exception as exc:
        print(f"___MEGqc___: Parallel processing failed ({exc}); falling back to sequential processing.")
        acc_by_type = {ch: ChTypeAccumulator() for ch in CH_TYPES}
        for run_key in run_keys:
            _update_accumulator_for_run(acc_by_type, run_records[run_key])
    return acc_by_type


def make_group_plots_meg_qc(
    dataset_path: str,
    derivatives_base: Optional[str] = None,
    n_jobs: int = 1,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
) -> Dict[str, Path]:
    """Build dataset-level QA reports from saved per-run derivatives.

    Parameters
    ----------
    dataset_path : str
        Path to the BIDS dataset.
    derivatives_base : str, optional
        Optional external parent directory for derivatives, matching the
        behavior of the main MEGqc plotting pipeline.
    n_jobs : int
        Number of parallel workers for subject-level run processing. Use ``1``
        for sequential mode, or ``-1`` to use all available cores.
    analysis_mode : str
        Analysis root selection mode (``legacy``, ``new``, ``reuse``, ``latest``).
        Plotting typically uses ``legacy`` or ``reuse``/``latest``.
    analysis_id : str, optional
        Profile ID used with ``analysis_mode='reuse'``.

    Returns
    -------
    dict
        Mapping ``{"report": Path(...)}`` for the generated dataset-level HTML.
    """

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
    dataset_name = os.path.basename(os.path.normpath(dataset_path))

    # Reports always go to the resolved (external) path.
    reports_dir = Path(megqc_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Source: prefer the external path, but fall back to the original dataset
    # when calculation was run without an external output path (Scenario C).
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
                f"___MEGqc___: Group QA: Scenario C — "
                f"reading calculation from original dataset: {calculation_dir}"
            )

    if not calculation_dir.exists():
        print(f"___MEGqc___: Group QA: calculation folder not found: {calculation_dir}")
        return {}

    run_records = _discover_run_records(calculation_dir)
    if not run_records:
        print(f"___MEGqc___: Group QA: no run-level TSV derivatives found under {calculation_dir}")
        return {}

    acc_by_type = _build_accumulators_for_runs(run_records, n_jobs=n_jobs)

    settings_snapshot = _load_settings_snapshot(source_megqc_root)
    combined_acc = _combine_accumulators(acc_by_type)

    # Build tab accumulators, filtering out types with no data.
    tab_accumulators: Dict[str, ChTypeAccumulator] = {}
    # Only add Combined tab if at least one of mag or grad has data
    if acc_by_type["mag"].run_count > 0 or acc_by_type["grad"].run_count > 0:
        tab_accumulators["Combined (mag+grad)"] = combined_acc
    if acc_by_type["mag"].run_count > 0:
        tab_accumulators["MAG"] = acc_by_type["mag"]
    if acc_by_type["grad"].run_count > 0:
        tab_accumulators["GRAD"] = acc_by_type["grad"]
    if acc_by_type["eeg"].run_count > 0:
        tab_accumulators["EEG"] = acc_by_type["eeg"]
    if all(acc.run_count == 0 for acc in tab_accumulators.values()):
        print("___MEGqc___: Group QA: no reports were generated.")
        return {}

    # ── Per-modality separate HTML reports (no combined agnostic report) ──
    # MEG report: Combined + MAG + GRAD tabs (only if any MEG data exists)
    meg_accs = {k: v for k, v in tab_accumulators.items()
                if k in ("Combined (mag+grad)", "MAG", "GRAD")}
    if any(acc.run_count > 0 for acc in meg_accs.values()):
        meg_html = _build_report_html(
            dataset_name=dataset_name,
            tab_accumulators=meg_accs,
            settings_snapshot=settings_snapshot,
        )
        meg_report_dir = reports_dir / "meg"
        meg_report_dir.mkdir(parents=True, exist_ok=True)
        meg_path = meg_report_dir / f"QA_group_report_{dataset_name}_meg.html"
        meg_path.write_text(meg_html, encoding="utf-8")
        print(f"___MEGqc___:   MEG report: {meg_path}")

    # EEG report: EEG tab only (only if EEG data exists)
    eeg_acc = tab_accumulators.get("EEG")
    if eeg_acc is not None and eeg_acc.run_count > 0:
        eeg_html = _build_report_html(
            dataset_name=dataset_name,
            tab_accumulators={"EEG": eeg_acc},
            settings_snapshot=settings_snapshot,
        )
        eeg_report_dir = reports_dir / "eeg"
        eeg_report_dir.mkdir(parents=True, exist_ok=True)
        eeg_path = eeg_report_dir / f"QA_group_report_{dataset_name}_eeg.html"
        eeg_path.write_text(eeg_html, encoding="utf-8")
        print(f"___MEGqc___:   EEG report: {eeg_path}")

    # Determine primary report path for return value
    meg_accs_check = {k: v for k, v in tab_accumulators.items()
                      if k in ("Combined (mag+grad)", "MAG", "GRAD")}
    if any(acc.run_count > 0 for acc in meg_accs_check.values()):
        primary_path = reports_dir / "meg" / f"QA_group_report_{dataset_name}_meg.html"
    else:
        primary_path = reports_dir / "eeg" / f"QA_group_report_{dataset_name}_eeg.html"

    return {"report": primary_path}
