#!/usr/bin/env python3
"""Simple wrapper for MEGnet-neuro end-to-end ICA labeling and cleanup."""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as op
import re
import sys
from pathlib import Path

import matplotlib
import mne
import numpy as np
from scipy import signal


def _bootstrap_package_root() -> None:
    """Allow running this file directly from any CWD.

    When executed as `python /path/to/easy_megnet.py`, Python only adds that
    script directory to `sys.path`. We add the repo/package root (the parent
    that contains the `MEGnet/` package directory) so absolute imports work.
    """
    if __package__:
        return

    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        if (parent / "MEGnet").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_bootstrap_package_root()

from MEGnet.megnet_init import main as megnet_init
from MEGnet.prep_inputs.ICA import load_data
from MEGnet.prep_inputs.ICA import main as run_ica_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS_ID_TO_NAME = {
    0: "Neural/other",
    1: "Eye blink (VEOG)",
    2: "Cardiac (ECG/EKG)",
    3: "Horizontal eye movement (saccade/HEOG)",
}
PROBABILITY_PRINT_PRECISION = 12


def _fmt_float(value: float, precision: int = PROBABILITY_PRINT_PRECISION) -> str:
    if not np.isfinite(float(value)):
        return ""
    return f"{float(value):.{precision}g}"


def _class_id_to_prob_col(class_id: int) -> str:
    if class_id == 0:
        return "prob_class0_neural_other"
    if class_id == 1:
        return "prob_class1_blink_veog"
    if class_id == 2:
        return "prob_class2_cardiac_ecg"
    if class_id == 3:
        return "prob_class3_saccade_heog"
    return f"prob_class{class_id}"


def _file_base(filename: str, outbasename: str | None) -> str:
    if outbasename:
        return outbasename
    return Path(filename).stem


def _to_int_list(values) -> list[int]:
    arr = np.asarray(values).reshape(-1)
    return [int(x) for x in arr]


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.std(arr))
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _parse_channel_list(value: str | None) -> list[str]:
    if value is None:
        return []
    out = [x.strip() for x in re.split(r"[,;]", value) if x.strip()]
    return out


def _get_meg_system(raw: mne.io.BaseRaw) -> str:
    try:
        from mne.channels.channels import _get_meg_system

        return str(_get_meg_system(raw.info))
    except Exception:
        return ""


def _is_ctf_system(raw: mne.io.BaseRaw) -> bool:
    return "CTF" in _get_meg_system(raw).upper()


def _pick_by_channel_name(raw: mne.io.BaseRaw, channel_names: list[str]) -> np.ndarray:
    if len(channel_names) == 0:
        return np.array([], dtype=int)

    name_to_idx = {name.upper(): idx for idx, name in enumerate(raw.ch_names)}
    picks: list[int] = []
    for channel_name in channel_names:
        key = channel_name.strip().upper()
        if key in name_to_idx:
            picks.append(name_to_idx[key])
            continue
        # Allow partial matching to accommodate vendor-specific naming variants.
        partial = [idx for idx, ch_name in enumerate(raw.ch_names) if key in ch_name.upper()]
        picks.extend(partial)

    if len(picks) == 0:
        return np.array([], dtype=int)
    return np.asarray(sorted(set(picks)), dtype=int)


def _pick_reference_channels(
    raw: mne.io.BaseRaw,
    kind: str,
    preferred_channel_names: list[str] | None = None,
) -> np.ndarray:
    preferred_channel_names = preferred_channel_names or []
    preferred_picks = _pick_by_channel_name(raw, preferred_channel_names)
    if len(preferred_picks) > 0:
        return preferred_picks

    if kind == "ecg":
        picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=True, eog=False, ref_meg=False)
        if len(picks) > 0:
            return picks
        keys = ("ECG", "EKG", "HEART", "CARD", "PULSE", "PPG")
    elif kind == "eog":
        picks = mne.pick_types(raw.info, meg=False, eeg=False, ecg=False, eog=True, ref_meg=False)
        if len(picks) > 0:
            return picks
        keys = ("EOG", "HEOG", "VEOG", "EYE", "BLINK")
    else:
        return np.array([], dtype=int)

    fallback = []
    for idx, name in enumerate(raw.ch_names):
        desc = str(raw.info["chs"][idx].get("desc", ""))
        probe = f"{name} {desc}".upper()
        if any(key in probe for key in keys):
            fallback.append(idx)

    if len(fallback) == 0 and _is_ctf_system(raw):
        system = _get_meg_system(raw) or "CTF"
        if kind == "ecg":
            print(
                f"{system} dataset detected: no ECG channel could be inferred automatically. "
                "If ECG is stored as EEG/misc, pass --ecg-channel."
            )
        else:
            print(
                f"{system} dataset detected: no EOG channel could be inferred automatically. "
                "If EOG is stored as EEG/misc, pass --eog-channels."
            )

    return np.asarray(fallback, dtype=int)


def _best_reference_match(component_signal: np.ndarray, reference_signals: np.ndarray) -> tuple[int, float] | None:
    if reference_signals.size == 0:
        return None

    comp_z = _zscore(component_signal)
    best_idx = -1
    best_corr = 0.0
    for ref_idx in range(reference_signals.shape[0]):
        ref_z = _zscore(reference_signals[ref_idx])
        if np.all(ref_z == 0) or np.all(comp_z == 0):
            corr = 0.0
        else:
            corr = float(np.corrcoef(comp_z, ref_z)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        if best_idx < 0 or abs(corr) > abs(best_corr):
            best_idx = ref_idx
            best_corr = corr

    if best_idx < 0:
        return None
    return best_idx, best_corr


def _save_trace_panel_plot(
    t: np.ndarray,
    traces: list[np.ndarray],
    labels: list[str],
    out_path: str,
    title: str,
    colors: list[str] | None = None,
) -> None:
    if len(traces) == 0:
        return
    fig, axes = plt.subplots(len(traces), 1, figsize=(13, max(3.0, 2.2 * len(traces))), sharex=True)
    if len(traces) == 1:
        axes = [axes]

    if colors is None:
        colors = ["tab:blue"] * len(traces)
    if len(colors) < len(traces):
        colors = colors + [colors[-1]] * (len(traces) - len(colors))

    for idx, (ax, trace, label) in enumerate(zip(axes, traces, labels)):
        ax.plot(t, trace, lw=1.1, color=colors[idx])
        ax.set_ylabel("z-score")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_reference_comparisons(
    raw_fif_path: str,
    ica_path: str,
    classes: list[int],
    output_dir: str,
    max_seconds: float = 60.0,
    ecg_channel_names: list[str] | None = None,
    eog_channel_names: list[str] | None = None,
) -> dict:
    if not op.exists(raw_fif_path):
        return {"status": "skipped", "reason": f"missing_data_file:{raw_fif_path}"}
    if not op.exists(ica_path):
        return {"status": "skipped", "reason": f"missing_ica_file:{ica_path}"}

    os.makedirs(output_dir, exist_ok=True)

    raw = mne.io.read_raw_fif(raw_fif_path, preload=True, allow_maxshield=True)
    raw_meg = raw.copy().pick_types(meg=True, eeg=False, ref_meg=False)
    ica = mne.preprocessing.read_ica(ica_path)
    sources = ica.get_sources(raw_meg).get_data()
    sfreq = float(raw.info["sfreq"])
    n_show = min(sources.shape[1], max(1, int(max_seconds * sfreq)))
    t = np.arange(n_show) / sfreq

    rows = []
    plot_paths = []

    # Cardiac handling
    ecg_picks = _pick_reference_channels(
        raw,
        "ecg",
        preferred_channel_names=ecg_channel_names,
    )
    ecg_comp_indices = [idx for idx, class_id in enumerate(classes) if class_id == 2]
    if len(ecg_comp_indices) > 0 and len(ecg_picks) > 0:
        cardiac_rows_before = len(rows)
        ecg_data = raw.get_data(picks=ecg_picks)
        ecg_names = [raw.ch_names[p] for p in ecg_picks]
        all_ecg_names = ";".join(ecg_names)
        for comp_idx in ecg_comp_indices:
            match = _best_reference_match(sources[comp_idx], ecg_data)
            if match is None:
                continue
            ref_idx, corr = match
            ref_name = ecg_names[ref_idx]

            ic_trace = _zscore(sources[comp_idx][:n_show])
            ref_trace = _zscore(ecg_data[ref_idx][:n_show])

            out_name = f"IC{comp_idx + 1:02d}_cardiac_{_sanitize_name(ref_name)}.png"
            out_path = op.join(output_dir, out_name)
            _save_trace_panel_plot(
                t=t,
                traces=[ic_trace, ref_trace],
                labels=[
                    f"IC{comp_idx + 1} ({CLASS_ID_TO_NAME.get(classes[comp_idx], 'Unknown')})",
                    f"ECG channel {ref_name} (corr={corr:.3f})",
                ],
                out_path=out_path,
                title=f"Cardiac comparison: IC{comp_idx + 1} vs {ref_name}",
                colors=["tab:blue", "tab:red"],
            )

            plot_paths.append(out_path)
            rows.append(
                {
                    "artifact_type": "cardiac",
                    "component_index_0based": comp_idx,
                    "component_number_1based": comp_idx + 1,
                    "predicted_class_id": classes[comp_idx],
                    "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                    "best_reference_channel": ref_name,
                    "all_reference_channels": all_ecg_names,
                    "correlation": f"{corr:.6f}",
                    "plot_file": out_path,
                }
            )
        # If matching fails for all components, still output IC-only cardiac plots.
        if len(rows) == cardiac_rows_before:
            traces = []
            labels = []
            for comp_idx in ecg_comp_indices:
                traces.append(_zscore(sources[comp_idx][:n_show]))
                labels.append(f"IC{comp_idx + 1} ({CLASS_ID_TO_NAME.get(classes[comp_idx], 'Unknown')})")
            out_path = op.join(output_dir, "IC_cardiac_only.png")
            _save_trace_panel_plot(
                t=t,
                traces=traces,
                labels=labels,
                out_path=out_path,
                title="Cardiac IC-only plot (ECG match unavailable)",
                colors=["tab:blue"] * len(traces),
            )
            plot_paths.append(out_path)
            for comp_idx in ecg_comp_indices:
                rows.append(
                    {
                        "artifact_type": "cardiac_ic_only",
                        "component_index_0based": comp_idx,
                        "component_number_1based": comp_idx + 1,
                        "predicted_class_id": classes[comp_idx],
                        "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                        "best_reference_channel": "",
                        "all_reference_channels": all_ecg_names,
                        "correlation": "",
                        "plot_file": out_path,
                    }
                )
    elif len(ecg_comp_indices) > 0:
        # If ECG is unavailable, still provide IC-only cardiac plots.
        traces = []
        labels = []
        for comp_idx in ecg_comp_indices:
            traces.append(_zscore(sources[comp_idx][:n_show]))
            labels.append(f"IC{comp_idx + 1} ({CLASS_ID_TO_NAME.get(classes[comp_idx], 'Unknown')})")

        out_path = op.join(output_dir, "IC_cardiac_only.png")
        _save_trace_panel_plot(
            t=t,
            traces=traces,
            labels=labels,
            out_path=out_path,
            title="Cardiac IC-only plot (no ECG channel available)",
            colors=["tab:blue"] * len(traces),
        )
        plot_paths.append(out_path)
        for comp_idx in ecg_comp_indices:
            rows.append(
                {
                    "artifact_type": "cardiac_ic_only",
                    "component_index_0based": comp_idx,
                    "component_number_1based": comp_idx + 1,
                    "predicted_class_id": classes[comp_idx],
                    "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                    "best_reference_channel": "",
                    "all_reference_channels": "",
                    "correlation": "",
                    "plot_file": out_path,
                }
            )

    # Ocular handling
    eog_picks = _pick_reference_channels(
        raw,
        "eog",
        preferred_channel_names=eog_channel_names,
    )
    ocular_candidates = {1: [idx for idx, cls in enumerate(classes) if cls == 1], 3: [idx for idx, cls in enumerate(classes) if cls == 3]}
    if len(ocular_candidates[1]) > 0 or len(ocular_candidates[3]) > 0:
        selected = {}
        class_meta = {
            1: {"label": "vEOG", "color": "tab:green"},
            3: {"label": "hEOG", "color": "tab:orange"},
        }

        eog_data = np.array([])
        eog_names: list[str] = []
        all_eog_names = ""
        if len(eog_picks) > 0:
            eog_data = raw.get_data(picks=eog_picks)
            eog_names = [raw.ch_names[p] for p in eog_picks]
            all_eog_names = ";".join(eog_names)

        for class_id in (1, 3):
            if len(ocular_candidates[class_id]) == 0:
                continue
            if len(eog_picks) > 0:
                best = None
                for comp_idx in ocular_candidates[class_id]:
                    match = _best_reference_match(sources[comp_idx], eog_data)
                    if match is None:
                        continue
                    ref_idx, corr = match
                    entry = {
                        "comp_idx": comp_idx,
                        "corr": corr,
                        "ref_name": eog_names[ref_idx],
                    }
                    if best is None or abs(corr) > abs(best["corr"]):
                        best = entry
                if best is None:
                    comp_idx = ocular_candidates[class_id][0]
                    best = {"comp_idx": comp_idx, "corr": None, "ref_name": ""}
                selected[class_id] = best
            else:
                comp_idx = ocular_candidates[class_id][0]
                selected[class_id] = {"comp_idx": comp_idx, "corr": None, "ref_name": ""}

        if selected:
            traces = []
            labels = []
            colors = []
            for class_id in (1, 3):
                if class_id not in selected:
                    continue
                entry = selected[class_id]
                comp_idx = entry["comp_idx"]
                meta = class_meta[class_id]
                traces.append(_zscore(sources[comp_idx][:n_show]))
                if entry["corr"] is None:
                    labels.append(f"{meta['label']} IC{comp_idx + 1} (no EOG channel)")
                else:
                    labels.append(
                        f"{meta['label']} IC{comp_idx + 1} (best {entry['ref_name']}, corr={entry['corr']:.3f})"
                    )
                colors.append(meta["color"])

            # Add each available EOG channel in its own panel.
            for idx, eog_name in enumerate(eog_names):
                traces.append(_zscore(eog_data[idx][:n_show]))
                labels.append(f"EOG channel {eog_name}")
                colors.append("tab:red")

            if len(eog_names) > 0:
                ocular_out_path = op.join(output_dir, "IC_ocular_combined_all_EOG.png")
                ocular_title = "Ocular comparison: vEOG/hEOG ICs + all EOG channels"
            else:
                ocular_out_path = op.join(output_dir, "IC_ocular_only.png")
                ocular_title = "Ocular IC-only plot (no EOG channel available)"

            _save_trace_panel_plot(
                t=t,
                traces=traces,
                labels=labels,
                out_path=ocular_out_path,
                title=ocular_title,
                colors=colors,
            )
            plot_paths.append(ocular_out_path)

            for class_id in (1, 3):
                if class_id not in selected:
                    continue
                entry = selected[class_id]
                comp_idx = entry["comp_idx"]
                rows.append(
                    {
                        "artifact_type": "ocular_combined" if len(eog_names) > 0 else "ocular_ic_only",
                        "component_index_0based": comp_idx,
                        "component_number_1based": comp_idx + 1,
                        "predicted_class_id": classes[comp_idx],
                        "predicted_class_name": CLASS_ID_TO_NAME.get(classes[comp_idx], "Unknown"),
                        "best_reference_channel": entry["ref_name"],
                        "all_reference_channels": all_eog_names,
                        "correlation": "" if entry["corr"] is None else f"{entry['corr']:.6f}",
                        "plot_file": ocular_out_path,
                    }
                )

    if not rows:
        return {"status": "skipped", "reason": "no_matching_components_or_reference_channels", "plots": []}

    csv_path = op.join(output_dir, "comparison_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "artifact_type",
                "component_index_0based",
                "component_number_1based",
                "predicted_class_id",
                "predicted_class_name",
                "best_reference_channel",
                "all_reference_channels",
                "correlation",
                "plot_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {"status": "ok", "plots": plot_paths, "summary_csv": csv_path}


def _classify_ica_with_probabilities(
    results_dir: str,
    outbasename: str | None,
    filename: str,
) -> tuple[list[int], list[int], np.ndarray]:
    from scipy.io import loadmat
    import MEGnet
    from MEGnet.megnet_utilities import fPredictChunkAndVoting_parrallel

    model_path = op.join(MEGnet.__path__[0], "model_v2k3", "model_v2.keras")
    if not op.exists(model_path):
        print('___MEGqc___: MEGnet model weights not found — downloading from HuggingFace...')
        megnet_init()
    else:
        print('___MEGqc___: MEGnet model weights found.')

    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import inspect as _inspect

    _orig_from_config = keras.src.ops.operation.Operation.from_config

    @classmethod
    def _lenient_from_config(cls, config):
        if isinstance(config, dict):
            try:
                sig = _inspect.signature(cls.__init__)
                valid = set(sig.parameters.keys()) - {"self"}
                cleaned = {k: v for k, v in config.items()
                           if k in valid or k in ("name", "dtype", "trainable")}
                return _orig_from_config.__func__(cls, cleaned)
            except Exception:
                pass
        return _orig_from_config.__func__(cls, config)

    keras.src.ops.operation.Operation.from_config = _lenient_from_config
    k_model = keras.models.load_model(model_path)
    keras.src.ops.operation.Operation.from_config = _orig_from_config

    file_base = outbasename if outbasename is not None else Path(filename).stem
    run_dir = op.join(results_dir, file_base)

    # Discover how many ICA component .mat files actually exist instead of
    # assuming a fixed count (short / cropped recordings may yield fewer).
    arr_sp_fnames = []
    for i in range(1, 51):          # generous upper bound
        p = op.join(run_dir, f"component{i}.mat")
        if op.isfile(p):
            arr_sp_fnames.append(p)
        else:
            break                    # components are numbered contiguously
    if not arr_sp_fnames:
        raise FileNotFoundError(
            f"No component*.mat files found in {run_dir}")

    arr_ts = loadmat(op.join(run_dir, "ICATimeSeries.mat"))["arrICATimeSeries"].T
    arr_sp = np.stack([loadmat(path)["array"] for path in arr_sp_fnames])

    # MEGnet model requires 15000 samples (60 s at 250 Hz).
    _min_len = 15000
    _actual_len = arr_ts.shape[-1]
    if _actual_len < _min_len:
        raise ValueError(
            f"ICA time series too short for MEGnet: {_actual_len} samples "
            f"(< {_min_len} / 60 s at 250 Hz). Recording is likely too short "
            f"or has been cropped."
        )

    probs_arr, _ = fPredictChunkAndVoting_parrallel(k_model, arr_ts, arr_sp)
    probs_arr = np.asarray(probs_arr, dtype=float)
    if probs_arr.ndim != 2:
        probs_arr = np.reshape(probs_arr, (probs_arr.shape[0], -1))

    # Normalize just in case tiny floating drift exists.
    row_sums = probs_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs_arr = probs_arr / row_sums

    classes = probs_arr.argmax(axis=1).astype(int).tolist()
    bads_idx = [idx for idx, class_id in enumerate(classes) if class_id in (1, 2, 3)]
    return classes, bads_idx, probs_arr


def _write_probability_table(
    run_dir: str,
    classes: list[int],
    probs: np.ndarray,
) -> str:
    out_csv = op.join(run_dir, "component_probabilities.csv")
    class_cols = [_class_id_to_prob_col(i) for i in range(probs.shape[1])]
    fieldnames = [
        "component_index_0based",
        "component_number_1based",
        "predicted_class_id",
        "predicted_class_name",
        "predicted_class_probability",
        *class_cols,
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(probs.shape[0]):
            predicted_class = int(classes[idx])
            row = {
                "component_index_0based": idx,
                "component_number_1based": idx + 1,
                "predicted_class_id": predicted_class,
                "predicted_class_name": CLASS_ID_TO_NAME.get(predicted_class, "Unknown"),
                "predicted_class_probability": _fmt_float(float(probs[idx, predicted_class])),
            }
            for class_id, col in enumerate(class_cols):
                row[col] = _fmt_float(float(probs[idx, class_id]))
            writer.writerow(row)
    return out_csv


def _parse_bool_cell(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _load_qc_score_table(table_path: str | None) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    if not table_path or not op.exists(table_path):
        return out

    with open(table_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                comp_idx = int(float(row.get("component_index_0based", "")))
            except Exception:
                continue

            try:
                score = float(row.get("aggregated_abs_score", "nan"))
            except Exception:
                score = float("nan")

            out[comp_idx] = {
                "score": score,
                "suggested": _parse_bool_cell(row.get("suggested_bad_by_find_bads", "")),
                "method": row.get("score_source_method", ""),
            }

    return out


def _rank_desc(values: list[float]) -> list[int | None]:
    arr = np.asarray(values, dtype=float)
    ranks: list[int | None] = [None] * int(arr.size)

    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return ranks

    finite_vals = arr[finite_idx]
    order = np.argsort(-finite_vals, kind="mergesort")
    prev_val: float | None = None
    prev_rank = 0
    for pos, ord_idx in enumerate(order):
        value = float(finite_vals[ord_idx])
        if prev_val is None or not np.isclose(value, prev_val, atol=1e-12, rtol=0.0):
            prev_rank = pos + 1
            prev_val = value
        ranks[int(finite_idx[ord_idx])] = prev_rank

    return ranks


def _write_combined_ranking_table(
    run_dir: str,
    classes: list[int],
    probs: np.ndarray,
    qc_score_tables: dict[str, str] | None = None,
) -> str:
    out_csv = op.join(run_dir, "component_ranking_combined.csv")
    n_comp = int(probs.shape[0])
    class_cols = [_class_id_to_prob_col(i) for i in range(probs.shape[1])]

    qc_score_tables = qc_score_tables or {}
    ecg_scores_by_comp = _load_qc_score_table(qc_score_tables.get("ECG"))
    eog_scores_by_comp = _load_qc_score_table(qc_score_tables.get("EOG"))

    def _get_prob(comp_idx: int, class_id: int) -> float:
        if 0 <= class_id < probs.shape[1]:
            return float(probs[comp_idx, class_id])
        return float("nan")

    predicted_probs = []
    prob_rank_values = {1: [], 2: [], 3: []}
    ecg_scores = []
    eog_scores = []
    for comp_idx in range(n_comp):
        class_id = int(classes[comp_idx]) if comp_idx < len(classes) else -1
        predicted_probs.append(_get_prob(comp_idx, class_id))
        for class_id_rank in prob_rank_values:
            prob_rank_values[class_id_rank].append(_get_prob(comp_idx, class_id_rank))
        ecg_scores.append(float(ecg_scores_by_comp.get(comp_idx, {}).get("score", float("nan"))))
        eog_scores.append(float(eog_scores_by_comp.get(comp_idx, {}).get("score", float("nan"))))

    rank_pred_prob = _rank_desc(predicted_probs)
    rank_prob_1 = _rank_desc(prob_rank_values[1])
    rank_prob_2 = _rank_desc(prob_rank_values[2])
    rank_prob_3 = _rank_desc(prob_rank_values[3])
    rank_ecg = _rank_desc(ecg_scores)
    rank_eog = _rank_desc(eog_scores)

    fieldnames = [
        "component_index_0based",
        "component_number_1based",
        "predicted_class_id",
        "predicted_class_name",
        "predicted_class_probability",
        "rank_predicted_class_probability_desc",
        *class_cols,
        "rank_prob_class1_blink_veog_desc",
        "rank_prob_class2_cardiac_ecg_desc",
        "rank_prob_class3_saccade_heog_desc",
        "ecg_score",
        "rank_ecg_score_desc",
        "ecg_suggested_bad_by_find_bads",
        "ecg_score_source_method",
        "eog_score",
        "rank_eog_score_desc",
        "eog_suggested_bad_by_find_bads",
        "eog_score_source_method",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for comp_idx in range(n_comp):
            predicted_class = int(classes[comp_idx]) if comp_idx < len(classes) else -1
            row = {
                "component_index_0based": comp_idx,
                "component_number_1based": comp_idx + 1,
                "predicted_class_id": predicted_class,
                "predicted_class_name": CLASS_ID_TO_NAME.get(predicted_class, "Unknown"),
                "predicted_class_probability": (
                    _fmt_float(predicted_probs[comp_idx]) if np.isfinite(predicted_probs[comp_idx]) else ""
                ),
                "rank_predicted_class_probability_desc": (
                    rank_pred_prob[comp_idx] if rank_pred_prob[comp_idx] is not None else ""
                ),
                "rank_prob_class1_blink_veog_desc": rank_prob_1[comp_idx] if rank_prob_1[comp_idx] is not None else "",
                "rank_prob_class2_cardiac_ecg_desc": rank_prob_2[comp_idx] if rank_prob_2[comp_idx] is not None else "",
                "rank_prob_class3_saccade_heog_desc": rank_prob_3[comp_idx] if rank_prob_3[comp_idx] is not None else "",
                "ecg_score": f"{ecg_scores[comp_idx]:.6f}" if np.isfinite(ecg_scores[comp_idx]) else "",
                "rank_ecg_score_desc": rank_ecg[comp_idx] if rank_ecg[comp_idx] is not None else "",
                "ecg_suggested_bad_by_find_bads": bool(
                    ecg_scores_by_comp.get(comp_idx, {}).get("suggested", False)
                ),
                "ecg_score_source_method": str(ecg_scores_by_comp.get(comp_idx, {}).get("method", "")),
                "eog_score": f"{eog_scores[comp_idx]:.6f}" if np.isfinite(eog_scores[comp_idx]) else "",
                "rank_eog_score_desc": rank_eog[comp_idx] if rank_eog[comp_idx] is not None else "",
                "eog_suggested_bad_by_find_bads": bool(
                    eog_scores_by_comp.get(comp_idx, {}).get("suggested", False)
                ),
                "eog_score_source_method": str(eog_scores_by_comp.get(comp_idx, {}).get("method", "")),
            }
            for class_id, col in enumerate(class_cols):
                value = _get_prob(comp_idx, class_id)
                row[col] = _fmt_float(value) if np.isfinite(value) else ""
            writer.writerow(row)

    return out_csv


def _build_report(file_base: str, classes: list[int], bads_idx: list[int]) -> dict:
    bads = set(bads_idx)
    records = []
    for idx, class_id in enumerate(classes):
        records.append(
            {
                "component_index_0based": idx,
                "component_number_1based": idx + 1,
                "class_id": class_id,
                "class_name": CLASS_ID_TO_NAME.get(class_id, "Unknown"),
                "remove_component": idx in bads,
            }
        )

    unique, counts = np.unique(np.asarray(classes), return_counts=True)
    class_counts = {
        CLASS_ID_TO_NAME.get(int(class_id), str(int(class_id))): int(count)
        for class_id, count in zip(unique, counts)
    }

    return {
        "file_base": file_base,
        "class_id_map": CLASS_ID_TO_NAME,
        "class_counts": class_counts,
        "remove_components_0based": bads_idx,
        "remove_components_1based": [idx + 1 for idx in bads_idx],
        "components": records,
    }


def _apply_ica_cleanup(
    raw_dataset: str, ica_path: str, bads_idx: list[int], out_clean_path: str, out_ica_applied_path: str
) -> None:
    raw = load_data(raw_dataset)
    ica = mne.preprocessing.read_ica(ica_path)
    ica.exclude = bads_idx
    ica.apply(raw)
    raw.save(out_clean_path, overwrite=True)
    ica.save(out_ica_applied_path, overwrite=True)


def _safe_fig_save(fig, out_path: str) -> None:
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _resolve_qc_out_dir(ica_file: str, data_file: str) -> str:
    data_stem = Path(data_file).stem
    return op.join(op.dirname(ica_file), "MEGnetExtPlots", data_stem)


def _ensure_megnet_qc_compat() -> None:
    # MEGnet monkey patches MNE topomap internals and misses this symbol in some envs.
    import MEGnet.prep_inputs.ICA as megnet_ica
    from mne.viz.topomap import _topomap_plot_sensors

    if not hasattr(megnet_ica, "_topomap_plot_sensors"):
        megnet_ica._topomap_plot_sensors = _topomap_plot_sensors


def _run_qc_plotting_fallback(ica_file: str, data_file: str, apply_filter: bool) -> None:
    out_dir = _resolve_qc_out_dir(ica_file=ica_file, data_file=data_file)
    os.makedirs(out_dir, exist_ok=True)

    ica = mne.preprocessing.read_ica(ica_file)
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
    if apply_filter:
        lfreq, hfreq = 1.0, 98.0
        line_freq = raw.info.get("line_freq", None)
        if line_freq:
            notch_freqs = np.arange(line_freq, hfreq, line_freq)
            if len(notch_freqs):
                raw.notch_filter(
                    notch_freqs,
                    picks=["meg", "eeg", "eog", "ecg"],
                    filter_length="auto",
                    notch_widths=2,
                    trans_bandwidth=1.0,
                    verbose=False,
                )
        raw.filter(
            lfreq,
            hfreq,
            picks=["meg", "eeg", "eog", "ecg"],
            filter_length="auto",
            method="fir",
            phase="zero",
            verbose=False,
        )

    exp_var = ica.get_explained_variance_ratio(raw)
    sen_types = set(raw.get_channel_types())
    with open(op.join(out_dir, "Explained_variance_ratio.csv"), "w", encoding="utf-8") as f:
        if "grad" not in sen_types and "mag" in exp_var:
            f.write("data_file,\tmag\n")
            f.write(f"{Path(data_file).name},\t{exp_var.get('mag', np.nan)}\n")
        elif "mag" not in sen_types and "grad" in exp_var:
            f.write("data_file,\tgrad\n")
            f.write(f"{Path(data_file).name},\t{exp_var.get('grad', np.nan)}\n")
        else:
            f.write("data_file,\tgrad,\tmag\n")
            f.write(f"{Path(data_file).name},\t{exp_var.get('grad', np.nan)},\t{exp_var.get('mag', np.nan)}\n")

    src_fig = ica.plot_sources(raw, picks=range(ica.n_components_), show=False, block=False)
    _safe_fig_save(src_fig, op.join(out_dir, "all_comp_time_series_plot.png"))

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, ref_meg=False)
    sources = ica.get_sources(raw_meg).get_data()
    sfreq = float(raw_meg.info["sfreq"])
    n_show = min(sources.shape[1], max(1, int(sfreq * 60)))
    t = np.arange(n_show) / sfreq
    for comp_idx in range(sources.shape[0]):
        sig = sources[comp_idx, :n_show]
        freqs, psd = signal.welch(sig, fs=sfreq, nperseg=min(2048, len(sig)))

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
        axes[0].plot(t, sig, lw=0.8, color="tab:blue")
        axes[0].set_title(f"ICA component {comp_idx} time series (first {n_show / sfreq:.1f}s)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("a.u.")
        axes[0].grid(alpha=0.2)

        axes[1].plot(freqs, psd, lw=0.8, color="tab:purple")
        axes[1].set_xlim(0, min(120, freqs.max()))
        axes[1].set_title("Welch PSD")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Power")
        axes[1].grid(alpha=0.2)
        fig.tight_layout()
        _safe_fig_save(fig, op.join(out_dir, f"properties_plot_IC{comp_idx:03d}.png"))

    for check in ("eog", "ecg"):
        try:
            _, scores = getattr(ica, f"find_bads_{check}")(raw)
        except Exception:
            continue
        try:
            score_arr = np.asarray(scores, dtype=float)
            if score_arr.size == 0:
                continue
            score_arr = np.squeeze(score_arr)
            n_comp = int(ica.n_components_)

            # Handle score arrays from single or multiple reference channels.
            # For multi-channel outputs (e.g., shape (2, n_components)),
            # aggregate to one value per component using max absolute score.
            if score_arr.ndim == 1:
                plot_scores = score_arr
            elif score_arr.ndim == 2:
                if score_arr.shape[1] == n_comp:
                    plot_scores = np.max(np.abs(score_arr), axis=0)
                elif score_arr.shape[0] == n_comp:
                    plot_scores = np.max(np.abs(score_arr), axis=1)
                else:
                    plot_scores = np.max(np.abs(score_arr), axis=0).reshape(-1)
            else:
                reduce_axes = tuple(range(score_arr.ndim - 1))
                plot_scores = np.max(np.abs(score_arr), axis=reduce_axes).reshape(-1)

            if plot_scores.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.bar(np.arange(plot_scores.size), plot_scores, color="tab:gray")
            ax.set_title(f"ICA component scores ({check.upper()})")
            ax.set_xlabel("ICA component index")
            ax.set_ylabel("Score")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            _safe_fig_save(fig, op.join(out_dir, f"score_plot_{check.upper()}.png"))
        except Exception as score_exc:
            print(f"Skipping {check.upper()} score plot in QC fallback: {score_exc}")

    print(f"QC fallback plots written to: {out_dir}")


def _apply_qc_filter_if_requested(raw: mne.io.BaseRaw, apply_filter: bool) -> None:
    if not apply_filter:
        return
    lfreq, hfreq = 1.0, 98.0
    line_freq = raw.info.get("line_freq", None)
    if line_freq:
        notch_freqs = np.arange(line_freq, hfreq, line_freq)
        if len(notch_freqs):
            raw.notch_filter(
                notch_freqs,
                picks=["meg", "eeg", "eog", "ecg"],
                filter_length="auto",
                notch_widths=2,
                trans_bandwidth=1.0,
                verbose=False,
            )
    raw.filter(
        lfreq,
        hfreq,
        picks=["meg", "eeg", "eog", "ecg"],
        filter_length="auto",
        method="fir",
        phase="zero",
        verbose=False,
    )


def _normalize_qc_scores(scores: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray | None]:
    score_arr = np.asarray(scores, dtype=float)
    if score_arr.size == 0:
        return np.array([], dtype=float), None

    score_arr = np.squeeze(score_arr)
    per_ref = None
    if score_arr.ndim == 1:
        if score_arr.size == n_comp:
            per_ref = score_arr.reshape(1, n_comp)
    elif score_arr.ndim == 2:
        if score_arr.shape[1] == n_comp:
            per_ref = score_arr
        elif score_arr.shape[0] == n_comp:
            per_ref = score_arr.T
        elif score_arr.size % n_comp == 0:
            per_ref = score_arr.reshape(-1, n_comp)

    if per_ref is not None:
        agg_abs = np.max(np.abs(per_ref), axis=0)
        return agg_abs, per_ref

    # Generic fallback if dimensions are unexpected.
    if score_arr.ndim == 1:
        agg_abs = np.abs(score_arr)
    else:
        reduce_axes = tuple(range(score_arr.ndim - 1))
        agg_abs = np.max(np.abs(score_arr), axis=reduce_axes).reshape(-1)

    if agg_abs.size > n_comp:
        agg_abs = agg_abs[:n_comp]
    elif agg_abs.size < n_comp:
        agg_abs = np.pad(agg_abs, (0, n_comp - agg_abs.size), constant_values=np.nan)
    return agg_abs, None


def _compute_manual_component_scores(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    check: str,
    preferred_channel_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    n_comp = int(ica.n_components_)
    ref_picks = _pick_reference_channels(
        raw,
        check,
        preferred_channel_names=preferred_channel_names,
    )
    if len(ref_picks) == 0:
        return np.zeros(n_comp, dtype=float), None, []

    raw_meg = raw.copy().pick_types(meg=True, eeg=False, ref_meg=False)
    src = ica.get_sources(raw_meg).get_data()
    ref = raw.get_data(picks=ref_picks)
    ref_names = [raw.ch_names[idx] for idx in ref_picks]

    n_time = min(src.shape[1], ref.shape[1])
    src = src[:, :n_time]
    ref = ref[:, :n_time]

    per_ref = np.zeros((len(ref_picks), n_comp), dtype=float)
    for ref_idx in range(len(ref_picks)):
        ref_z = _zscore(ref[ref_idx])
        for comp_idx in range(n_comp):
            src_z = _zscore(src[comp_idx])
            if np.all(ref_z == 0) or np.all(src_z == 0):
                corr = 0.0
            else:
                corr = float(np.corrcoef(src_z, ref_z)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            per_ref[ref_idx, comp_idx] = corr

    agg_abs = np.max(np.abs(per_ref), axis=0)
    return agg_abs, per_ref, ref_names


def _export_qc_score_tables(
    ica_file: str,
    data_file: str,
    apply_filter: bool,
    ecg_channel_names: list[str] | None = None,
    eog_channel_names: list[str] | None = None,
) -> dict:
    out_dir = _resolve_qc_out_dir(ica_file=ica_file, data_file=data_file)
    os.makedirs(out_dir, exist_ok=True)
    out_tables: dict[str, str] = {}

    ica = mne.preprocessing.read_ica(ica_file)
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True, preload=True)
    _apply_qc_filter_if_requested(raw, apply_filter=apply_filter)

    n_comp = int(ica.n_components_)
    for check in ("eog", "ecg"):
        preferred_channel_names = eog_channel_names if check == "eog" else ecg_channel_names
        ref_picks_for_find = _pick_reference_channels(
            raw,
            check,
            preferred_channel_names=preferred_channel_names,
        )
        ref_names_for_find = [raw.ch_names[idx] for idx in ref_picks_for_find]

        method = "find_bads"
        source_error = ""
        suggested = set()
        ref_names: list[str] = []
        per_ref: np.ndarray | None = None
        agg_abs: np.ndarray = np.array([], dtype=float)

        try:
            kwargs = {}
            if check == "ecg" and len(ref_names_for_find) > 0:
                kwargs["ch_name"] = ref_names_for_find[0]
            elif check == "eog" and len(ref_names_for_find) > 0:
                kwargs["ch_name"] = (
                    ref_names_for_find[0]
                    if len(ref_names_for_find) == 1
                    else ref_names_for_find
                )

            idxs, scores = getattr(ica, f"find_bads_{check}")(raw, **kwargs)
            agg_abs, per_ref = _normalize_qc_scores(scores=scores, n_comp=n_comp)
            suggested = set(int(i) for i in np.asarray(idxs).reshape(-1).tolist())
            if per_ref is not None:
                if len(ref_names_for_find) == per_ref.shape[0]:
                    ref_names = ref_names_for_find
                else:
                    ref_names = [f"ref_{i + 1}" for i in range(per_ref.shape[0])]
            if agg_abs.size == 0:
                raise RuntimeError("find_bads returned empty scores")
        except Exception as exc:
            method = "manual_corr"
            source_error = str(exc)
            agg_abs, per_ref, ref_names = _compute_manual_component_scores(
                ica=ica,
                raw=raw,
                check=check,
                preferred_channel_names=preferred_channel_names,
            )

        # If no reference channels exist, still emit a numeric table.
        if agg_abs.size == 0:
            method = "no_reference_channels"
            agg_abs = np.zeros(n_comp, dtype=float)
            per_ref = None
            ref_names = []

        # Always emit a score plot from the same values used in the table.
        try:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.bar(np.arange(agg_abs.size), agg_abs, color="tab:gray")
            ax.set_title(f"ICA component scores ({check.upper()}) [{method}]")
            ax.set_xlabel("ICA component index")
            ax.set_ylabel("Score")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            _safe_fig_save(fig, op.join(out_dir, f"score_plot_{check.upper()}.png"))
        except Exception as plot_exc:
            print(f"Skipping {check.upper()} score plot export: {plot_exc}")

        table_path = op.join(out_dir, f"score_table_{check.upper()}.csv")
        fieldnames = [
            "component_index_0based",
            "component_number_1based",
            "suggested_bad_by_find_bads",
            "aggregated_abs_score",
            "score_source_method",
            "reference_channel_names",
            "score_source_error",
        ]
        if per_ref is not None:
            fieldnames.extend([f"reference_{i + 1}_score" for i in range(per_ref.shape[0])])

        with open(table_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for comp_idx in range(n_comp):
                row = {
                    "component_index_0based": comp_idx,
                    "component_number_1based": comp_idx + 1,
                    "suggested_bad_by_find_bads": comp_idx in suggested,
                    "aggregated_abs_score": f"{float(agg_abs[comp_idx]):.6f}",
                    "score_source_method": method,
                    "reference_channel_names": ";".join(ref_names),
                    "score_source_error": source_error,
                }
                if per_ref is not None:
                    for ref_idx in range(per_ref.shape[0]):
                        row[f"reference_{ref_idx + 1}_score"] = f"{float(per_ref[ref_idx, comp_idx]):.6f}"
                writer.writerow(row)

        out_tables[check.upper()] = table_path

    return out_tables


def _run_qc_plotting(ica_file: str, data_file: str, apply_filter: bool, block: bool) -> None:
    from MEGnet.megnet_qc_plots import plot_all

    try:
        _ensure_megnet_qc_compat()
        plot_all(
            results_dir=None,
            ica_file=ica_file,
            data_file=data_file,
            apply_filter=apply_filter,
            block=block,
            apply_ica=False,
        )
    except Exception as exc:
        print(f"MEGnet QC plotting failed ({exc}); running wrapper fallback QC plotting.")
        _run_qc_plotting_fallback(ica_file=ica_file, data_file=data_file, apply_filter=apply_filter)
    finally:
        plt.close("all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run MEGnet preprocessing + ICA + classification + optional ICA application "
            "in one command."
        )
    )
    parser.add_argument(
        "--filename",
        required=True,
        help="Path to MEG dataset (.fif, .ds, .sqd, BTI path) used by MEGnet.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory where MEGnet output folder will be created.",
    )
    parser.add_argument(
        "--line-freq",
        type=float,
        choices=[50.0, 60.0],
        required=True,
        help="Mains frequency (50 or 60 Hz).",
    )
    parser.add_argument(
        "--filename-raw",
        default=None,
        help="Optional non-SSS raw FIF file for MEGIN bad-channel detection.",
    )
    parser.add_argument(
        "--outbasename",
        default=None,
        help="Optional output folder name inside --results-dir.",
    )
    parser.add_argument(
        "--bad-channels",
        default="",
        help="Comma-separated channels to drop before ICA (example: MEG0113,MEG2443).",
    )
    parser.add_argument(
        "--ecg-channel",
        default="",
        help=(
            "Optional ECG channel name override (recommended for CTF datasets where ECG is stored "
            "as EEG/misc)."
        ),
    )
    parser.add_argument(
        "--eog-channels",
        default="",
        help=(
            "Optional comma-separated EOG channel name overrides (recommended for CTF datasets "
            "where EOG is stored as EEG/misc)."
        ),
    )
    parser.add_argument(
        "--classify-only",
        action="store_true",
        help="Skip preprocessing/ICA generation and classify existing MEGnet outputs only.",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip megnet_init weight check/download.",
    )
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        help="Do not apply/remove predicted bad components from the raw data.",
    )
    parser.add_argument(
        "--run-qc",
        action="store_true",
        help="Run MEGnet QC plotting after classification.",
    )
    parser.add_argument(
        "--qc-apply-filter",
        action="store_true",
        help="Pass --apply_filter behavior to QC plotting.",
    )
    parser.add_argument(
        "--qc-block",
        action="store_true",
        help="Block QC plotting windows (interactive mode).",
    )
    parser.add_argument(
        "--run-ref-compare",
        action="store_true",
        help="Plot predicted cardiac/ocular ICs against ECG/EOG channels when available.",
    )
    parser.add_argument(
        "--compare-max-seconds",
        type=float,
        default=60.0,
        help="Duration in seconds to show per comparison plot.",
    )
    parser.add_argument(
        "--compare-out-dir",
        default=None,
        help="Optional output directory for IC-vs-reference comparison plots.",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Optional JSON report path. Default: <results-subdir>/megnet_summary.json",
    )
    return parser


def _exc_info(exc: Exception) -> dict:
    return {"type": exc.__class__.__name__, "message": str(exc)}


def _write_report(report_path: str, report: dict) -> None:
    parent = op.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    file_base = _file_base(args.filename, args.outbasename)
    results_subdir = op.join(args.results_dir, file_base)
    os.makedirs(results_subdir, exist_ok=True)
    report_path = args.report_file or op.join(results_subdir, "megnet_summary.json")

    report: dict = {
        "file_base": file_base,
        "input_filename": args.filename,
        "status": "running",
        "stages": {},
        "reference_channel_overrides": {
            "ecg": _parse_channel_list(args.ecg_channel),
            "eog": _parse_channel_list(args.eog_channels),
        },
    }
    _write_report(report_path, report)

    if not op.exists(args.filename):
        report["status"] = "failed"
        report["stages"]["input"] = {
            "status": "failed",
            "error": {"type": "FileNotFoundError", "message": f"Input dataset not found: {args.filename}"},
        }
        _write_report(report_path, report)
        print(f"Input dataset not found: {args.filename}")
        print(f"Summary report: {report_path}")
        return 1

    if not args.skip_init:
        try:
            megnet_init()
            report["stages"]["init"] = {"status": "ok"}
        except Exception as exc:
            report["status"] = "failed"
            report["stages"]["init"] = {"status": "failed", "error": _exc_info(exc)}
            _write_report(report_path, report)
            print(f"Initialization failed: {exc}")
            print(f"Summary report: {report_path}")
            return 1
    else:
        report["stages"]["init"] = {"status": "skipped"}
    _write_report(report_path, report)

    bad_channels = [x.strip() for x in args.bad_channels.split(",") if x.strip()]
    ecg_channel_names = _parse_channel_list(args.ecg_channel)
    eog_channel_names = _parse_channel_list(args.eog_channels)
    if len(ecg_channel_names) > 1:
        # `find_bads_ecg` accepts only one channel name; keep the first.
        ecg_channel_names = ecg_channel_names[:1]

    if not args.classify_only:
        try:
            run_ica_pipeline(
                args.filename,
                outbasename=args.outbasename,
                mains_freq=args.line_freq,
                save_preproc=True,
                save_ica=True,
                seedval=0,
                results_dir=args.results_dir,
                filename_raw=args.filename_raw,
                do_assess_bads=bool(args.filename_raw),
                bad_channels=bad_channels,
            )
            report["stages"]["pipeline"] = {"status": "ok"}
        except Exception as exc:
            report["status"] = "failed"
            report["stages"]["pipeline"] = {"status": "failed", "error": _exc_info(exc)}
            _write_report(report_path, report)
            print(f"Pipeline failed: {exc}")
            print(f"Summary report: {report_path}")
            return 1
    else:
        report["stages"]["pipeline"] = {"status": "skipped"}
    _write_report(report_path, report)

    try:
        classes, bads_idx, probs_arr = _classify_ica_with_probabilities(
            results_dir=args.results_dir,
            outbasename=args.outbasename,
            filename=args.filename,
        )
        prob_table_csv = _write_probability_table(
            run_dir=results_subdir,
            classes=classes,
            probs=probs_arr,
        )
        combined_ranking_csv = _write_combined_ranking_table(
            run_dir=results_subdir,
            classes=classes,
            probs=probs_arr,
            qc_score_tables=None,
        )
        report["stages"]["classify"] = {
            "status": "ok",
            "probability_table_csv": prob_table_csv,
            "combined_ranking_csv": combined_ranking_csv,
        }
        report["probability_table_csv"] = prob_table_csv
        report["combined_ranking_csv"] = combined_ranking_csv
    except Exception as exc:
        hint = None
        if "list index out of range" in str(exc):
            hint = (
                "Classification may fail when usable ICA time-series chunks cannot be generated "
                "(e.g., very short or malformed inputs)."
            )
        report["status"] = "failed"
        out = {"status": "failed", "error": _exc_info(exc)}
        if hint:
            out["hint"] = hint
        report["stages"]["classify"] = out
        _write_report(report_path, report)
        print(f"Classification failed: {exc}")
        if hint:
            print(hint)
        print(f"Summary report: {report_path}")
        return 1
    bads_idx = sorted(set(_to_int_list(bads_idx)))
    report.update(_build_report(file_base=file_base, classes=classes, bads_idx=bads_idx))
    _write_report(report_path, report)

    preproc_fif = op.join(results_subdir, f"{file_base}_250srate_meg.fif")
    ica_file = op.join(results_subdir, f"{file_base}_0-ica.fif")
    ica_applied_file = ica_file.replace("_0-ica.fif", "_0-ica_applied.fif")

    out_clean = None
    out_ica_applied = None
    if not args.skip_apply:
        try:
            if not op.exists(ica_file):
                raise FileNotFoundError(
                    f"Cannot apply ICA because this file is missing: {ica_file}. "
                    "Run without --classify-only, or generate ICA first."
                )
            out_clean = op.join(results_subdir, "ica_clean.fif")
            out_ica_applied = ica_applied_file
            _apply_ica_cleanup(
                raw_dataset=args.filename,
                ica_path=ica_file,
                bads_idx=bads_idx,
                out_clean_path=out_clean,
                out_ica_applied_path=out_ica_applied,
            )
            report["stages"]["apply"] = {"status": "ok", "ica_clean_file": out_clean, "ica_applied_file": out_ica_applied}
        except Exception as exc:
            report["status"] = "failed"
            report["stages"]["apply"] = {"status": "failed", "error": _exc_info(exc)}
            _write_report(report_path, report)
            print(f"ICA apply failed: {exc}")
            print(f"Summary report: {report_path}")
            return 1
    else:
        report["stages"]["apply"] = {"status": "skipped"}
    _write_report(report_path, report)

    if args.run_qc:
        try:
            if not op.exists(preproc_fif):
                raise FileNotFoundError(f"QC plotting requires this file: {preproc_fif}")
            qc_ica_file = out_ica_applied or (ica_applied_file if op.exists(ica_applied_file) else ica_file)
            if not op.exists(qc_ica_file):
                raise FileNotFoundError(f"QC plotting requires an ICA file: {qc_ica_file}")
        except Exception as exc:
            report.setdefault("warnings", []).append(f"QC setup failed: {exc}")
            report["stages"]["qc"] = {"status": "failed", "error": _exc_info(exc)}
            print(f"QC setup failed: {exc}")
        else:
            plot_error = None
            try:
                _run_qc_plotting(
                    ica_file=qc_ica_file,
                    data_file=preproc_fif,
                    apply_filter=args.qc_apply_filter,
                    block=args.qc_block,
                )
            except Exception as exc:
                plot_error = exc
                report.setdefault("warnings", []).append(f"QC plot generation failed: {exc}")
                print(f"QC plot generation failed: {exc}")

            table_error = None
            score_tables: dict[str, str] = {}
            try:
                score_tables = _export_qc_score_tables(
                    ica_file=qc_ica_file,
                    data_file=preproc_fif,
                    apply_filter=args.qc_apply_filter,
                    ecg_channel_names=ecg_channel_names,
                    eog_channel_names=eog_channel_names,
                )
            except Exception as exc:
                table_error = exc
                report.setdefault("warnings", []).append(f"QC score table export failed: {exc}")
                print(f"QC score table export failed: {exc}")

            report["qc"] = {
                "status": "ok" if table_error is None else "partial",
                "ica_file": qc_ica_file,
                "data_file": preproc_fif,
                "score_tables": score_tables,
            }
            if score_tables:
                try:
                    combined_ranking_csv = _write_combined_ranking_table(
                        run_dir=results_subdir,
                        classes=classes,
                        probs=probs_arr,
                        qc_score_tables=score_tables,
                    )
                    report["combined_ranking_csv"] = combined_ranking_csv
                    report["qc"]["combined_ranking_csv"] = combined_ranking_csv
                except Exception as exc:
                    report.setdefault("warnings", []).append(
                        f"Combined ranking table export failed: {exc}"
                    )
                    print(f"Combined ranking table export failed: {exc}")
            if plot_error:
                report["qc"]["plot_error"] = str(plot_error)
            if table_error:
                report["qc"]["table_error"] = str(table_error)

            if table_error is None:
                report["stages"]["qc"] = {"status": "ok"}
            else:
                report["stages"]["qc"] = {"status": "failed", "error": _exc_info(table_error)}
    else:
        report["stages"]["qc"] = {"status": "skipped"}
    _write_report(report_path, report)

    if args.run_ref_compare:
        try:
            compare_out_dir = args.compare_out_dir or op.join(results_subdir, "IC_ref_comparisons")
            compare_info = _run_reference_comparisons(
                raw_fif_path=preproc_fif,
                ica_path=ica_file if op.exists(ica_file) else ica_applied_file,
                classes=classes,
                output_dir=compare_out_dir,
                max_seconds=float(args.compare_max_seconds),
                ecg_channel_names=ecg_channel_names,
                eog_channel_names=eog_channel_names,
            )
            report["reference_comparison"] = compare_info
            report["stages"]["reference_comparison"] = {"status": "ok"}
        except Exception as exc:
            report.setdefault("warnings", []).append(f"Reference comparison failed: {exc}")
            report["stages"]["reference_comparison"] = {"status": "failed", "error": _exc_info(exc)}
            print(f"Reference comparison failed: {exc}")
    else:
        report["stages"]["reference_comparison"] = {"status": "skipped"}

    if report.get("status") == "running":
        report["status"] = "ok_with_warnings" if report.get("warnings") else "ok"

    _write_report(report_path, report)

    print(f"Results directory: {results_subdir}")
    print(f"Predicted class IDs for components 1..20: {classes}")
    print(f"Predicted removable components (0-based): {bads_idx}")
    print(f"Predicted removable components (1-based): {[idx + 1 for idx in bads_idx]}")
    if "probability_table_csv" in report:
        print(f"Probability table CSV: {report['probability_table_csv']}")
    if "combined_ranking_csv" in report:
        print(f"Combined ranking table CSV: {report['combined_ranking_csv']}")
    if args.run_ref_compare and "reference_comparison" in report:
        print(f"Reference comparison status: {report['reference_comparison'].get('status')}")
    if "qc" in report and isinstance(report["qc"], dict) and report["qc"].get("score_tables"):
        print(f"QC score tables: {report['qc']['score_tables']}")
    print(f"Summary report: {report_path}")
    if out_clean is not None and out_ica_applied is not None:
        print(f"ICA-cleaned raw file: {out_clean}")
        print(f"ICA file with exclude list: {out_ica_applied}")
    if report.get("warnings"):
        for warning in report["warnings"]:
            print(f"Warning: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
