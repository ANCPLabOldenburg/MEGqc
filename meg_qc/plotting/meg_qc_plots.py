import sys
import os
import ancpbids
import json
import csv
import datetime as dt
import html
from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style
from collections import defaultdict
from itertools import count
import re
from typing import Any, Dict, List, Sequence
from pprint import pprint
import gc
from ancpbids import DatasetOptions
from pathlib import Path
import time
from typing import Tuple, Optional
from contextlib import contextmanager
import tempfile

import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from plotly.offline import get_plotlyjs
import mne
from meg_qc.calculation.meg_qc_pipeline import resolve_analysis_root

# Get the absolute path of the parent directory of the current script
parent_dir = os.path.dirname(os.getcwd())
gradparent_dir = os.path.dirname(parent_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)
sys.path.append(gradparent_dir)

from meg_qc.calculation.objects import QC_derivative

# Configure plotting helpers at import time so worker processes spawned by
# joblib use the same plotting/reporting functions as the main process.


def _load_plotting_backend():
    """Expose the single supported plotting backend and report helpers."""

    # If the backend has been loaded already, do nothing.
    if 'make_joined_report_mne' in globals():
        return

    import meg_qc.plotting.universal_plots as _plots

    # Keep alias stable for modules importing ``meg_qc.plotting.universal_plots``.
    sys.modules['meg_qc.plotting.universal_plots'] = _plots
    globals().update(
        {
            name: getattr(_plots, name)
            for name in dir(_plots)
            if not name.startswith('_')
        }
    )

    from meg_qc.plotting.universal_html_report import (
        make_joined_report_mne,
        make_summary_qc_report,
    )

    globals().update(
        {
            'make_joined_report_mne': make_joined_report_mne,
            'make_summary_qc_report': make_summary_qc_report,
        }
    )


_load_plotting_backend()


def resolve_output_roots(
    dataset_path: str,
    external_derivatives_root: Optional[str],
) -> Tuple[str, str, str]:
    """Return output root plus read/write derivatives roots.

    ``dataset_derivatives_root`` always points to the derivatives under the
    input dataset. ``output_derivatives_root`` points to where new reports will
    be written (dataset root by default, external root when provided).
    """

    ds_name = os.path.basename(os.path.normpath(dataset_path))
    output_root = dataset_path if external_derivatives_root is None else os.path.join(external_derivatives_root, ds_name)
    dataset_derivatives_root = os.path.join(dataset_path, 'derivatives')
    output_derivatives_root = os.path.join(output_root, 'derivatives')
    os.makedirs(output_derivatives_root, exist_ok=True)
    return output_root, dataset_derivatives_root, output_derivatives_root


def build_overlay_dataset(dataset_path: str, derivatives_root: str):
    """Create a temporary overlay so ANCPBIDS sees external derivatives.

    ANCPBIDS expects the derivatives folder to live under the dataset root. When
    users direct outputs to an external path, we mirror the original dataset via
    symlinks into a temporary directory and drop a ``derivatives`` link that
    points to the external location. All symlinks stay outside the original
    dataset, so read-only datasets remain untouched.
    """

    overlay_tmp = tempfile.TemporaryDirectory(prefix='megqc_bids_overlay_')
    overlay_root = overlay_tmp.name

    for entry in os.listdir(dataset_path):
        if entry == 'derivatives':
            # Never point back to the original derivatives tree; we want the
            # external one to be used instead.
            continue

        src = os.path.join(dataset_path, entry)
        dst = os.path.join(overlay_root, entry)

        if not os.path.exists(dst):
            os.symlink(src, dst)

    os.symlink(derivatives_root, os.path.join(overlay_root, 'derivatives'))
    return overlay_tmp, overlay_root


@contextmanager
def temporary_dataset_base(dataset, base_dir: str):
    """Temporarily repoint the ANCPBIDS dataset to a new base directory."""

    original_base = getattr(dataset, 'base_dir_', None)
    dataset.base_dir_ = base_dir
    try:
        yield
    finally:
        dataset.base_dir_ = original_base


def _ensure_megqc_output_layout(
    output_derivatives_root: str,
    analysis_segments: Optional[List[str]] = None,
) -> Path:
    """Create minimal output folders/metadata for MEGqc reports.

    We keep ANCPBIDS for input discovery/querying but write consolidated subject
    reports as plain HTML files. This helper guarantees the target derivative
    layout exists (``derivatives/Meg_QC/reports``) and creates a lightweight
    ``dataset_description.json`` when missing.
    """

    analysis_segments = analysis_segments or []
    megqc_root = Path(output_derivatives_root) / "Meg_QC"
    for seg in analysis_segments:
        megqc_root = megqc_root / str(seg)
    reports_root = megqc_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    dataset_desc_path = megqc_root / "dataset_description.json"
    if not dataset_desc_path.exists():
        dataset_description = {
            "Name": "MEG QC Pipeline",
            "BIDSVersion": "1.0.1",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "MEG QC Pipeline",
                }
            ],
        }
        dataset_desc_path.write_text(
            json.dumps(dataset_description, indent=2),
            encoding="utf-8",
        )

    return reports_root

# IMPORTANT: keep this order of imports, first need to add parent dir to sys.path, then import from it.

# ____________________________

# How plotting in MEGqc works:
# During calculation save in the right folders the csvs with data for plotting
# During plotting step - read the csvs (find using ancpbids), plot them, save them as htmls in the right folders.


def create_categories_for_selector(entities: dict):

    """
    Create categories based on what metrics have already been calculated and detected as ancp bids as entities in MEGqc derivatives folder.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.

    Returns
    -------
    categories : dict
        A dictionary of entities and their subcategories with modified names
    """

    # Create a copy of entities
    categories = entities.copy()

    # Rename 'description' to 'METRIC' and sort the values
    categories = {
        ('METRIC' if k == 'description' else k): sorted(v, key=str)
        for k, v in categories.items()
    }

    #From METRIC remove whatever is not metric.
    #Cos METRIC is originally a desc entity which can contain just anything:

    if 'METRIC' in categories:
        valid_metrics = ['_ALL_METRICS_', 'STDs', 'PSDs', 'PtPsManual', 'PtPsAuto', 'ECGs', 'EOGs', 'Head', 'Muscle']
        categories['METRIC'] = [x for x in categories['METRIC'] if x.lower() in [metric.lower() for metric in valid_metrics]]

    #add '_ALL_' to the beginning of the list for each category:

    for category, subcategories in categories.items():
        categories[category] = ['_ALL_'+category+'s_'] + subcategories

    # Add 'm_or_g' category
    categories['m_or_g'] = ['_ALL_sensors', 'mag', 'grad']

    return categories


def selector(entities: dict):

    """
    Creates a in-terminal visual selector for the user to choose the entities and settings for plotting.

    Loop over categories (keys)
    for every key use a subfunction that will create a selector for the subcategories.

    Parameters
    ----------
    entities : dict
        A dictionary of entities and their subcategories.

    Returns
    -------
    selected_entities : dict
        A dictionary of selected entities.
    plot_settings : dict
        A dictionary of selected settings for plotting.

    """

    # SELECT ENTITIES and SETTINGS
    # Define the categories and subcategories
    categories = create_categories_for_selector(entities)

    selected = {}
    # Create a list of values with category titles
    for key, values in categories.items():
        results, quit_selector = select_subcategory(categories[key], key)

        print('___MEGqc___: select_subcategory: ', key, results)

        if quit_selector: # if user clicked cancel - stop:
            print('___MEGqc___: You clicked cancel. Please start over.')
            return None, None

        selected[key] = results


    # Separate into selected_entities and plot_settings
    selected_entities = {key: values for key, values in selected.items() if key != 'm_or_g'}
    plot_settings = {key: values for key, values in selected.items() if key == 'm_or_g'}

    return selected_entities, plot_settings


def select_subcategory(subcategories: List, category_title: str, window_title: str = "What would you like to plot? Click to select."):

    """
    Create a checkbox list dialog for the user to select subcategories.
    Example:
    sub: 009, 012, 013

    Parameters
    ----------
    subcategories : List
        A list of subcategories, such as: sub, ses, task, run, metric, mag/grad.
    category_title : str
        The title of the category.
    window_title : str
        The title of the checkbox list dialog, for visual.

    Returns
    -------
    results : List
        A list of selected subcategories.
    quit_selector : bool
        A boolean indicating whether the user clicked Cancel.

    """

    quit_selector = False

    # Create a list of values with category titles
    values = [(str(items), str(items)) for items in subcategories]

    while True:
        results = checkboxlist_dialog(
            title=window_title,
            text=category_title,
            values=values,
            style=Style.from_dict({
                'dialog': 'bg:#cdbbb3',
                'button': 'bg:#bf99a4',
                'checkbox': '#e8612c',
                'dialog.body': 'bg:#a9cfd0',
                'dialog shadow': 'bg:#c98982',
                'frame.label': '#fcaca3',
                'dialog.body label': '#fd8bb6',
            })
        ).run()

        # Set quit_selector to True if the user clicked Cancel (results is None)
        quit_selector = results is None

        if quit_selector or results:
            break
        else:
            print('___MEGqc___: Please select at least one subcategory or click Cancel.')


    # if '_ALL_' was chosen - choose all categories, except _ALL_ itself:
    if results: #if something was chosen
        for r in results:
            if '_ALL_' in r.upper():
                results = [str(category) for category in subcategories if '_ALL_' not in str(category).upper()]
                #Important! Keep ....if '_ALL_' not in str(category).upper() with underscores!
                #otherwise it will excude tasks like 'oddbALL' and such
                break

    return results, quit_selector


def get_ds_entities(dataset, calculated_derivs_folder: str, output_root: str):

    """
    Get the entities of the dataset using ancpbids, only get derivative entities, not all raw data.

    Parameters
    ----------
    dataset : ancpbids object
        The dataset object.
    calculated_derivs_folder : str
        The path to the calculated derivatives folder.
    output_root : str
        Base directory where derivatives are stored (may differ from the
        original BIDS dataset when users provide an external location).

    Returns
    -------
    entities : dict
        A dictionary of entities and their subcategories.

    """

    def _safe_query_entities():
        """Query entities while tolerating empty results/Windows ``None`` returns."""

        try:
            return dataset.query_entities(scope=calculated_derivs_folder) or {}
        except TypeError:
            # ``query_entities`` can raise ``TypeError`` when ``query`` returns
            # ``None`` (e.g., when a derivatives folder does not exist yet on
            # some platforms). Treat that situation as an empty mapping so we
            # can try fallbacks before failing.
            return {}

    with temporary_dataset_base(dataset, output_root):
        entities = _safe_query_entities()

    if not entities:
        raise FileNotFoundError(f'___MEGqc___: No calculated derivatives found for this ds!')

    print('___MEGqc___: ', 'Entities found in the dataset: ', entities)
    # we only get entities of calculated derivatives here, not entire raw ds.

    return entities


#
# -------------------------- Subject HTML report helpers --------------------------
#
# The plotting backend computes QC_derivative objects per metric from machine-
# readable derivatives (TSV/JSON/FIF). The helpers below reorganize those objects
# into one subject report with:
#   - Top-level metric tabs
#   - Run/task subtabs inside each metric
#   - Plot-type subtabs inside each run
# and lazy Plotly rendering (inline JSON store, no external sidecar files).
#

METRIC_ORDER = [
    "STDs",
    "PtPsManual",
    "PtPsAuto",
    "PSDs",
    "ECGs",
    "EOGs",
    "Muscle",
    "Head",
    "stimulus",
]

METRIC_LABELS = {
    "STDs": "STD",
    "PtPsManual": "PtP (manual)",
    "PtPsAuto": "PtP (auto)",
    "PSDs": "PSD",
    "ECGs": "ECG",
    "EOGs": "EOG",
    "Muscle": "Muscle",
    "Head": "Head",
    "stimulus": "Stimulus",
}


def _metric_to_report_section(metric: str) -> Optional[str]:
    """Map a metric descriptor to its report section key.

    Notes
    -----
    The original report code routes both manual and auto PtP through the same
    section bucket (``PTP_MANUAL``). This mapping keeps that behavior so that
    the visual outputs remain consistent with the existing plotting pipeline.
    """
    metric_upper = str(metric).upper()
    if "STD" in metric_upper:
        return "STD"
    if "PTP" in metric_upper:
        return "PTP_MANUAL"
    if "PSD" in metric_upper:
        return "PSD"
    if "ECG" in metric_upper:
        return "ECG"
    if "EOG" in metric_upper:
        return "EOG"
    if "MUSCLE" in metric_upper:
        return "MUSCLE"
    if "HEAD" in metric_upper:
        return "HEAD"
    if "STIM" in metric_upper:
        return "STIMULUS"
    return None


def _sanitize_token(value: str) -> str:
    """Create a safe token for HTML ids."""
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip())
    return token.strip("-") or "item"


def _human_metric_label(metric: str) -> str:
    """Return user-facing metric label for tabs."""
    return METRIC_LABELS.get(metric, metric)


def _human_run_label(raw_entity_name: str) -> str:
    """Return compact run label used in subtabs.

    ``raw_entity_name`` already contains BIDS entities without ``desc`` and is
    therefore an ideal stable key for a run/task panel.
    """
    if not raw_entity_name:
        return "run"
    return str(raw_entity_name)


def _extract_bids_entity(raw_entity_name: str, entity: str) -> Optional[str]:
    """Extract one BIDS entity value from a run identifier string."""
    match = re.search(rf"(?:^|_){re.escape(entity)}-([^_]+)", str(raw_entity_name))
    return match.group(1) if match else None


def _build_run_tab_labels(raw_entity_names: Sequence[str]) -> Dict[str, str]:
    """Create concise, task-first labels for run tabs.

    Preference:
    1. Task name only when unique.
    2. Task name with run/session suffix when duplicated.
    3. Fallback to original run identifier when task is missing.
    """
    task_counts: Dict[str, int] = defaultdict(int)
    parsed: Dict[str, Dict[str, Optional[str]]] = {}
    for raw in raw_entity_names:
        task = _extract_bids_entity(raw, "task")
        parsed[str(raw)] = {
            "task": task,
            "run": _extract_bids_entity(raw, "run"),
            "ses": _extract_bids_entity(raw, "ses"),
            "acq": _extract_bids_entity(raw, "acq"),
            "rec": _extract_bids_entity(raw, "rec"),
        }
        if task:
            task_counts[task] += 1

    out: Dict[str, str] = {}
    for raw in raw_entity_names:
        key = str(raw)
        task = parsed[key]["task"]
        if not task:
            out[key] = _human_run_label(key)
            continue
        if task_counts[task] <= 1:
            out[key] = task
            continue

        suffix_parts = []
        for ent in ("run", "ses", "acq", "rec"):
            val = parsed[key][ent]
            if val:
                suffix_parts.append(f"{ent}-{val}")
        suffix = ", ".join(suffix_parts) if suffix_parts else key
        out[key] = f"{task} ({suffix})"
    return out


def _format_metric_note_html(metric_note: str) -> str:
    """Render metric notes with readable line breaks."""
    text = str(metric_note or "").strip()
    if not text:
        return ""
    safe = html.escape(text)
    safe = re.sub(r"&lt;br\s*/?&gt;", "<br>", safe, flags=re.IGNORECASE)
    safe = re.sub(r"&lt;p&gt;", "<br>", safe, flags=re.IGNORECASE)
    safe = re.sub(r"&lt;/p&gt;", "", safe, flags=re.IGNORECASE)
    safe = safe.replace("\n", "<br>")
    return safe


def _safe_load_raw_info_html(raw_info_path: Optional[str]) -> str:
    """Return rendered MNE info HTML from a RawInfo FIF path."""
    if not raw_info_path:
        return ""
    try:
        info = mne.io.read_info(raw_info_path)
        return str(info._repr_html_() or "")
    except Exception:
        return ""


def _human_derivative_tab_label(metric_key: str, raw_name: str) -> str:
    """Map internal derivative names to concise plot labels."""
    name = str(raw_name or "").strip()
    name_l = name.lower()
    metric_l = str(metric_key or "").lower()

    if name_l == "sensors_positions":
        return "Channel layout (3D)"

    if "std" in metric_l:
        if "per_channel" in name_l:
            return "Channel x epoch heatmap"
        if "topomap" in name_l:
            return "Channel-wise STD topomap (3D)"
        if "all_data" in name_l:
            return "Channel-wise STD distribution"
    if "ptp" in metric_l:
        if "per_channel" in name_l:
            return "Channel x epoch heatmap"
        if "topomap" in name_l:
            return "Channel-wise PtP topomap (3D)"
        if "all_data" in name_l:
            return "Channel-wise PtP distribution"
    if "psd" in metric_l:
        if "topomap" in name_l:
            return "Channel-wise PSD topomap (3D)"
        if "all_data" in name_l:
            return "PSD curves by channel"
        if "noise" in name_l:
            return "Relative power (noise frequencies)"
        if "waves" in name_l:
            return "Relative power (canonical bands)"
    if "ecg" in metric_l:
        if "topomap" in name_l:
            return "Channel-wise ECG topomap (3D)"
        if "mean_ch_data" in name_l:
            return "ECG channel waveform"
        if "most_affected" in name_l:
            return "Highest correlation-magnitude channels"
        if "middle_affected" in name_l:
            return "Middle correlation-magnitude channels"
        if "least_affected" in name_l:
            return "Lowest correlation-magnitude channels"
    if "eog" in metric_l:
        if "topomap" in name_l:
            return "Channel-wise EOG topomap (3D)"
        if "mean_ch_data" in name_l:
            return "EOG channel waveform"
        if "most_affected" in name_l:
            return "Highest correlation-magnitude channels"
        if "middle_affected" in name_l:
            return "Middle correlation-magnitude channels"
        if "least_affected" in name_l:
            return "Lowest correlation-magnitude channels"

    # Generic readable fallback.
    cleaned = re.sub(r"[_\\-]+", " ", name).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("Magnetometers", "").replace("Gradiometers", "").strip()
    return cleaned or name or "Plot"


def _safe_load_json(path: Optional[str]) -> Optional[Any]:
    """Load JSON content safely, returning ``None`` on any failure."""
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_run_sensor_derivatives(derivs_for_this_raw: Sequence["Deriv_to_plot"]) -> Tuple[List[QC_derivative], List[str]]:
    """Create one sensor-position figure set for a run.

    We intentionally generate these figures only once per run and place them in
    the Overview tab to avoid repeating the same geometry under every metric.
    """
    priority_metrics = ("STDs", "PtPsManual", "PtPsAuto", "PSDs", "ECGs", "EOGs")
    for metric in priority_metrics:
        paths = [d.path for d in derivs_for_this_raw if d.metric == metric]
        for path in paths:
            try:
                figs = plot_sensors_3d_csv(path)
            except Exception:
                figs = []
            if figs:
                return figs, [path]
    return [], []


# Lazy Plotly storage kept in-memory per report build.
# We keep one serialized payload per figure and parse it on-demand in the
# browser when that placeholder becomes visible. This avoids one giant
# ``JSON.parse`` over hundreds of MB.
_LAZY_PLOT_PAYLOADS: Dict[str, str] = {}
_LAZY_PLOT_COUNTER = count(1)
_LAZY_PAYLOAD_COUNTER = count(1)
_PLOTLY_JS_BUNDLE: Optional[str] = None


def _reset_lazy_figure_store() -> None:
    """Clear lazy figure registry before creating a new subject report."""
    global _LAZY_PLOT_COUNTER, _LAZY_PAYLOAD_COUNTER
    _LAZY_PLOT_PAYLOADS.clear()
    _LAZY_PLOT_COUNTER = count(1)
    _LAZY_PAYLOAD_COUNTER = count(1)


def _inline_plotly_bundle_script() -> str:
    """Return an inline Plotly JS bundle script tag.

    Subject reports must stay fully standalone/offline. Embedding Plotly avoids
    reliance on a CDN, which otherwise results in empty placeholders when the
    browser cannot fetch external scripts.
    """

    global _PLOTLY_JS_BUNDLE
    if _PLOTLY_JS_BUNDLE is None:
        # Escape closing tags defensively so the HTML parser keeps the bundle
        # inside this script element.
        _PLOTLY_JS_BUNDLE = get_plotlyjs().replace("</", "<\\/")
    return f"<script>{_PLOTLY_JS_BUNDLE}</script>"


def _lazy_payload_script_tags_html() -> str:
    """Return inline JSON script tags for lazily-rendered Plotly payloads."""
    return "".join(
        f"<script id='{payload_id}' type='application/json'>{payload_json}</script>"
        for payload_id, payload_json in _LAZY_PLOT_PAYLOADS.items()
    )


def _has_xy_or_scene_axes(fig: go.Figure) -> bool:
    """Return True when figure layout exposes cartesian or 3D scene axes."""
    layout_dict = fig.to_plotly_json().get("layout", {})
    for key in layout_dict.keys():
        k = str(key)
        if k.startswith("xaxis") or k.startswith("yaxis") or re.fullmatch(r"scene\d*", k):
            return True
    return False


def _attach_distribution_style_controls(
    fig: go.Figure,
    *,
    default_line_width: float = 2.0,
    default_marker_size: float = 8.0,
) -> None:
    """Attach numeric line/dot style controls when relevant traces are present."""
    if fig is None or not getattr(fig, "data", None):
        return

    marker_level_to_size = {1: 3.0, 2: 4.0, 3: 5.5, 4: 7.0, 5: 8.5, 6: 10.0, 7: 12.0, 8: 14.0}

    has_line_traces = False
    has_marker_traces = False
    for tr in fig.data:
        t = str(getattr(tr, "type", "") or "").lower()
        mode = str(getattr(tr, "mode", "") or "").lower()
        if t in {"scatter", "scattergl"} and "lines" in mode:
            has_line_traces = True
        elif t in {"violin", "box", "histogram", "bar"}:
            has_line_traces = True
        if t in {"scatter", "scattergl"} and "markers" in mode:
            has_marker_traces = True

    if (not has_line_traces) and (not has_marker_traces):
        return

    def _line_restyle(level: int) -> Dict[str, List[Optional[float]]]:
        line_width: List[Optional[float]] = []
        marker_line_width: List[Optional[float]] = []
        lw = float(level)
        for tr in fig.data:
            t = str(getattr(tr, "type", "") or "").lower()
            mode = str(getattr(tr, "mode", "") or "").lower()
            if (t in {"scatter", "scattergl"} and "lines" in mode) or (t in {"violin", "box", "histogram", "bar"}):
                line_width.append(lw)
            else:
                line_width.append(None)

            if t == "histogram":
                marker_line_width.append(max(0.5, lw * 0.6))
            elif t in {"scatter", "scattergl"} and "markers" in mode:
                marker_line_width.append(max(0.25, lw * 0.2))
            else:
                marker_line_width.append(None)
        return {"line.width": line_width, "marker.line.width": marker_line_width}

    def _marker_restyle(level: int) -> Dict[str, List[Optional[float]]]:
        size = float(marker_level_to_size.get(level, default_marker_size))
        marker_size: List[Optional[float]] = []
        for tr in fig.data:
            t = str(getattr(tr, "type", "") or "").lower()
            mode = str(getattr(tr, "mode", "") or "").lower()
            if t in {"scatter", "scattergl"} and "markers" in mode:
                marker_size.append(size)
            else:
                marker_size.append(None)
        return {"marker.size": marker_size}

    levels = list(range(1, 9))
    line_active = max(0, min(7, int(round(default_line_width)) - 1))
    marker_active_guess = min(levels, key=lambda lv: abs(float(marker_level_to_size[lv]) - float(default_marker_size)))
    marker_active = max(0, min(7, int(marker_active_guess) - 1))

    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    menu_names = {str(getattr(m, "name", "") or "") for m in existing}

    if has_line_traces and ("Line thickness level" not in menu_names):
        existing.append(
            dict(
                name="Line thickness level",
                type="buttons",
                direction="right",
                showactive=True,
                active=line_active,
                buttons=[dict(label=str(level), method="restyle", args=[_line_restyle(level)]) for level in levels],
            )
        )
    if has_marker_traces and ("Dot size level" not in menu_names):
        existing.append(
            dict(
                name="Dot size level",
                type="buttons",
                direction="right",
                showactive=True,
                active=marker_active,
                buttons=[dict(label=str(level), method="restyle", args=[_marker_restyle(level)]) for level in levels],
            )
        )

    if existing:
        fig.update_layout(updatemenus=existing)


def _attach_axis_label_size_controller(fig: go.Figure) -> None:
    """Attach numeric axis-label/ticks-size controller when axes are available."""
    if fig is None:
        return
    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    if any(str(getattr(m, "name", "") or "") == "Axis label/ticks size level" for m in existing):
        return

    layout_dict = fig.to_plotly_json().get("layout", {})
    axis_keys = [k for k in layout_dict.keys() if str(k).startswith("xaxis") or str(k).startswith("yaxis")]
    scene_keys = [k for k in layout_dict.keys() if re.fullmatch(r"scene\d*", str(k) or "")]
    if not axis_keys:
        cartesian_types = {"scatter", "scattergl", "bar", "histogram", "box", "violin", "heatmap", "contour"}
        has_cartesian = any(str(getattr(tr, "type", "") or "").lower() in cartesian_types for tr in fig.data)
        if has_cartesian:
            axis_keys = ["xaxis", "yaxis"]
    if (not axis_keys) and (not scene_keys):
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
    """Infer a readable external control title from updatemenu semantics."""
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
    return f"Figure control {idx + 1}"


def _extract_figure_controls(fig: go.Figure) -> List[Dict[str, object]]:
    """Extract Plotly updatemenus as external HTML controls and clear layout menus."""
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

    # Explicit assignment is more reliable than update_layout(updatemenus=[]).
    fig.layout.updatemenus = None
    return controls


def _register_lazy_plotly_figure(fig: go.Figure, *, min_height_px: int = 520) -> str:
    """Register one Plotly figure and return a placeholder ``div``.

    The browser renders placeholders only when their tab is visible. This keeps
    initial page load responsive even when a subject has many heavy figures.
    """
    fig_id = f"lazy-plot-{next(_LAZY_PLOT_COUNTER)}"
    payload_id = f"lazy-payload-{next(_LAZY_PAYLOAD_COUNTER)}"
    fig_out = go.Figure(fig)

    # Reserve extra top margin and title padding to prevent overlaps with the
    # plotting area when many traces/annotations are present.
    title = getattr(fig_out.layout, "title", None)
    if title is not None and getattr(title, "text", None):
        title_json = title.to_plotly_json() if hasattr(title, "to_plotly_json") else dict(title)
        title_json.setdefault("y", 0.99)
        title_json.setdefault("yanchor", "top")
        pad = dict(title_json.get("pad", {}))
        pad["b"] = max(int(pad.get("b", 0)), 30)
        pad["t"] = max(int(pad.get("t", 0)), 8)
        title_json["pad"] = pad
        fig_out.update_layout(title=title_json)
        margin = fig_out.layout.margin.to_plotly_json() if fig_out.layout.margin else {}
        margin["t"] = max(int(margin.get("t", 80)), 132)
        fig_out.update_layout(margin=margin)

    # Attach figure-level style controllers where meaningful, then externalize
    # all updatemenus into a dedicated panel below the plot.
    _attach_distribution_style_controls(fig_out)
    _attach_axis_label_size_controller(fig_out)
    controls = _extract_figure_controls(fig_out)

    height = fig_out.layout.height
    if not isinstance(height, (int, float)):
        height = 640
    height = max(int(height), int(min_height_px))

    payload_json = json.dumps(
        {
            "figure": fig_out.to_plotly_json(),
            "config": {"responsive": True, "displaylogo": False},
            "controls": controls,
        },
        cls=PlotlyJSONEncoder,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    _LAZY_PLOT_PAYLOADS[payload_id] = payload_json
    return (
        "<div class='lazy-plot-wrap'>"
        f"<div id='{fig_id}' class='js-lazy-plot' data-payload-id='{payload_id}' "
        f"style='height:{height}px; width:100%;'></div>"
        "<div class='plot-controls'></div>"
        "</div>"
    )


def _build_lazy_summary_iframe(summary_html: str, *, frame_id: str) -> str:
    """Create a lazily-populated iframe for one formatted summary block.

    ``srcdoc`` payload is stored in ``data-srcdoc`` and only attached when the
    parent tab is activated. This avoids the browser parsing all nested summary
    reports during initial load.
    """
    escaped_srcdoc = html.escape(summary_html, quote=True)
    return (
        f"<iframe id='{frame_id}' class='summary-iframe js-summary-iframe' "
        f"data-srcdoc=\"{escaped_srcdoc}\" title='Run summary' loading='lazy'></iframe>"
    )


def _plot_block_from_derivative(deriv: QC_derivative, *, display_title: Optional[str] = None) -> str:
    """Render one derivative into an HTML block.

    Plotly derivatives are registered as lazy placeholders. Matplotlib
    derivatives are converted immediately to ``img`` HTML.
    """
    title = html.escape(str(display_title or getattr(deriv, "name", "plot")))
    description = html.escape(str(getattr(deriv, "description_for_user", "") or ""))
    if deriv.content_type == "plotly":
        try:
            fig = go.Figure(deriv.content)
            content_html = _register_lazy_plotly_figure(fig)
        except Exception as exc:
            content_html = f"<p>Failed to render Plotly figure: {html.escape(str(exc))}</p>"
    elif deriv.content_type == "matplotlib":
        try:
            content_html = deriv.convert_fig_to_html() or "<p>Matplotlib figure is empty.</p>"
        except Exception as exc:
            content_html = f"<p>Failed to render Matplotlib figure: {html.escape(str(exc))}</p>"
    else:
        content_html = "<p>This derivative has no supported figure content.</p>"

    return (
        "<div class='figure-card'>"
        f"<h4>{title}</h4>"
        f"{content_html}"
        f"<p class='figure-note'>{description}</p>"
        "</div>"
    )


def _infer_derivative_channel_type(deriv: QC_derivative) -> str:
    """Infer channel type bucket from derivative metadata.

    Returns one of ``MAG``, ``GRAD``, or ``ALL``. The fallback ``ALL`` keeps
    figures that are not channel-type specific (for example overview traces).
    """
    name = str(getattr(deriv, "name", "") or "")
    desc = str(getattr(deriv, "description_for_user", "") or "")
    blob = f"{name} {desc}".lower()

    has_mag = ("magnetometer" in blob) or re.search(r"(^|[^a-z])mag([^a-z]|$)", blob) is not None
    has_grad = ("gradiometer" in blob) or re.search(r"(^|[^a-z])grad([^a-z]|$)", blob) is not None

    if has_mag and not has_grad:
        return "MAG"
    if has_grad and not has_mag:
        return "GRAD"
    return "ALL"


def _build_derivative_plot_tabs(
    *,
    group_id: str,
    metric_key: str,
    derivatives: Sequence[QC_derivative],
    level: int,
) -> str:
    """Build the innermost plot tabs from derivative objects."""
    fig_tabs: List[Tuple[str, str]] = []
    used_labels: Dict[str, int] = defaultdict(int)
    for idx, deriv in enumerate(derivatives, start=1):
        base = str(getattr(deriv, "name", "") or f"Plot {idx}")
        clean_base = _human_derivative_tab_label(metric_key, base)
        used_labels[clean_base] += 1
        label = clean_base if used_labels[clean_base] == 1 else f"{clean_base} ({used_labels[clean_base]})"
        fig_tabs.append((label, _plot_block_from_derivative(deriv, display_title=label)))
    return _build_subtabs_html(group_id=group_id, tabs=fig_tabs, level=level)


def _build_subtabs_html(group_id: str, tabs: Sequence[Tuple[str, str]], *, level: int = 1) -> str:
    """Render one reusable tab group with explicit hierarchy depth."""
    if not tabs:
        return "<p>No content available.</p>"

    buttons = []
    panels = []
    for idx, (label, body_html) in enumerate(tabs):
        panel_id = f"{group_id}-panel-{idx}"
        active = " active" if idx == 0 else ""
        buttons.append(
            f"<button class='subtab-btn{active}' data-tab-group='{group_id}' "
            f"data-target='{panel_id}'>{html.escape(label)}</button>"
        )
        panels.append(
            f"<div id='{panel_id}' class='subtab-content{active}' data-tab-group='{group_id}'>{body_html}</div>"
        )
    lvl = max(1, int(level))
    return (
        f"<div class='subtab-group level-{lvl}'>"
        f"<div class='subtab-row'>{''.join(buttons)}</div>"
        f"{''.join(panels)}"
        "</div>"
    )


def _build_metric_run_panel(
    *,
    metric_key: str,
    run_label: str,
    derivatives: Sequence[QC_derivative],
    metric_note: str,
    source_paths: Sequence[str],
) -> str:
    """Build one run-specific panel inside a metric tab."""
    blocks = []
    if metric_note:
        blocks.append(f"<p><strong>Metric notes:</strong> {_format_metric_note_html(metric_note)}</p>")
    if source_paths:
        src_items = "".join(
            f"<li><code>{html.escape(str(p))}</code></li>"
            for p in sorted(set(str(p) for p in source_paths))
        )
        blocks.append(f"<details><summary>Machine-readable inputs</summary><ul>{src_items}</ul></details>")

    if derivatives:
        ch_groups: Dict[str, List[QC_derivative]] = defaultdict(list)
        for deriv in derivatives:
            ch_groups[_infer_derivative_channel_type(deriv)].append(deriv)

        # Channel-type organization:
        # - MAG tab when magnetometer-specific figures exist
        # - GRAD tab when gradiometer-specific figures exist
        # - General tab for non-specific figures
        ch_tabs: List[Tuple[str, str]] = []
        order = [("MAG", "MAG"), ("GRAD", "GRAD"), ("ALL", "General")]
        for key, label in order:
            items = ch_groups.get(key, [])
            if not items:
                continue
            ch_tabs.append(
                (
                    label,
                    _build_derivative_plot_tabs(
                        group_id=f"figs-{_sanitize_token(metric_key)}-{_sanitize_token(run_label)}-{key.lower()}",
                        metric_key=metric_key,
                        derivatives=items,
                        level=4,
                    ),
                )
            )
        if len(ch_tabs) > 1:
            blocks.append(
                _build_subtabs_html(
                    group_id=f"chtype-{_sanitize_token(metric_key)}-{_sanitize_token(run_label)}",
                    tabs=ch_tabs,
                    level=3,
                )
            )
        else:
            blocks.append(ch_tabs[0][1])
    else:
        blocks.append("<p>No figures were generated for this metric/run combination.</p>")

    return (
        "<div class='run-panel'>"
        f"<h3>{html.escape(_human_metric_label(metric_key))} - {html.escape(run_label)}</h3>"
        f"{''.join(blocks)}"
        "</div>"
    )


def _build_subject_overview_section(
    *,
    subject: str,
    dataset_name: str,
    metrics_payload: Dict[str, List[Dict[str, Any]]],
    overview_payload: List[Dict[str, Any]],
    summary_payload: List[Dict[str, Any]],
) -> str:
    """Create overview tab with compact run/metric availability."""
    metric_keys = [m for m in METRIC_ORDER if metrics_payload.get(m)]
    run_labels = sorted(
        {
            entry["run_label"]
            for entries in metrics_payload.values()
            for entry in entries
        }
    )

    rows = []
    for run_label in run_labels:
        availability = []
        for metric in metric_keys:
            is_available = any(e["run_label"] == run_label for e in metrics_payload.get(metric, []))
            availability.append("Yes" if is_available else "No")
        cols = "".join(f"<td>{val}</td>" for val in availability)
        rows.append(f"<tr><td><code>{html.escape(run_label)}</code></td>{cols}</tr>")

    header_cols = "".join(f"<th>{html.escape(_human_metric_label(m))}</th>" for m in metric_keys)
    empty_row_html = "<tr><td colspan='99'>No runs found.</td></tr>"
    table_html = (
        "<table>"
        "<thead><tr><th>Run / task</th>"
        f"{header_cols}</tr></thead>"
        f"<tbody>{''.join(rows) if rows else empty_row_html}</tbody>"
        "</table>"
    )

    n_runs = len({item["run_label"] for item in summary_payload})
    n_metrics = len(metric_keys)
    overview_html = (
        "<div class='overview-grid'>"
        "<div class='overview-card'>"
        "<h3>Subject summary</h3>"
        f"<p><strong>Dataset:</strong> {html.escape(dataset_name)}</p>"
        f"<p><strong>Subject:</strong> sub-{html.escape(subject)}</p>"
        f"<p><strong>N runs:</strong> {n_runs}</p>"
        f"<p><strong>N metrics with figures:</strong> {n_metrics}</p>"
        "</div>"
        "<div class='overview-card'>"
        "<h3>Run x metric availability</h3>"
        f"{table_html}"
        "</div>"
        "</div>"
    )

    sensor_tabs: List[Tuple[str, str]] = []
    for item in overview_payload:
        derivs = item.get("sensor_derivatives", []) or []
        if not derivs:
            continue
        run_label = str(item.get("run_label", "run"))
        source_paths = item.get("source_paths", []) or []
        src_html = ""
        if source_paths:
            src_items = "".join(
                f"<li><code>{html.escape(str(p))}</code></li>"
                for p in sorted(set(str(p) for p in source_paths))
            )
            src_html = f"<details><summary>Sources</summary><ul>{src_items}</ul></details>"
        sensor_blocks = "".join(_plot_block_from_derivative(d) for d in derivs)
        sensor_tabs.append((run_label, src_html + sensor_blocks))

    if sensor_tabs:
        overview_html += (
            "<h3>Sensor positions (3D, one panel per run)</h3>"
            "<p>Sensor geometry is shown once here to avoid repeating the same view under every metric tab.</p>"
            + _build_subtabs_html("overview-sensors-runs", sensor_tabs, level=2)
        )

    header_tabs: List[Tuple[str, str]] = []
    for item in summary_payload:
        run_label = str(item.get("run_label", "run"))
        raw_info_html = str(item.get("raw_info_html", "") or "")
        if not raw_info_html:
            continue
        raw_info_path = item.get("raw_info_path")
        src_html = ""
        if raw_info_path:
            src_html = (
                "<details><summary>Raw info source</summary>"
                f"<ul><li><code>{html.escape(str(raw_info_path))}</code></li></ul></details>"
            )
        panel_html = src_html + f"<div class='raw-info-wrap'>{raw_info_html}</div>"
        header_tabs.append((run_label, panel_html))

    if header_tabs:
        overview_html += (
            "<h3>Recording header information</h3>"
            "<p>Per-run MNE header metadata from the original raw info file.</p>"
            + _build_subtabs_html("overview-header-runs", header_tabs, level=2)
        )

    return overview_html


def _build_subject_summary_section(
    *,
    subject: str,
    summary_payload: List[Dict[str, Any]],
) -> str:
    """Build metric-oriented QC summary tabs for one subject.

    Layout:
    - Top-level tabs by metric (GQI, PSD, ECG, EOG, ...)
    - For each metric, nested tabs by run/task
    """
    run_payloads: List[Dict[str, Any]] = []
    for item in summary_payload:
        report_strings_path = item.get("report_strings_path")
        simple_metrics_path = item.get("simple_metrics_path")
        run_payloads.append(
            {
                "run_label": str(item.get("run_label", "run")),
                "report_strings_path": report_strings_path,
                "simple_metrics_path": simple_metrics_path,
                "report_strings": _safe_read_json(report_strings_path),
                "simple_metrics": _safe_read_json(simple_metrics_path),
                "source_paths": item.get("source_paths", []) or [],
            }
        )

    gqi_rows = _collect_subject_gqi_task_rows(subject=subject, summary_payload=summary_payload)
    metric_tabs: List[Tuple[str, str]] = [
        ("GQI", _build_subject_gqi_metric_panel(gqi_rows))
    ]

    metric_order = [
        "PSD",
        "ECG",
        "EOG",
        "STD",
        "PTP_MANUAL",
        "PTP_AUTO",
        "MUSCLE",
        "HEAD",
        "STIMULUS",
        "INITIAL_INFO",
    ]
    metric_labels = {
        "PTP_MANUAL": "PtP (manual)",
        "PTP_AUTO": "PtP (auto)",
    }

    for metric in metric_order:
        run_tabs: List[Tuple[str, str]] = []
        for run_item in run_payloads:
            run_label = run_item["run_label"]
            report_strings = run_item.get("report_strings", {}) or {}
            simple_metrics = run_item.get("simple_metrics", {}) or {}
            report_text = str(report_strings.get(metric, "") or "")
            metric_data = simple_metrics.get(metric)

            panel_html = _build_subject_qc_metric_run_panel(
                metric=metric,
                run_label=run_label,
                report_text=report_text,
                metric_data=metric_data,
                gqi_rows=gqi_rows,
                source_paths=run_item.get("source_paths", []),
                report_strings_path=run_item.get("report_strings_path"),
                simple_metrics_path=run_item.get("simple_metrics_path"),
            )
            if panel_html:
                run_tabs.append((run_label, panel_html))

        if run_tabs:
            metric_tabs.append(
                (
                    metric_labels.get(metric, metric),
                    _build_subtabs_html(
                        group_id=f"subject-qc-metric-{_sanitize_token(metric)}",
                        tabs=run_tabs,
                        level=3,
                    ),
                )
            )

    if not metric_tabs:
        return "<p>No QC summary material is available for this subject.</p>"

    return _build_subtabs_html("subject-qc-summary-metrics", metric_tabs, level=2)


def _resolve_analysis_root_from_artifact(artifact_path: str) -> Optional[Path]:
    """Resolve the MEGqc analysis root from one derivative artifact path.

    Supports both layouts:
    - legacy: ``.../derivatives/Meg_QC/...``
    - profile: ``.../derivatives/Meg_QC/profiles/<analysis_id>/...``
    """
    if not artifact_path:
        return None
    p = Path(artifact_path)
    parts = p.parts
    if "Meg_QC" not in parts:
        return None

    idx = max(i for i, seg in enumerate(parts) if seg == "Meg_QC")
    root = Path(*parts[: idx + 1])
    if idx + 2 < len(parts) and parts[idx + 1] == "profiles":
        return root / "profiles" / parts[idx + 2]
    return root


def _collect_subject_gqi_task_rows(
    *,
    subject: str,
    summary_payload: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collect subject-level GQI rows (task x attempt) from attempt TSV files."""
    subject_tokens = {str(subject), f"sub-{subject}"}
    analysis_roots: List[Path] = []
    for item in summary_payload:
        report_strings_path = item.get("report_strings_path")
        simple_metrics_path = item.get("simple_metrics_path")
        for p in [report_strings_path, simple_metrics_path]:
            root = _resolve_analysis_root_from_artifact(str(p) if p else "")
            if root and root not in analysis_roots:
                analysis_roots.append(root)

    rows: List[Dict[str, Any]] = []
    for root in analysis_roots:
        group_metrics_dir = root / "summary_reports" / "group_metrics"
        if not group_metrics_dir.exists():
            continue

        for tsv_path in sorted(group_metrics_dir.glob("Global_Quality_Index_attempt_*.tsv")):
            match = re.search(r"attempt_(\d+)\.tsv$", tsv_path.name)
            if not match:
                continue
            attempt = int(match.group(1))

            try:
                with tsv_path.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for record in reader:
                        subj_val = str(record.get("subject", "")).strip()
                        if subj_val not in subject_tokens:
                            continue
                        rows.append(
                            {
                                "attempt": attempt,
                                "task": str(record.get("task", "")).strip(),
                                "GQI": str(record.get("GQI", "")).strip(),
                                "ECG_mag_high_corr_num": str(record.get("ECG_mag_high_corr_num", "")).strip(),
                                "ECG_mag_high_corr_percentage": str(record.get("ECG_mag_high_corr_percentage", "")).strip(),
                                "ECG_grad_high_corr_num": str(record.get("ECG_grad_high_corr_num", "")).strip(),
                                "ECG_grad_high_corr_percentage": str(record.get("ECG_grad_high_corr_percentage", "")).strip(),
                                "EOG_mag_high_corr_num": str(record.get("EOG_mag_high_corr_num", "")).strip(),
                                "EOG_mag_high_corr_percentage": str(record.get("EOG_mag_high_corr_percentage", "")).strip(),
                                "EOG_grad_high_corr_num": str(record.get("EOG_grad_high_corr_num", "")).strip(),
                                "EOG_grad_high_corr_percentage": str(record.get("EOG_grad_high_corr_percentage", "")).strip(),
                                "tsv_path": str(tsv_path),
                            }
                        )
            except Exception:
                continue

    rows.sort(key=lambda x: (x.get("attempt", 0), x.get("task", "")))
    return rows


def _build_subject_gqi_metric_panel(rows: List[Dict[str, Any]]) -> str:
    """Render GQI metric panel with clean task-level rows."""
    if not rows:
        return (
            "<div class='overview-card'>"
            "<h3>Global Quality Index (GQI)</h3>"
            "<p>No GQI attempt tables were found, or no rows matched this subject.</p>"
            "</div>"
        )

    body_rows = []
    source_paths = []
    for row in rows:
        source_path = str(row.get("tsv_path", "") or "")
        source_name = os.path.basename(source_path) if source_path else "-"
        if source_path:
            source_paths.append(source_path)
        body_rows.append(
            "<tr>"
            f"<td>{row['attempt']}</td>"
            f"<td>{html.escape(row.get('task', '') or '-')}</td>"
            f"<td>{row.get('GQI', '') or '-'}</td>"
            f"<td><code>{html.escape(source_name)}</code></td>"
            "</tr>"
        )

    table_html = (
        "<table>"
        "<thead><tr>"
        "<th>Attempt</th><th>Task</th><th>GQI</th><th>Source TSV</th>"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    sources_details = ""
    unique_sources = sorted(set(source_paths))
    if unique_sources:
        src_items = "".join(
            f"<li><code>{html.escape(p)}</code></li>" for p in unique_sources
        )
        sources_details = (
            "<details><summary>Source file paths</summary>"
            f"<ul>{src_items}</ul></details>"
        )
    return (
        "<div class='overview-card'>"
        "<h3>Global Quality Index (GQI)</h3>"
        "<p>Task-level rows discovered from "
        "<code>summary_reports/group_metrics/Global_Quality_Index_attempt_&lt;n&gt;.tsv</code>. "
        "Values shown are filtered for this subject.</p>"
        f"{table_html}"
        f"{sources_details}"
        "</div>"
    )


def _safe_read_json(path: Optional[str]) -> Dict[str, Any]:
    """Read JSON file with a tolerant fallback."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _normalize_task_token(task: str) -> str:
    """Normalize task/run labels so run tabs match GQI task rows."""
    token = str(task or "").strip().lower()
    token = token.replace("task-", "").replace("run-", "")
    token = re.sub(r"[_\\s]+", "", token)
    return token


def _html_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a compact HTML table."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return (
        "<table>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table>"
    )


def _build_psd_summary_from_simple_metrics(metric_data: Any) -> str:
    """Build concise PSD summary from SimpleMetrics JSON."""
    if not isinstance(metric_data, dict):
        return "<p>No PSD simple-metrics block is available for this run.</p>"

    rows: List[List[str]] = []
    rows.append(["Unit (MAG)", html.escape(str(metric_data.get("measurement_unit_mag", "-")))])
    rows.append(["Unit (GRAD)", html.escape(str(metric_data.get("measurement_unit_grad", "-")))])
    for ch_type in ("mag", "grad"):
        global_block = ((metric_data.get("PSD_global") or {}).get(ch_type) or {})
        local_block = ((metric_data.get("PSD_local") or {}).get(ch_type) or {})

        noisy_freq_count = global_block.get("noisy_frequencies_count: ", "-")
        local_details = (local_block.get("details") or {})
        total_ch = len(local_details) if isinstance(local_details, dict) else 0
        affected_ch = 0
        if isinstance(local_details, dict):
            for _, channel_payload in local_details.items():
                if isinstance(channel_payload, dict) and channel_payload:
                    affected_ch += 1
        affected_pct = (100.0 * affected_ch / total_ch) if total_ch else 0.0

        rows.append([f"{ch_type.upper()} global noisy-frequency count", html.escape(str(noisy_freq_count))])
        rows.append(
            [
                f"{ch_type.upper()} local noisy-channel burden",
                html.escape(f"{affected_ch}/{total_ch} ({affected_pct:.1f}%)"),
            ]
        )

    return (
        "<h4>PSD summary (from SimpleMetrics)</h4>"
        + _html_table(["Field", "Value"], rows)
    )


def _build_ecg_eog_affected_summary(
    *,
    metric: str,
    run_label: str,
    gqi_rows: List[Dict[str, Any]],
) -> str:
    """Return ECG/EOG affected-channel summary for one task from GQI rows."""
    task_norm = _normalize_task_token(run_label)
    matched = [
        r for r in gqi_rows
        if _normalize_task_token(r.get("task", "")) == task_norm
    ]
    if not matched:
        return (
            f"<p>{metric}: affected-channel counts/percentages are unavailable for this run "
            "(no matching GQI task row).</p>"
        )

    # Prefer latest attempt when multiple attempts exist for the same task.
    row = sorted(matched, key=lambda x: int(x.get("attempt", 0)))[-1]
    prefix = metric.upper()
    rows = [
        ["Attempt", html.escape(str(row.get("attempt", "-")))],
        ["Task", html.escape(str(row.get("task", "-")))],
        ["MAG affected channels", html.escape(f"{row.get(f'{prefix}_mag_high_corr_num', '-')}"
                                              f" ({row.get(f'{prefix}_mag_high_corr_percentage', '-')}%)")],
        ["GRAD affected channels", html.escape(f"{row.get(f'{prefix}_grad_high_corr_num', '-')}"
                                               f" ({row.get(f'{prefix}_grad_high_corr_percentage', '-')}%)")],
    ]
    return (
        f"<h4>{metric.upper()} affected-channel summary (from GQI task row)</h4>"
        + _html_table(["Field", "Value"], rows)
    )


def _extract_channel_names_for_table(channel_dict: Any) -> str:
    """Format noisy/flat channel dict payload as a comma-separated string."""
    if not isinstance(channel_dict, dict):
        return html.escape(str(channel_dict))
    if not channel_dict:
        return "-"
    return html.escape(", ".join(channel_dict.keys()))


def _render_mag_grad_rows(sensor_kind: str, payload: Dict[str, Any]) -> List[str]:
    """Render metric-specific mag/grad subsection rows (STD/PtP style)."""
    rows: List[str] = []
    sensor_label = "MAGNETOMETERS" if sensor_kind == "mag" else "GRADIOMETERS"
    rows.append(
        f"<tr><td colspan='2' style='border:1px solid #ccc; text-align:center; padding:6px;'><strong>{sensor_label}</strong></td></tr>"
    )

    if "number_of_noisy_ch" in payload:
        for key in [
            "number_of_noisy_ch",
            "percent_of_noisy_ch",
            "number_of_flat_ch",
            "percent_of_flat_ch",
            "std_lvl",
        ]:
            val = payload.get(key, "N/A")
            rows.append(
                f"<tr><td style='border:1px solid #ccc; padding:6px;'><strong>{html.escape(str(key))}</strong></td>"
                f"<td style='border:1px solid #ccc; padding:6px;'>{html.escape(str(val))}</td></tr>"
            )
        details = payload.get("details", {}) or {}
        rows.append(
            "<tr><td style='border:1px solid #ccc; padding:6px;'><strong>noisy_ch</strong></td>"
            f"<td style='border:1px solid #ccc; padding:6px;'>{_extract_channel_names_for_table(details.get('noisy_ch', {}))}</td></tr>"
        )
        rows.append(
            "<tr><td style='border:1px solid #ccc; padding:6px;'><strong>flat_ch</strong></td>"
            f"<td style='border:1px solid #ccc; padding:6px;'>{_extract_channel_names_for_table(details.get('flat_ch', {}))}</td></tr>"
        )
        return rows

    if "total_num_noisy_ep" in payload:
        for key in [
            "allow_percent_noisy_flat_epochs",
            "noisy_channel_multiplier",
            "flat_multiplier",
            "total_num_noisy_ep",
            "total_perc_noisy_ep",
            "total_num_flat_ep",
            "total_perc_flat_ep",
        ]:
            val = payload.get(key, "N/A")
            rows.append(
                f"<tr><td style='border:1px solid #ccc; padding:6px;'><strong>{html.escape(str(key))}</strong></td>"
                f"<td style='border:1px solid #ccc; padding:6px;'>{html.escape(str(val))}</td></tr>"
            )
        return rows

    rows.append(
        "<tr><td colspan='2' style='border:1px solid #ccc; padding:6px;'>No issues found here</td></tr>"
    )
    return rows


def _render_detailed_metric_rows(data: Any, *, parent_metric: str) -> List[str]:
    """Recursively render metric payload into rich table rows."""
    rows: List[str] = []
    if isinstance(data, list):
        if not data:
            rows.append("<tr><td style='border:1px solid #ccc; padding:6px;'><strong>data</strong></td>"
                        "<td style='border:1px solid #ccc; padding:6px;'>No data</td></tr>")
            return rows
        rows.append(
            "<tr><td style='border:1px solid #ccc; padding:6px;'><strong>items</strong></td>"
            f"<td style='border:1px solid #ccc; padding:6px;'>{html.escape(str(len(data)))}</td></tr>"
        )
        return rows

    if not isinstance(data, dict):
        rows.append(
            "<tr><td style='border:1px solid #ccc; padding:6px;'><strong>value</strong></td>"
            f"<td style='border:1px solid #ccc; padding:6px;'>{html.escape(str(data))}</td></tr>"
        )
        return rows

    for key, value in data.items():
        key_txt = html.escape(str(key))
        if isinstance(value, dict):
            rows.append(
                f"<tr><td colspan='2' style='border:1px solid #ccc; padding:8px; background:#e0f7fa;'><strong>{key_txt}</strong></td></tr>"
            )
            if key in {"mag", "grad"}:
                rows.extend(_render_mag_grad_rows(key, value))
                continue
            if key == "details" and parent_metric in {"STD", "PTP_MANUAL", "PTP_AUTO"}:
                rows.append(
                    "<tr><td style='border:1px solid #ccc; padding:6px;'><strong>noisy_ch</strong></td>"
                    f"<td style='border:1px solid #ccc; padding:6px;'>{_extract_channel_names_for_table(value.get('noisy_ch', {}))}</td></tr>"
                )
                rows.append(
                    "<tr><td style='border:1px solid #ccc; padding:6px;'><strong>flat_ch</strong></td>"
                    f"<td style='border:1px solid #ccc; padding:6px;'>{_extract_channel_names_for_table(value.get('flat_ch', {}))}</td></tr>"
                )
                continue
            rows.extend(_render_detailed_metric_rows(value, parent_metric=parent_metric))
            continue

        if isinstance(value, list):
            val_txt = html.escape(", ".join(str(v) for v in value)) if value else "-"
        else:
            val_txt = html.escape(str(value))
        rows.append(
            f"<tr><td style='border:1px solid #ccc; padding:6px;'><strong>{key_txt}</strong></td>"
            f"<td style='border:1px solid #ccc; padding:6px;'>{val_txt}</td></tr>"
        )
    return rows


def _build_detailed_metric_table(metric: str, metric_data: Any) -> str:
    """Build a detailed metric table preserving nested channel/epoch info."""
    if metric_data is None:
        return "<p>No SimpleMetrics payload is available for this metric/run.</p>"
    rows = _render_detailed_metric_rows(metric_data, parent_metric=metric)
    return (
        "<table style='width:100%; border-collapse:collapse;'>"
        "<thead><tr style='background:#f2f2f2;'>"
        "<th style='border:1px solid #ccc; padding:6px; text-align:left;'>Field</th>"
        "<th style='border:1px solid #ccc; padding:6px; text-align:left;'>Value</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _build_subject_qc_metric_run_panel(
    *,
    metric: str,
    run_label: str,
    report_text: str,
    metric_data: Any,
    gqi_rows: List[Dict[str, Any]],
    source_paths: List[str],
    report_strings_path: Optional[str],
    simple_metrics_path: Optional[str],
) -> str:
    """Build one run panel for one QC summary metric."""
    if not report_text and metric_data is None:
        return ""

    blocks: List[str] = []
    if report_text:
        blocks.append(
            "<p><strong>Report summary:</strong> "
            f"{_format_metric_note_html(report_text)}</p>"
        )

    if metric == "PSD":
        blocks.append(_build_psd_summary_from_simple_metrics(metric_data))
    elif metric in {"ECG", "EOG"}:
        blocks.append(
            _build_ecg_eog_affected_summary(
                metric=metric,
                run_label=run_label,
                gqi_rows=gqi_rows,
            )
        )

    if metric_data is not None:
        blocks.append("<h4>SimpleMetrics details</h4>")
        blocks.append(_build_detailed_metric_table(metric, metric_data))

    all_sources = sorted(
        set(
            [str(p) for p in (source_paths or [])]
            + [str(p) for p in [report_strings_path, simple_metrics_path] if p]
        )
    )
    if all_sources:
        src_items = "".join(
            f"<li><code>{html.escape(str(p))}</code></li>" for p in all_sources
        )
        blocks.append(f"<details><summary>Derivative sources</summary><ul>{src_items}</ul></details>")

    return "".join(blocks)


def _build_subject_report_html(
    *,
    subject: str,
    dataset_name: str,
    metrics_payload: Dict[str, List[Dict[str, Any]]],
    overview_payload: List[Dict[str, Any]],
    summary_payload: List[Dict[str, Any]],
) -> str:
    """Compose one self-contained subject report.

    This report is intentionally standalone: it embeds all JS/CSS and lazy plot
    payloads in one HTML file to avoid external dependencies or sidecar files.
    """
    _reset_lazy_figure_store()

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_tabs: List[Tuple[str, str]] = [
        (
            "Overview",
            _build_subject_overview_section(
                subject=subject,
                dataset_name=dataset_name,
                metrics_payload=metrics_payload,
                overview_payload=overview_payload,
                summary_payload=summary_payload,
            ),
        )
    ]

    for metric in METRIC_ORDER:
        metric_entries = metrics_payload.get(metric, [])
        if not metric_entries:
            continue
        run_tabs = []
        for entry in sorted(metric_entries, key=lambda d: d["run_label"]):
            run_tabs.append(
                (
                    entry["run_label"],
                    _build_metric_run_panel(
                        metric_key=metric,
                        run_label=entry["run_label"],
                        derivatives=entry["derivatives"],
                        metric_note=entry.get("metric_note", ""),
                        source_paths=entry.get("source_paths", []),
                    ),
                )
            )
        top_tabs.append(
            (
                _human_metric_label(metric),
                _build_subtabs_html(
                    group_id=f"metric-runs-{_sanitize_token(metric)}",
                    tabs=run_tabs,
                    level=2,
                ),
            )
        )

    top_tabs.append(
        (
            "QC summary",
            _build_subject_summary_section(
                subject=subject,
                summary_payload=summary_payload,
            ),
        )
    )

    tab_buttons = []
    tab_panels = []
    for idx, (label, panel_html) in enumerate(top_tabs):
        tab_id = f"subject-tab-{idx}"
        active = " active" if idx == 0 else ""
        tab_buttons.append(
            f"<button class='tab-btn{active}' data-target='{tab_id}'>{html.escape(label)}</button>"
        )
        tab_panels.append(f"<div id='{tab_id}' class='tab-content{active}'>{panel_html}</div>")

    plotly_bundle_script = _inline_plotly_bundle_script()
    lazy_payload_scripts = _lazy_payload_script_tags_html()

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Subject QA report sub-{html.escape(subject)}</title>
  {plotly_bundle_script}
  <style>
    body {{
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      margin: 0;
      color: #1c2b3a;
      background: linear-gradient(135deg, #f4f8ff, #eef5ff 40%, #f7fbff);
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 24px 18px 48px;
    }}
    section {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #dce9f7;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 6px 22px rgba(7, 41, 74, 0.08);
    }}
    h1, h2, h3, h4 {{ margin: 0 0 10px; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 22px; margin-top: 18px; }}
    h3 {{ font-size: 17px; margin-top: 12px; }}
    h4 {{ font-size: 15px; color: #234e74; margin-top: 10px; }}
    p {{ line-height: 1.45; }}
    pre {{
      white-space: pre-wrap;
      border: 1px solid #d6e2f0;
      border-radius: 10px;
      padding: 10px;
      background: #f7fbff;
      font-size: 12px;
    }}
    code {{
      background: #edf4ff;
      border: 1px solid #d3e2f7;
      border-radius: 6px;
      padding: 1px 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 13px;
    }}
    th, td {{
      border: 1px solid #d3e2f2;
      padding: 7px 8px;
      text-align: left;
    }}
    th {{
      background: #ecf4ff;
      font-weight: 700;
    }}
    .overview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }}
    .overview-card {{
      border: 1px solid #d6e3f5;
      border-radius: 11px;
      padding: 10px;
      background: #fbfdff;
    }}
    .tab-row, .subtab-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 10px 0;
    }}
    .tab-btn {{
      border: 1px solid #8cb1df;
      border-radius: 10px;
      background: #e5f1ff;
      color: #163a5d;
      font-size: 14px;
      font-weight: 700;
      padding: 8px 13px;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: #1d4ed8;
      border-color: #1d4ed8;
      color: #ffffff;
    }}
    .tab-content {{
      display: none;
      margin-top: 8px;
    }}
    .tab-content.active {{
      display: block;
    }}
    .subtab-group {{
      border: 1px solid #d7e5f6;
      border-radius: 11px;
      padding: 10px;
      margin-top: 10px;
    }}
    .subtab-group.level-2 {{
      background: #edf5ff;
      border-color: #9ec1ea;
    }}
    .subtab-group.level-3 {{
      background: #f7fbff;
      border-color: #c2d8f1;
    }}
    .subtab-btn {{
      border: 1px solid #9fc1e6;
      border-radius: 9px;
      background: #eaf3ff;
      color: #204768;
      padding: 6px 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }}
    .subtab-btn.active {{
      background: #cfe3ff;
      border-color: #79a9e4;
      color: #195b2b;
      font-weight: 700;
    }}
    .subtab-content {{
      display: none;
    }}
    .subtab-content.active {{
      display: block;
    }}
    .figure-card {{
      border-top: 1px solid #dce8f7;
      margin-top: 10px;
      padding-top: 10px;
    }}
    .figure-note {{
      margin: 8px 0 2px;
      color: #2a4765;
      font-size: 13px;
    }}
    .js-plotly-plot {{
      width: 100% !important;
    }}
    .lazy-plot-wrap {{
      margin-top: 4px;
    }}
    .plot-controls {{
      display: none;
      margin-top: 10px;
      border: 1px solid #b8d2ee;
      border-radius: 10px;
      background: #f1f7ff;
      padding: 10px 10px 8px;
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
      font-size: 14px;
      font-weight: 700;
      color: #1f4a73;
      margin: 0 0 5px;
    }}
    .plot-control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }}
    .plot-control-btn {{
      border: 1px solid #3b82f6;
      border-radius: 7px;
      background: #e9f3ff;
      color: #0f4c81;
      padding: 3px 7px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
    }}
    .plot-control-btn.active {{
      background: #cfe3ff;
      border-color: #2c6ed5;
      color: #0b5;
    }}
    .summary-iframe {{
      width: 100%;
      min-height: 980px;
      border: 1px solid #c8dbef;
      border-radius: 10px;
      background: #ffffff;
    }}
    .raw-info-wrap {{
      border: 1px solid #d6e3f5;
      border-radius: 10px;
      background: #ffffff;
      padding: 8px;
      overflow: auto;
      max-height: 980px;
    }}
    .raw-info-wrap table {{
      width: 100%;
      font-size: 13px;
    }}
    .loading-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(240, 246, 255, 0.95);
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
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <div id="report-loading-overlay" class="loading-overlay">
    <div class="loading-card">
      <div class="loading-spinner"></div>
      <h3>Loading subject report</h3>
      <p>Rendering visible figures...</p>
    </div>
  </div>
  <main>
    <section>
      <h1>MEG QC subject report</h1>
      <p><strong>Dataset:</strong> {html.escape(dataset_name)} | <strong>Subject:</strong> sub-{html.escape(subject)} | <strong>Generated:</strong> {generated}</p>
      <p>This report consolidates all metrics into one HTML file. Figures are lazily rendered when their tab becomes visible.</p>
      <div class="tab-row">
        {''.join(tab_buttons)}
      </div>
      {''.join(tab_panels)}
    </section>
  </main>
  {lazy_payload_scripts}
  <script>
    (function() {{
      const loadingOverlay = document.getElementById('report-loading-overlay');
      const topButtons = Array.from(document.querySelectorAll('.tab-btn'));
      const topPanels = Array.from(document.querySelectorAll('.tab-content'));
      const lazyPayloadCache = {{}};

      function getPayloadFromScript(payloadId) {{
        if (!payloadId) return null;
        if (lazyPayloadCache[payloadId]) return lazyPayloadCache[payloadId];
        const payloadEl = document.getElementById(payloadId);
        if (!payloadEl || !payloadEl.textContent) return null;
        try {{
          const payload = JSON.parse(payloadEl.textContent);
          lazyPayloadCache[payloadId] = payload;
          // Free the original JSON blob from the DOM after first parse.
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
        if (!loadingOverlay || loadingOverlay.dataset.hidden === '1') return;
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
          const activeIdx = Number.isFinite(group && group.active) ? Number(group.active) : 0;
          buttons.forEach((btn, idx) => {{
            const bEl = document.createElement('button');
            bEl.type = 'button';
            bEl.className = 'plot-control-btn' + (idx === activeIdx ? ' active' : '');
            bEl.textContent = String((btn && btn.label) || '');
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

      function renderLazyInScope(scopeRoot) {{
        const scope = scopeRoot || document;
        if (typeof Plotly === 'undefined') {{
          Array.from(scope.querySelectorAll('.js-lazy-plot')).forEach((el) => {{
            if (el.dataset.rendered === '1') return;
            el.dataset.rendered = '1';
            el.innerHTML = "<div style='border:1px solid #f5c2c7;background:#fff5f5;color:#8a1c1c;border-radius:8px;padding:10px;font-size:13px;'>Plotly JavaScript did not load. Open this report in a browser with JavaScript enabled.</div>";
          }});
          return Promise.resolve();
        }}
        const placeholders = Array.from(scope.querySelectorAll('.js-lazy-plot'));
        const promises = [];
        placeholders.forEach((el) => {{
          if (el.dataset.rendered === '1') return;
          if (el.offsetParent === null) return;
          const payloadId = el.dataset.payloadId;
          const payload = getPayloadFromScript(payloadId);
          if (!payload || !payload.figure) return;
          try {{
            const rendered = Plotly.newPlot(
              el,
              payload.figure.data || [],
              payload.figure.layout || {{}},
              payload.config || {{ responsive: true, displaylogo: false }}
            );
            el.dataset.rendered = '1';
            renderExternalControls(el, payload.controls || []);
            if (rendered && typeof rendered.then === 'function') {{
              promises.push(rendered.catch(() => undefined));
            }}
          }} catch (err) {{
            // continue rendering others
          }}
        }});
        return promises.length ? Promise.all(promises).then(() => undefined) : Promise.resolve();
      }}

      function resizeVisiblePlots(scopeRoot) {{
        if (typeof Plotly === 'undefined') return;
        const scope = scopeRoot || document;
        Array.from(scope.querySelectorAll('.js-plotly-plot')).forEach((plotEl) => {{
          try {{
            Plotly.Plots.resize(plotEl);
          }} catch (err) {{
            // no-op
          }}
        }});
      }}

      function hydrateSummaryInScope(scopeRoot) {{
        const scope = scopeRoot || document;
        const frames = Array.from(scope.querySelectorAll('.js-summary-iframe'));
        frames.forEach((frameEl) => {{
          if (frameEl.dataset.loaded === '1') return;
          if (frameEl.offsetParent === null) return;
          const srcdoc = frameEl.dataset.srcdoc || '';
          if (!srcdoc) return;
          frameEl.setAttribute('srcdoc', srcdoc);
          frameEl.dataset.loaded = '1';
          frameEl.removeAttribute('data-srcdoc');
        }});
      }}

      function activateTopTab(targetId) {{
        topPanels.forEach((p) => p.classList.toggle('active', p.id === targetId));
        topButtons.forEach((b) => b.classList.toggle('active', b.dataset.target === targetId));
        const panel = document.getElementById(targetId);
        if (!panel) return Promise.resolve();
        return renderLazyInScope(panel).then(() => {{
          hydrateSummaryInScope(panel);
          resizeVisiblePlots(panel);
        }});
      }}

      function activateSubtab(groupId, targetId) {{
        const buttons = Array.from(document.querySelectorAll(`.subtab-btn[data-tab-group="${{groupId}}"]`));
        const panels = Array.from(document.querySelectorAll(`.subtab-content[data-tab-group="${{groupId}}"]`));
        panels.forEach((p) => p.classList.toggle('active', p.id === targetId));
        buttons.forEach((b) => b.classList.toggle('active', b.dataset.target === targetId));
        const activePanel = document.getElementById(targetId);
        if (activePanel) {{
          renderLazyInScope(activePanel).then(() => {{
            hydrateSummaryInScope(activePanel);
            resizeVisiblePlots(activePanel);
          }});
        }}
      }}

      topButtons.forEach((btn) => {{
        btn.addEventListener('click', () => {{
          activateTopTab(btn.dataset.target);
        }});
      }});

      const subButtons = Array.from(document.querySelectorAll('.subtab-btn'));
      const seenGroups = new Set();
      subButtons.forEach((btn) => {{
        const gid = btn.dataset.tabGroup;
        if (!gid) return;
        seenGroups.add(gid);
        btn.addEventListener('click', () => activateSubtab(gid, btn.dataset.target));
      }});
      seenGroups.forEach((gid) => {{
        const first = document.querySelector(`.subtab-btn[data-tab-group="${{gid}}"]`);
        if (first) {{
          activateSubtab(gid, first.dataset.target);
        }}
      }});

      window.addEventListener('resize', () => {{
        const activeTop = topPanels.find((p) => p.classList.contains('active'));
        if (activeTop) {{
          resizeVisiblePlots(activeTop);
        }}
      }});

      window.requestAnimationFrame(() => {{
        const firstTarget = topButtons.length ? topButtons[0].dataset.target : null;
        const runPromise = firstTarget ? activateTopTab(firstTarget) : Promise.resolve();
        runPromise.then(() => {{
          window.setTimeout(hideLoadingOverlay, 120);
        }}).catch(() => hideLoadingOverlay());
      }});
    }})();
  </script>
</body>
</html>
"""


def _build_metric_derivatives(
    raw_info_path: str,
    metric: str,
    tsv_paths: List,
    report_str_path: str,
    plot_settings,
    include_sensor_plots: bool = True,
) -> Tuple[Dict[str, List[QC_derivative]], Dict[str, str]]:
    """Build derivative objects and report strings for one metric/run.

    This helper centralizes metric-specific plot computation so we can reuse the
    same plotting logic both for legacy MNE reports and the new consolidated
    subject report.
    """
    m_or_g_chosen = plot_settings['m_or_g']
    # Delegate figure creation to universal_plots backend. This keeps plotting
    # internals out of the orchestration layer and mirrors the original design
    # of MEGqc where report orchestrators call universal plotting backends.
    qc_derivs = build_metric_derivatives_from_tsv(
        metric=metric,
        tsv_paths=tsv_paths,
        m_or_g_chosen=m_or_g_chosen,
        include_sensor_plots=include_sensor_plots,
    )

    if not report_str_path:  # if no report strings were saved.
        report_strings = {
            'INITIAL_INFO': '',
            'TIME_SERIES': '',
            'STD': '',
            'PSD': '',
            'PTP_MANUAL': '',
            'PTP_AUTO': '',
            'ECG': '',
            'EOG': '',
            'HEAD': '',
            'MUSCLE': '',
            'SENSORS': '',
            'STIMULUS': ''
        }
    else:
        with open(report_str_path, "r", encoding="utf-8") as json_file:
            report_strings = json.load(json_file)

    return qc_derivs, report_strings


def csv_to_html_report(raw_info_path: str, metric: str, tsv_paths: List, report_str_path: str, plot_settings):

    """
    Create an HTML report from the CSV files.

    Parameters
    ----------
    raw_info_path : str
        The path to the raw info object.
    metric : str
        The metric to be plotted.
    tsv_paths : List
        A list of paths to the CSV files.
    report_str_path : str
        The path to the JSON file containing the report strings.
    plot_settings : dict
        A dictionary of selected settings for plotting.

    Returns
    -------
    report_html_string : str
        The HTML report as a string.

    """

    qc_derivs, report_strings = _build_metric_derivatives(
        raw_info_path=raw_info_path,
        metric=metric,
        tsv_paths=tsv_paths,
        report_str_path=report_str_path,
        plot_settings=plot_settings,
        include_sensor_plots=True,
    )
    report_html_string = make_joined_report_mne(raw_info_path, qc_derivs, report_strings)

    return report_html_string


def extract_raw_entities_from_obj(obj):

    """
    Function to create a key from the object excluding the 'desc' attribute

    Parameters
    ----------
    obj : ancpbids object
        An object from ancpbids.

    Returns
    -------
    tuple
        A tuple containing the name, extension, and suffix of the object.

    """
    # Remove the 'desc' part from the name, so we get the name of original raw that the deriv belongs to:
    raw_name = re.sub(r'_desc-[^_]+', '', obj.name)
    return (raw_name, obj.extension, obj.suffix)


def sort_tsvs_by_raw(tsvs_by_metric: dict):

    """
    For every metric, if we got same raw entitites, we can combine derivatives for the same raw into a list.
    Since we collected entities not from raw but from derivatives, we need to remove the desc part from the name.
    After that we combine files with the same 'name' in entity_val objects in 1 list:

    Parameters
    ----------
    tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files.

    Returns
    -------
    combined_tsvs_by_metric : dict
        A dictionary of metrics and their corresponding TSV files combined by raw entity

    """

    sorted_tsvs_by_metric_by_raw = {}

    for metric, obj_dict in tsvs_by_metric.items():
        combined_dict = defaultdict(list)

        for obj, tsv_path in obj_dict.items():
            raw_entities = extract_raw_entities_from_obj(obj)
            combined_dict[raw_entities].extend(tsv_path)

        # Convert keys back to original objects
        final_dict = {}
        for raw_entities, paths in combined_dict.items():
            # Find the first object with the same key
            for obj in obj_dict.keys():
                if extract_raw_entities_from_obj(obj) == raw_entities:
                    final_dict[obj] = paths
                    break

        sorted_tsvs_by_metric_by_raw[metric] = final_dict

    pprint('___MEGqc___: ', 'sorted_tsvs_by_metric_by_raw: ', sorted_tsvs_by_metric_by_raw)

    return sorted_tsvs_by_metric_by_raw

class Deriv_to_plot:

    """
    A class to represent the derivatives to be plotted.

    Attributes
    ----------
    path : str
        The path to the TSV file.
    metric : str
        The metric to be plotted.
    deriv_entity_obj : dict
        The entity object of the derivative created with ANCPBIDS.
    raw_entity_name : str
        The name of the raw entity.
    subject : str
        The subject ID.

    Methods
    -------
    __repr__()
        Return a string representation of the object.
    print_detailed_entities()
        Print the detailed entities of the object.
    find_raw_entity_name()
        Find the raw entity name from the deriv entity name.

    """

    def __init__(self, path: str, metric: str, deriv_entity_obj, raw_entity_name: str = None):

        self.path = path
        self.metric = metric
        self.deriv_entity_obj = deriv_entity_obj
        self.raw_entity_name = raw_entity_name

        # Extract subject ID using a BIDS-compliant regex (alphanumeric labels)
        name = deriv_entity_obj.get('name', '') or ''
        match = re.search(r'sub-([A-Za-z0-9]+)_', name)
        self.subject = match.group(1) if match else None

    def __repr__(self):

        return (
            f"Deriv_to_plot(\n"
            f"    subject={self.subject},\n"
            f"    path={self.path},\n"
            f"    metric={self.metric},\n"
            f"    deriv_entity_obj={self.deriv_entity_obj},\n"
            f"    raw_entity_name={self.raw_entity_name}\n"
            f")"
        )

    def print_detailed_entities(self):

        """
        Print the detailed entities of the object, cos in ANCP representation it s cut.
        Here skipping the last value, cos it s the contents.
        If u got a lot of contents, like html it ll print you a book XD.
        """

        keys = list(self.deriv_entity_obj.keys())
        for val in keys[:-1]:  # Iterate over all keys except the last one
            print('_Deriv_: ', val, self.deriv_entity_obj[val])

    def find_raw_entity_name(self):

        """
        Find the raw entity name from the deriv entity name
        """

        self.raw_entity_name = re.sub(r'_desc-.*', '', self.deriv_entity_obj['name'])


from joblib import Parallel, delayed


def process_subject(
        sub: str,
        derivs_to_plot: list,
        chosen_entities: dict,
        plot_settings: dict,
        output_derivatives_root: str,
        dataset_name: str,
        analysis_segments: Optional[List[str]] = None,
):
    """Build one consolidated HTML report for a single subject.

    The legacy implementation wrote one HTML per metric and run. This function
    now keeps the same metric computations but reorganizes output into one
    subject-level report with nested tabs and inline lazy Plotly rendering.
    """

    reports_root = _ensure_megqc_output_layout(
        output_derivatives_root,
        analysis_segments=analysis_segments,
    )

    # Sort run keys for deterministic tab ordering.
    existing_raws_per_sub = sorted(set(
        d.raw_entity_name for d in derivs_to_plot if d.subject == sub
    ))
    run_tab_labels = _build_run_tab_labels(existing_raws_per_sub)

    metrics_to_plot = [
        m for m in chosen_entities['METRIC']
        if m not in ['RawInfo', 'ReportStrings', 'SimpleMetrics']
    ]
    # ``metrics_payload`` drives the nested tab hierarchy:
    # metric -> list of run entries with derivatives.
    metrics_payload: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    overview_payload: List[Dict[str, Any]] = []

    # ``summary_payload`` stores run-level metadata and JSON references used
    # by the "QC summary" tab.
    summary_payload: List[Dict[str, Any]] = []

    for raw_entity_name in existing_raws_per_sub:
        derivs_for_this_raw = [
            d for d in derivs_to_plot if d.raw_entity_name == raw_entity_name
        ]
        if not derivs_for_this_raw:
            continue

        raw_info_path = None
        report_str_path = None
        simple_metrics_path = None
        for d in derivs_for_this_raw:
            if d.metric == 'RawInfo':
                raw_info_path = d.path
            elif d.metric == 'ReportStrings':
                report_str_path = d.path
            elif d.metric == 'SimpleMetrics':
                simple_metrics_path = d.path

        run_label = run_tab_labels.get(raw_entity_name, _human_run_label(raw_entity_name))
        run_source_paths = [
            d.path
            for d in derivs_for_this_raw
            if d.metric not in ['RawInfo', 'ReportStrings', 'SimpleMetrics']
        ]
        raw_info_html = _safe_load_raw_info_html(raw_info_path)
        sensor_derivs, sensor_paths = _collect_run_sensor_derivatives(derivs_for_this_raw)
        overview_payload.append(
            {
                "run_label": run_label,
                "sensor_derivatives": sensor_derivs,
                "source_paths": sensor_paths,
            }
        )
        summary_payload.append(
            {
                "run_label": run_label,
                "raw_info_path": raw_info_path,
                "raw_info_html": raw_info_html,
                "report_strings_path": report_str_path,
                "simple_metrics_path": simple_metrics_path,
                "source_paths": run_source_paths,
            }
        )

        for metric in metrics_to_plot:
            tsv_paths = [d.path for d in derivs_for_this_raw if d.metric == metric]
            if not tsv_paths:
                continue

            try:
                qc_derivs, report_strings = _build_metric_derivatives(
                    raw_info_path=raw_info_path,
                    metric=metric,
                    tsv_paths=tsv_paths,
                    report_str_path=report_str_path,
                    plot_settings=plot_settings,
                    include_sensor_plots=False,
                )
            except Exception as exc:
                print(
                    f"___MEGqc___: Failed to build derivatives for "
                    f"sub-{sub} / {run_label} / {metric}: {exc}"
                )
                continue

            section_key = _metric_to_report_section(metric)
            if not section_key:
                continue
            derivatives = qc_derivs.get(section_key, [])
            metric_note = str(report_strings.get(section_key, "") or "")
            if (not derivatives) and (not metric_note):
                continue

            metrics_payload[metric].append(
                {
                    "run_label": run_label,
                    "derivatives": derivatives,
                    "metric_note": metric_note,
                    "source_paths": tsv_paths,
                }
            )

    # Emit one subject-level HTML report.
    report_html = _build_subject_report_html(
        subject=sub,
        dataset_name=dataset_name,
        metrics_payload=metrics_payload,
        overview_payload=overview_payload,
        summary_payload=summary_payload,
    )

    subject_folder = reports_root / f"sub-{sub}"
    subject_folder.mkdir(parents=True, exist_ok=True)
    report_path = subject_folder / f"sub-{sub}_desc-subject_qa_report_meg.html"
    report_path.write_text(report_html, encoding="utf-8")
    return


def make_plots_meg_qc(
    dataset_path: str,
    n_jobs: int = 1,
    derivatives_base: Optional[str] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
):
    """
    Create plots for the MEG QC pipeline, but WITHOUT the interactive selector.
    Instead, we assume 'all' for every entity (subject, task, session, run, metric).
    """

    # Ensure plotting backend and report helpers are available
    _load_plotting_backend()

    start_time = time.time()


    try:
        dataset = ancpbids.load_dataset(dataset_path, DatasetOptions(lazy_loading=True))
        schema = dataset.get_schema()
    except Exception:
        print('___MEGqc___: ',
              'No data found in the given directory path! \nCheck directory path in config file and presence of data.')
        return

    (
        output_root,
        _derivatives_root,
        _megqc_root,
        _resolved_analysis_id,
        analysis_segments,
    ) = resolve_analysis_root(
        dataset_path=dataset_path,
        external_derivatives_root=derivatives_base,
        analysis_mode=analysis_mode,
        analysis_id=analysis_id,
        create_if_missing=True,
    )
    dataset_derivatives_root = os.path.join(dataset_path, "derivatives")
    output_derivatives_root = os.path.join(output_root, "derivatives")

    # Query derivatives source:
    # - Prefer external derivatives tree when it already contains Meg_QC
    #   calculations (useful for fully external pipelines).
    # - Otherwise fall back to derivatives inside the input dataset, while
    #   still writing reports into ``output_root``.
    source_derivatives_root = output_derivatives_root
    calc_rel = os.path.join('Meg_QC', *analysis_segments, 'calculation')
    if not os.path.isdir(os.path.join(source_derivatives_root, calc_rel)):
        source_derivatives_root = dataset_derivatives_root

    print(f"___MEGqc___: Reading derivatives from: {source_derivatives_root}")

    query_dataset = dataset
    query_base = dataset_path
    overlay_tmp = None

    # If query derivatives are not the dataset-local derivatives, build a
    # lightweight overlay so ANCPBIDS can resolve scope='derivatives/...'.
    if os.path.abspath(source_derivatives_root) != os.path.abspath(dataset_derivatives_root):
        overlay_tmp, overlay_root = build_overlay_dataset(dataset_path, source_derivatives_root)
        query_base = overlay_root
        query_dataset = ancpbids.load_dataset(overlay_root, DatasetOptions(lazy_loading=True))
        print(f"___MEGqc___: Using overlay dataset for queries at: {overlay_root}")

    calculated_derivs_folder = os.path.join(
        'derivatives', 'Meg_QC', *analysis_segments, 'calculation'
    )

    # Create output derivative folders once before subject-parallel processing.
    _ensure_megqc_output_layout(
        output_derivatives_root,
        analysis_segments=analysis_segments,
    )

    # --------------------------------------------------------------------------------
    # REPLACE THE SELECTOR WITH A HARDCODED "ALL" CHOICE
    # --------------------------------------------------------------------------------
    # 1) Get all discovered entities from the derivatives scope
    entities_found = get_ds_entities(query_dataset, calculated_derivs_folder, query_base)

    # Suppose 'description' is the metric list
    all_metrics = entities_found.get('description', [])

    # If you want them deduplicated, do:
    all_metrics = list(set(all_metrics))

    # Collapse individual PSD descriptions into a single entry so that the
    # general PSD report (``PSDs``) can gather all derivatives at once.  This
    # prevents the loss of the ``PSDs`` report when only the noise/waves
    # derivatives are present in the dataset.
    psd_related = {'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad'}
    if psd_related.intersection(all_metrics):
        all_metrics = [m for m in all_metrics if m not in psd_related]
        if 'PSDs' not in all_metrics:
            all_metrics.append('PSDs')

    # Retain only recognised metrics and normalise some aliases. This prevents
    # intermediate derivatives like ``ECGchannel`` from being treated as
    # standalone metrics and generating separate HTML reports.
    valid_metrics = {
        'STDs': 'STDs',
        'STD': 'STDs',
        'PSDs': 'PSDs',
        'PtPsManual': 'PtPsManual',
        'PtPsAuto': 'PtPsAuto',
        'ECGs': 'ECGs',
        'EOGs': 'EOGs',
        'Head': 'Head',
        'Muscle': 'Muscle',
        'RawInfo': 'RawInfo',
        'ReportStrings': 'ReportStrings',
        'SimpleMetrics': 'SimpleMetrics',
    }
    all_metrics = [valid_metrics[m] for m in all_metrics if m in valid_metrics]
    # Preserve order while removing duplicates
    #all_metrics = list(dict.fromkeys(all_metrics))

    # Now store it in chosen_entities as a list
    chosen_entities = {
        'subject': list(entities_found.get('subject', [])),
        'task': list(entities_found.get('task', [])),
        'session': list(entities_found.get('session', [])),
        'run': list(entities_found.get('run', [])),
        'METRIC': all_metrics
    }

    # And now you can append or pop, etc.
    chosen_entities['METRIC'].append('stimulus')
    chosen_entities['METRIC'].append('RawInfo')
    chosen_entities['METRIC'].append('ReportStrings')
    # Ensure SimpleMetrics is always present so that summary reports can be built
    chosen_entities['METRIC'].append('SimpleMetrics')

    # 5) Define a simple plot_settings. Example: always 'mag' and 'grad'
    plot_settings = {'m_or_g': ['mag', 'grad']}

    print('___MEGqc___: CHOSEN entities to plot:', chosen_entities)
    print('___MEGqc___: CHOSEN settings:', plot_settings)
    # --------------------------------------------------------------------------------

    try:
        # 2. Collect TSVs for each sub + metric
        tsvs_to_plot_by_metric = {}
        tsv_entities_by_metric = {}

        for metric in chosen_entities['METRIC']:
            query_args = {
                'subj': chosen_entities['subject'],
                'task': chosen_entities['task'],
                'suffix': 'meg',
                'extension': ['tsv', 'json', 'fif'],
                'return_type': 'filename',
                'desc': '',
                'scope': calculated_derivs_folder,
            }

            # If the user (now "all") had multiple possible descs for PSDs, ECGs, etc.
            if metric == 'PSDs':
                # Include all PSD derivatives (noise and waves) so the PSD report is
                # generated correctly.
                query_args['desc'] = ['PSDs', 'PSDnoiseMag', 'PSDnoiseGrad', 'PSDwavesMag', 'PSDwavesGrad']
            elif metric == 'ECGs':
                query_args['desc'] = ['ECGchannel', 'ECGs']
            elif metric == 'EOGs':
                query_args['desc'] = ['EOGchannel', 'EOGs']
            else:
                query_args['desc'] = [metric]

            # Optional session/run
            if chosen_entities['session']:
                query_args['session'] = chosen_entities['session']
            if chosen_entities['run']:
                query_args['run'] = chosen_entities['run']

            with temporary_dataset_base(query_dataset, query_base):
                tsv_paths = list(query_dataset.query(**query_args))
            tsvs_to_plot_by_metric[metric] = sorted(tsv_paths)

            # Now query object form for ancpbids entities
            query_args['return_type'] = 'object'
            with temporary_dataset_base(query_dataset, query_base):
                entities_obj = sorted(list(query_dataset.query(**query_args)), key=lambda k: k['name'])
            tsv_entities_by_metric[metric] = entities_obj

        # Convert them into a list of Deriv_to_plot objects
        derivs_to_plot = []
        for (tsv_metric, tsv_paths), (entity_metric, entity_vals) in zip(
            tsvs_to_plot_by_metric.items(),
            tsv_entities_by_metric.items()
        ):
            if tsv_metric != entity_metric:
                raise ValueError('Different metrics in tsvs_to_plot_by_metric and entities_per_file')
            if len(tsv_paths) != len(entity_vals):
                raise ValueError(f'Different number of tsvs and entities for metric: {tsv_metric}')

            for tsv_path, deriv_entities in zip(tsv_paths, entity_vals):
                file_name_in_path = os.path.basename(tsv_path).split('_meg.')[0]
                file_name_in_obj = deriv_entities['name'].split('_meg.')[0]

                if file_name_in_obj not in file_name_in_path:
                    raise ValueError('Different names in tsvs_to_plot_by_metric and entities_per_file')

                deriv = Deriv_to_plot(path=tsv_path, metric=tsv_metric, deriv_entity_obj=deriv_entities)
                deriv.find_raw_entity_name()
                derivs_to_plot.append(deriv)

        # Parallel execution per subject
        Parallel(n_jobs=n_jobs)(
            delayed(process_subject)(
                sub=sub,
                derivs_to_plot=derivs_to_plot,
                chosen_entities=chosen_entities,
                plot_settings=plot_settings,
                output_derivatives_root=output_derivatives_root,
                dataset_name=os.path.basename(os.path.normpath(dataset_path)),
                analysis_segments=analysis_segments,
            )
            for sub in chosen_entities['subject']
        )

        end_time = time.time()
        elapsed_seconds = end_time - start_time
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print("---------------------------------------------------------------")
        print(f"PLOTTING MODULE FINISHED. Elapsed time: {elapsed_seconds:.2f} seconds.")
    finally:
        # Ensure the temporary overlay is cleaned up.
        if overlay_tmp is not None:
            overlay_tmp.cleanup()
    return


# ____________________________
# RUN IT:

# make_plots_meg_qc(dataset_path='/data/areer/MEG_QC_stuff/data/openneuro/ds003483')

# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds003483')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds000117')
# make_plots_meg_qc(dataset_path='/Users/jenya/Local Storage/Job Uni Rieger lab/data/ds83')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/openneuro/ds004330')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/camcan')

# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds000246')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds000247')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds002761')
# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/CTF/ds004398')


# make_plots_meg_qc(dataset_path='/Volumes/SSD_DATA/MEG_data/BIDS/ceegridCut')
