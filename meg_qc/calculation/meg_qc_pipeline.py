import os
import gc
import re
import ancpbids
from ancpbids.query import query_entities
from ancpbids import DatasetOptions
import time
import json
import sys
import mne
import shutil
import glob
import hashlib
from typing import List, Union
from joblib import Parallel, delayed
import time
import datetime as dt
import importlib.metadata


# Needed to import the modules without specifying the full path, for command line and jupyter notebook
sys.path.append(os.path.join('.'))
sys.path.append(os.path.join('.', 'meg_qc', 'calculation'))

# relative path for `make html` (docs)
sys.path.append(os.path.join('..', 'meg_qc', 'calculation'))

# relative path for `make html` (docs) run from https://readthedocs.org/
# every time rst file is nested inside of another, need to add one more path level here:
sys.path.append(os.path.join('..', '..', 'meg_qc', 'calculation'))
sys.path.append(os.path.join('..', '..', '..', 'meg_qc', 'calculation'))
sys.path.append(os.path.join('..', '..', '..', '..', 'meg_qc', 'calculation'))

from meg_qc.calculation.initial_meg_qc import (
    delete_temp_folder,
    get_all_config_params,
    get_internal_config_params,
    initial_processing,
    remove_fif_and_splits,
)
# from meg_qc.plotting.universal_html_report import make_joined_report, make_joined_report_mne
from meg_qc.plotting.universal_plots import QC_derivative

from meg_qc.calculation.metrics.STD_meg_qc import STD_meg_qc
from meg_qc.calculation.metrics.PSD_meg_qc import PSD_meg_qc
from meg_qc.calculation.metrics.Peaks_manual_meg_qc import PP_manual_meg_qc
from meg_qc.calculation.metrics.Peaks_manual_meg_qc_numba import PP_manual_meg_qc_numba
from meg_qc.calculation.metrics.Peaks_auto_meg_qc import PP_auto_meg_qc
from meg_qc.calculation.metrics.ECG_EOG_meg_qc import ECG_meg_qc, EOG_meg_qc
from meg_qc.calculation.metrics.Head_meg_qc import HEAD_movement_meg_qc
from meg_qc.calculation.metrics.muscle_meg_qc import MUSCLE_meg_qc

import os
import json
import pandas as pd
from typing import Union, Optional, Dict, Tuple
from contextlib import contextmanager

from meg_qc.calculation.metrics.summary_report_GQI import generate_gqi_summary


_ANALYSIS_MODES = {"legacy", "new", "reuse", "latest"}
# Prompt-based policies were removed to guarantee non-blocking execution in
# CLI batch runs and GUI worker subprocesses.
_CONFIG_POLICIES = {"provided", "latest_saved", "fail"}
_PROCESSED_SUBJECT_POLICIES = {"skip", "rerun", "fail"}


def _timestamp_analysis_id() -> str:
    """Return a compact profile identifier suitable for filesystem paths."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_roots(dataset_path: str, external_derivatives_root: Optional[str]) -> Tuple[str, str]:
    """Return the dataset output root and derivatives folder respecting overrides.

    Parameters
    ----------
    dataset_path : str
        Path to the original BIDS dataset.
    external_derivatives_root : Optional[str]
        User-provided folder in which a dataset-named directory will be created
        to host derivatives. If ``None`` the derivatives live inside the
        original dataset.

    Returns
    -------
    tuple
        ``(output_root, derivatives_root)`` where ``output_root`` is the base
        dataset directory used when writing derivatives and ``derivatives_root``
        points to the "derivatives" folder inside ``output_root``.
    """

    ds_name = os.path.basename(os.path.normpath(dataset_path))
    output_root = dataset_path if external_derivatives_root is None else os.path.join(external_derivatives_root, ds_name)
    derivatives_root = os.path.join(output_root, 'derivatives')
    os.makedirs(derivatives_root, exist_ok=True)

    # When output is external, seed output_root with a dataset_description.json.
    # The plotting module loads ANCPBIDS directly from output_root (no symlink
    # overlay) so this file must be present for schema-version detection.
    if external_derivatives_root is not None:
        desc_dst = os.path.join(output_root, "dataset_description.json")
        if not os.path.exists(desc_dst):
            desc_src = os.path.join(dataset_path, "dataset_description.json")
            if os.path.exists(desc_src):
                try:
                    import shutil as _shutil
                    _shutil.copy2(desc_src, desc_dst)
                except OSError:
                    pass
            if not os.path.exists(desc_dst):
                import json as _json
                stub = {"Name": os.path.basename(output_root), "BIDSVersion": "1.8.0"}
                try:
                    with open(desc_dst, 'w', encoding='utf-8') as _fh:
                        _json.dump(stub, _fh, indent=2)
                except OSError:
                    pass  # Non-fatal; ancpbids falls back gracefully.

    return output_root, derivatives_root


def _scan_profile_dir(profiles_root: str) -> List[Tuple[str, float]]:
    """Scan one profiles directory and return ``(name, mtime)`` pairs sorted by mtime desc.

    Returns an empty list when the directory does not exist.
    """
    if not os.path.isdir(profiles_root):
        return []
    candidates = []
    for entry in os.listdir(profiles_root):
        full = os.path.join(profiles_root, entry)
        if os.path.isdir(full):
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                mtime = 0.0
            candidates.append((entry, mtime))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def list_analysis_profiles(
    dataset_path: str,
    external_derivatives_root: Optional[str] = None,
) -> List[str]:
    """List available MEGqc profile IDs for one dataset.

    Profiles are discovered under ``derivatives/Meg_QC/profiles/<analysis_id>``.
    Returned IDs are sorted by latest modification time first.

    When an external derivatives root is provided **both** the external output
    location and the original dataset's derivatives folder are searched.
    External profiles take precedence; original-dataset profiles are appended
    for any IDs not already found in the external location.

    This makes profile mode work correctly for Scenario C: the user ran
    calculation without an external path (so profiles live inside the original
    BIDS dataset) but now wants to direct plotting reports to an external
    output path.  Without this dual search the GUI "Load profiles" dialog shows
    nothing and ``resolve_analysis_root`` raises ``FileNotFoundError``.
    """
    _, derivatives_root = resolve_output_roots(dataset_path, external_derivatives_root)
    primary_profiles_root = os.path.join(derivatives_root, "Meg_QC", "profiles")
    candidates = _scan_profile_dir(primary_profiles_root)

    # Also search the original dataset when an external path is set and the two
    # locations differ (covers the case where calc ran without --derivatives_output).
    if external_derivatives_root is not None:
        original_profiles_root = os.path.join(
            dataset_path, "derivatives", "Meg_QC", "profiles"
        )
        if os.path.abspath(original_profiles_root) != os.path.abspath(primary_profiles_root):
            original_candidates = _scan_profile_dir(original_profiles_root)
            existing_names = {name for name, _ in candidates}
            for name, mtime in original_candidates:
                if name not in existing_names:
                    candidates.append((name, mtime))
            candidates.sort(key=lambda item: item[1], reverse=True)

    return [name for name, _ in candidates]


def resolve_analysis_root(
    dataset_path: str,
    external_derivatives_root: Optional[str] = None,
    analysis_mode: str = "legacy",
    analysis_id: Optional[str] = None,
    create_if_missing: bool = False,
) -> Tuple[str, str, str, Optional[str], List[str]]:
    """Resolve output roots plus profile-specific MEGqc folder.

    Returns
    -------
    tuple
        ``(output_root, derivatives_root, megqc_root, resolved_analysis_id, analysis_segments)``
        where ``analysis_segments`` is ``[]`` in legacy mode and
        ``["profiles", resolved_analysis_id]`` otherwise.
    """
    mode = str(analysis_mode or "legacy").strip().lower()
    if mode not in _ANALYSIS_MODES:
        raise ValueError(
            f"Invalid analysis_mode '{analysis_mode}'. Supported modes: "
            f"{', '.join(sorted(_ANALYSIS_MODES))}."
        )

    output_root, derivatives_root = resolve_output_roots(dataset_path, external_derivatives_root)
    legacy_root = os.path.join(derivatives_root, "Meg_QC")
    if mode == "legacy":
        if create_if_missing:
            os.makedirs(legacy_root, exist_ok=True)
        return output_root, derivatives_root, legacy_root, None, []

    profiles_root = os.path.join(legacy_root, "profiles")
    if create_if_missing:
        os.makedirs(profiles_root, exist_ok=True)

    resolved_id = analysis_id.strip() if isinstance(analysis_id, str) and analysis_id.strip() else None
    available = list_analysis_profiles(dataset_path, external_derivatives_root)

    if mode == "new":
        resolved_id = resolved_id or _timestamp_analysis_id()
    elif mode == "reuse":
        if not resolved_id:
            raise ValueError("analysis_mode='reuse' requires analysis_id.")
        if resolved_id not in available and not create_if_missing:
            raise FileNotFoundError(
                f"Profile '{resolved_id}' not found under {profiles_root}."
            )
    elif mode == "latest":
        if available:
            resolved_id = available[0]
        else:
            raise FileNotFoundError(
                f"No analysis profiles found under {profiles_root}. "
                "Use analysis_mode='new' to create one, or analysis_mode='legacy'."
            )

    profile_root = os.path.join(profiles_root, str(resolved_id))
    if create_if_missing:
        os.makedirs(profile_root, exist_ok=True)

    analysis_segments = ["profiles", str(resolved_id)]
    return output_root, derivatives_root, profile_root, resolved_id, analysis_segments


def _config_dir_from_megqc_root(megqc_root: str) -> str:
    return os.path.join(megqc_root, "config")


def _list_used_settings_files(config_dir: str) -> List[str]:
    """Return saved config snapshots sorted by latest first."""
    if not os.path.isdir(config_dir):
        return []
    # Support both _meg.ini (default) and _eeg.ini suffixed config snapshots
    files = []
    for suffix in ("meg", "eeg"):
        pattern = os.path.join(config_dir, f"*_desc-UsedSettings*_{suffix}.ini")
        files.extend(glob.glob(pattern))
    # Deduplicate (in case of overlapping globs) and sort newest first
    files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _resolve_config_by_policy(
    default_config_file_path: str,
    config_dir: str,
    existing_config_policy: str = "provided",
    interactive_prompts: bool = False,
) -> Optional[str]:
    """Resolve the config file for one run according to explicit policy."""
    policy = str(existing_config_policy or "provided").strip().lower()
    if policy not in _CONFIG_POLICIES:
        raise ValueError(
            f"Invalid existing_config_policy '{existing_config_policy}'. Supported: "
            f"{', '.join(sorted(_CONFIG_POLICIES))}."
        )

    existing = _list_used_settings_files(config_dir)
    if policy == "provided":
        return default_config_file_path
    if policy == "latest_saved":
        return existing[0] if existing else default_config_file_path
    if policy == "fail":
        if existing:
            raise RuntimeError(
                "Existing config snapshots were found for this dataset/profile. "
                "Choose existing_config_policy='provided' or 'latest_saved'."
            )
        return default_config_file_path

    raise RuntimeError(
        "Prompt-based config policy is no longer supported. "
        "Use existing_config_policy='provided', 'latest_saved', or 'fail'."
    )


@contextmanager
def temporary_dataset_base(dataset, base_dir: str):
    """Temporarily point an ANCPBIDS dataset to a different base directory.

    This is used to redirect derivative writing without interfering with how
    raw files are located inside the original BIDS dataset.
    """

    original_base = getattr(dataset, 'base_dir_', None)
    dataset.base_dir_ = base_dir
    try:
        yield
    finally:
        dataset.base_dir_ = original_base


def _ensure_derivative_dataset_description_filename(derivative) -> None:
    """Guarantee a writable dataset_description filename for ANCPBIDS writes.

    Some datasets contain partially written legacy ``derivatives/Meg_QC`` trees.
    In that state ANCPBIDS can materialize ``derivative.dataset_description``
    with ``name=None``. During ``write_derivative`` this makes the writer target
    the folder path (``.../derivatives/Meg_QC``) as if it were a file, which
    triggers ``No file writer registered``.

    We normalize this eagerly so repeated runs remain robust regardless of
    pre-existing derivative state.
    """

    dataset_description = getattr(derivative, "dataset_description", None)
    if dataset_description is None:
        return

    # Only patch missing/blank names; preserve explicit names if present.
    if not getattr(dataset_description, "name", None):
        dataset_description.name = "dataset_description.json"

def ctf_workaround(dataset, sid):
    artifacts = dataset.query(suffix="meg", return_type="object", subj=sid, scope='raw')
    # convert to folders of found files
    folders = map(lambda a: a.get_parent().get_absolute_path(), artifacts)
    # remove duplicates
    folders = set(folders)
    # convert to liust before filtering
    folders = list(folders)

    # filter for folders which end with ".ds" (including os specific path separator)
    # folders = list(filter(lambda f: f.endswith(f"{os.sep}.ds"), folders))

    # Filter for folders which end with ".ds"
    filtered_folders = [f for f in folders if f.endswith('.ds')]

    return sorted(filtered_folders)


def get_files_list(sid: str, dataset_path: str, dataset, m_or_g_chosen: list = None):
    """
    Use BIDS-compliant modality classification (via ancpbids) to discover
    MEG and EEG files for a given subject.

    The **primary modality signal** is the BIDS suffix that the dataset
    itself provides (``suffix='meg'`` for MEG recordings, ``suffix='eeg'``
    for independent EEG recordings).  This is the canonical way BIDS
    separates modalities and works regardless of the underlying file format
    (.fif, .ds, .edf, .set, …).

    A secondary filesystem walk is still performed to distinguish FIF-based
    MEG from CTF (.ds) MEG, since they require different loading strategies.

    Parameters
    ----------
    sid : str
        Subject ID to get the files for.
    dataset_path : str
        Path to the BIDS-conform data set to run the QC on.
    dataset : ancpbids.Dataset
        Dataset object to work with.
    m_or_g_chosen : list, optional
        Channel types chosen by the user (e.g. ['mag', 'grad'] or ['eeg']).
        Used to resolve ambiguity in multimodal datasets that contain both
        MEG and EEG files.

    Returns
    -------
    list_of_files : list
        List of paths to the data files for each subject.
    entities_per_file : list
        List of entities for each file in list_of_files.
    """

    # ── 1. Determine user preference from config ─────────────────────────
    _user_wants_eeg = False
    _user_wants_meg = False
    if m_or_g_chosen:
        _user_wants_eeg = 'eeg' in m_or_g_chosen
        _user_wants_meg = any(t in m_or_g_chosen for t in ('mag', 'grad'))

    # ── 2. Query ancpbids for MEG and EEG files using BIDS suffix ────────
    # This is the primary modality classifier: BIDS suffix in the dataset
    # structure (e.g., sub-XX/ses-meg/meg/*_meg.fif  or
    #                     sub-XX/ses-eeg/eeg/*_eeg.set).
    # ancpbids returns files whose BIDS suffix matches, regardless of format.

    # MEG files (suffix='meg')
    meg_files_raw = sorted(
        list(dataset.query(suffix='meg', return_type='filename', subj=sid, scope='raw')))
    # EEG files (suffix='eeg')
    eeg_files_raw = sorted(
        list(dataset.query(suffix='eeg', return_type='filename', subj=sid, scope='raw')))

    has_meg = len(meg_files_raw) > 0
    has_eeg = len(eeg_files_raw) > 0

    # ── 3. Distinguish FIF vs CTF inside the MEG pool ────────────────────
    # We still need to know whether the MEG files are FIF or CTF because
    # CTF requires a special loading path (ctf_workaround).
    has_fif = False
    has_ctf = False

    for root, dirs, files in os.walk(dataset_path):
        # Exclude the 'derivatives' folder — our own FIF derivatives could
        # otherwise confuse CTF datasets.
        dirs[:] = [d for d in dirs if d != 'derivatives']
        if any(file.endswith('.fif') for file in files):
            has_fif = True
        if any(d.endswith('.ds') for d in dirs):
            has_ctf = True
        if has_fif and has_ctf:
            raise ValueError(
                'Both fif and ctf files found in the dataset. '
                'Cannot define how to read the ds.')

    # ── 4. Resolve multimodal ambiguity using user preference ────────────
    if has_meg and has_eeg:
        if _user_wants_eeg and _user_wants_meg:
            print('___MEGqc___: Multimodal dataset detected (MEG + EEG). '
                  'Config requests both MEG and EEG channels — processing all files.')
        elif _user_wants_eeg and not _user_wants_meg:
            print('___MEGqc___: Multimodal dataset detected (MEG + EEG). '
                  'Config requests EEG channels — processing EEG files only.')
            has_meg = False
        elif _user_wants_meg and not _user_wants_eeg:
            print('___MEGqc___: Multimodal dataset detected (MEG + EEG). '
                  'Config requests MEG channels — processing MEG files only.')
            has_eeg = False
        else:
            # Unspecified — default to MEG (legacy behaviour)
            print('___MEGqc___: Multimodal dataset detected (MEG + EEG). '
                  'Defaulting to MEG files. Set ch_types = eeg in config '
                  'to process EEG instead.')
            has_eeg = False

    # ── 5. Collect files from each modality ──────────────────────────────
    list_of_files = []
    entities_per_file = []

    if has_meg:
        if has_ctf:
            # CTF: the data files are directories ending in .ds — use the
            # existing workaround that resolves to the .ds folder paths.
            ctf_files = ctf_workaround(dataset, sid)
            ctf_entities = dataset.query(subj=sid, suffix='meg', extension='.res4', scope='raw')
            ctf_entities = sorted(ctf_entities, key=lambda k: k['name'])
            list_of_files += ctf_files
            entities_per_file += ctf_entities
        else:
            # FIF (or any other non-CTF MEG format)
            meg_files = sorted(
                list(dataset.query(suffix='meg', extension='.fif',
                                   return_type='filename', subj=sid, scope='raw')))
            meg_entities = dataset.query(subj=sid, suffix='meg', extension='.fif', scope='raw')
            meg_entities = sorted(meg_entities, key=lambda k: k['name'])
            list_of_files += meg_files
            entities_per_file += meg_entities

    if has_eeg:
        # Query all files with BIDS suffix='eeg', regardless of extension.
        # ancpbids already filters by the BIDS modality folder (eeg/).
        eeg_files = sorted(
            list(dataset.query(suffix='eeg', return_type='filename', subj=sid, scope='raw')))
        eeg_entities = dataset.query(subj=sid, suffix='eeg', scope='raw')
        eeg_entities = sorted(eeg_entities, key=lambda k: k['name']) if eeg_entities else []

        # Keep only actual data files (skip sidecars like .json, .tsv, .fdt)
        _EEG_DATA_EXTENSIONS = {'.edf', '.bdf', '.vhdr', '.set', '.cnt', '.mff', '.fif'}

        # Filter files and entities independently by extension, then match
        # by filename to avoid positional misalignment when one query
        # includes companion files (.fdt, .json, etc.) that the other does not.
        eeg_data_files = [f for f in eeg_files
                          if os.path.splitext(f)[1].lower() in _EEG_DATA_EXTENSIONS]
        # Build a name→entity lookup from the entity list
        _entity_by_name = {}
        for e in eeg_entities:
            ename = e['name'] if isinstance(e, dict) else e.name
            _entity_by_name[ename] = e

        eeg_data_entities = []
        eeg_data_files_matched = []
        for f in eeg_data_files:
            fname = os.path.basename(f)
            if fname in _entity_by_name:
                eeg_data_files_matched.append(f)
                eeg_data_entities.append(_entity_by_name[fname])
            else:
                # Entity not found by exact name — create a minimal dict
                print(f'___MEGqc___: No entity match for {fname}; creating synthetic entity.')
                eeg_data_files_matched.append(f)
                eeg_data_entities.append({'name': fname})
        eeg_data_files = eeg_data_files_matched

        if eeg_data_files:
            detected_ext = os.path.splitext(eeg_data_files[0])[1].lower()
            print(f'___MEGqc___: EEG modality detected via BIDS suffix '
                  f'(extension: {detected_ext}, {len(eeg_data_files)} file(s))')
            list_of_files += eeg_data_files
            entities_per_file += eeg_data_entities
        else:
            # Fallback: ancpbids returned no data files — try glob
            _EEG_EXTENSIONS = {'.edf', '.bdf', '.vhdr', '.set', '.cnt'}
            print('___MEGqc___: ancpbids returned no EEG data files; using glob fallback...')
            eeg_glob_files = []
            for ext in _EEG_EXTENSIONS:
                for gp in [
                    os.path.join(dataset_path, f'sub-{sid}', 'eeg', f'*_eeg{ext}'),
                    os.path.join(dataset_path, f'sub-{sid}', 'ses-*', 'eeg', f'*_eeg{ext}'),
                ]:
                    eeg_glob_files = sorted(glob.glob(gp))
                    if eeg_glob_files:
                        break
                if eeg_glob_files:
                    break
            if eeg_glob_files:
                eeg_glob_entities = [{'name': os.path.basename(f)} for f in eeg_glob_files]
                list_of_files += eeg_glob_files
                entities_per_file += eeg_glob_entities

    if not list_of_files:
        raise ValueError(
            'No MEG (fif/ctf) or EEG (edf/bdf/vhdr/set/cnt) files found in the dataset.')

    # Deduplicate split FIF files so we only process the first chunk
    # -----------------------------------------------------------------
    # Some recordings are stored as BIDS splits (e.g., ``_split-01`` and
    # ``_split-02``). MNE stitches them automatically when reading the first
    # part, so we must ignore the later chunks to avoid treating them as
    # separate recordings. We keep only the first path encountered for each
    # base recording and drop the ``split`` entity from the ANCPBIDS artifact
    # to prevent split tags from leaking into derivative filenames.
    filtered_files = []
    filtered_entities = []
    seen_recordings = set()

    for file_path, entity in zip(list_of_files, entities_per_file):
        base_name = os.path.basename(file_path)
        base_root, _ = os.path.splitext(base_name)
        normalized_root = re.sub(r"_split-\d+", "", base_root)

        if normalized_root in seen_recordings:
            continue

        seen_recordings.add(normalized_root)
        filtered_files.append(file_path)

        # Remove split entity when present so downstream derivatives do not
        # include split tags. We guard access to support different artifact
        # representations (dict-like or with an ``entities`` attribute).
        try:
            if hasattr(entity, 'entities') and isinstance(entity.entities, dict):
                entity.entities.pop('split', None)
            if isinstance(entity, dict):
                entity.pop('split', None)
        except Exception:
            # We want to avoid breaking other entity handling; silently ignore
            # any unexpected structure and keep the artifact as-is.
            pass

        filtered_entities.append(entity)

    list_of_files = filtered_files
    entities_per_file = filtered_entities

    # Find if we have crosstalk in list of files and entities_per_file, give notification that they will be skipped:
    # read about crosstalk files here: https://bids-specification.readthedocs.io/en/stable/appendices/meg-file-formats.html
    crosstalk_files = [f for f in list_of_files if 'crosstalk' in f]
    if crosstalk_files:
        print('___MEGqc___: ', 'Crosstalk files found in the list of files. They will be skipped.')

    list_of_files = [f for f in list_of_files if 'crosstalk' not in f]
    entities_per_file = [e for e in entities_per_file if 'crosstalk' not in e['name']]

    # Check if the names in list_of_files and entities_per_file are the same:
    # Support both _meg. and _eeg. suffixes in the name comparison.
    if len(list_of_files) != len(entities_per_file):
        print(f'___MEGqc___: WARNING: list_of_files ({len(list_of_files)}) and '
              f'entities_per_file ({len(entities_per_file)}) have different lengths. '
              'Attempting to reconcile...')
        # Trim to the shorter list to avoid index errors
        min_len = min(len(list_of_files), len(entities_per_file))
        list_of_files = list_of_files[:min_len]
        entities_per_file = entities_per_file[:min_len]

    for i in range(len(list_of_files)):
        base_path = os.path.basename(list_of_files[i])
        ent = entities_per_file[i]
        entity_name = ent['name'] if isinstance(ent, dict) else ent.name
        # Strip modality suffix (_meg|eeg.) and extension for comparison
        file_name_in_path = re.split(r'_(?:meg|eeg)\.', base_path)[0]
        file_name_in_obj = re.split(r'_(?:meg|eeg)\.', entity_name)[0]

        if file_name_in_obj not in file_name_in_path:
            print(f'___MEGqc___: WARNING: Name mismatch at index {i}: '
                  f'file={base_path}, entity={entity_name}. '
                  'Continuing with best-effort matching.')
            # Do NOT raise — allow processing to continue

    # we can also check that final file of path in list of files is same as name in jsons

    return list_of_files, entities_per_file


def create_config_artifact(root_folder, config_file_path: str, f_name_to_save: str, all_taken_raw_files: List[str]):
    """
    Save the config file used for this run as a derivative.

    Note: it is important the config and json to it have the exact same name except the extention!
    The code relies on it later on in add_raw_to_config_json() function.


    Parameters
    ----------
    root_folder : ancpbids folder-like object
        Folder-like node where the ``config`` folder should be created.
    config_file_path : str
        Path to the config file used for this ds conversion
    f_name_to_save : str
        Name of the config file to save.
    all_taken_raw_files : list
        List of all the raw files processed in this run, for this ds.

    """

    # get current time stamp for config file

    timestamp = time.strftime("Date%Y%m%dTime%H%M%S")

    f_name_to_save = f_name_to_save + str(timestamp)

    config_folder = root_folder.create_folder(name='config')
    config_artifact = config_folder.create_artifact()

    config_artifact.content = lambda file_path, cont=config_file_path: shutil.copy(cont, file_path)
    config_artifact.add_entity('desc', f_name_to_save)  # file name
    config_artifact.suffix = 'settings'
    config_artifact.extension = '.ini'

    # Create a seconf json file with config name as key and all taken raw files as value
    # and prepare it to be save as derivative

    config_json = {f_name_to_save: all_taken_raw_files}

    config_json_artifact = config_folder.create_artifact()
    config_json_artifact.content = lambda file_path, cont=config_json: json.dump(cont, open(file_path, 'w'), indent=4)
    config_json_artifact.add_entity('desc', f_name_to_save)  # file name
    config_json_artifact.suffix = 'settings'
    config_json_artifact.extension = '.json'

    return


def ask_user_rerun_subs(reuse_config_file_path: str, sub_list: List[str]):
    """
    Ask the user if he wants to rerun the same subjects again or skip them.

    Parameters
    ----------
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.
    sub_list : list
        List of subjects to run the QC on.

    Returns
    -------
    sub_list : list
        Updated list of subjects to run the QC on.

    """

    # Deprecated helper kept only for API compatibility.
    raise RuntimeError(
        "ask_user_rerun_subs() is deprecated. "
        "Use processed_subjects_policy ('skip'|'rerun'|'fail')."
    )


def _subjects_overlap_from_config(config_file_path: Optional[str], requested_subs: List[str]) -> List[str]:
    """Return requested subjects that were already processed for one config snapshot."""
    if not config_file_path:
        return []
    prev_files, _ = get_list_of_raws_for_config(config_file_path)
    if not prev_files:
        return []
    parsed = []
    for name in prev_files:
        try:
            parsed.append(str(name).split('sub-')[1].split('_')[0])
        except Exception:
            continue
    done_subs = set(parsed)
    return [sub for sub in requested_subs if sub in done_subs]


def _apply_processed_subjects_policy(
    requested_subs: List[str],
    overlap_subs: List[str],
    processed_subjects_policy: str = "skip",
    interactive_prompts: bool = False,
) -> List[str]:
    """Apply explicit policy for already-processed subjects.

    Parameters
    ----------
    requested_subs
        Subjects requested for current run.
    overlap_subs
        Subjects already represented in the selected config snapshot.
    processed_subjects_policy
        One of: ``skip``, ``rerun``, ``fail``.
    interactive_prompts
        Deprecated (kept for API compatibility). Prompt policies were removed.
    """
    policy = str(processed_subjects_policy or "skip").strip().lower()
    if policy not in _PROCESSED_SUBJECT_POLICIES:
        raise ValueError(
            f"Invalid processed_subjects_policy '{processed_subjects_policy}'. Supported: "
            f"{', '.join(sorted(_PROCESSED_SUBJECT_POLICIES))}."
        )
    if not overlap_subs:
        return requested_subs

    print('___MEGqc___: ', 'These requested subjects were already processed before:', overlap_subs)

    if policy == "skip":
        updated = [sub for sub in requested_subs if sub not in set(overlap_subs)]
        print('___MEGqc___: ', 'Subjects to process after skip policy:', updated)
        return updated
    if policy == "rerun":
        print('___MEGqc___: ', 'Rerun policy selected; all requested subjects will be processed.')
        return requested_subs
    if policy == "fail":
        raise RuntimeError(
            "Already-processed subjects detected and processed_subjects_policy='fail'. "
            "Use 'skip' or 'rerun' to continue."
        )

    raise RuntimeError(
        "Prompt-based processed-subject policy is no longer supported. "
        "Use processed_subjects_policy='skip', 'rerun', or 'fail'."
    )


def _write_profile_manifest(
    megqc_root: str,
    *,
    analysis_mode: str,
    analysis_id: Optional[str],
    config_file_path: str,
    existing_config_policy: str,
    processed_subjects_policy: str,
) -> None:
    """Persist lightweight metadata for one profile run.

    This manifest supports future GUI profile introspection and reproducibility
    without scanning every artifact.
    """
    os.makedirs(megqc_root, exist_ok=True)
    cfg_hash = None
    try:
        with open(config_file_path, "rb") as f:
            cfg_hash = hashlib.sha1(f.read()).hexdigest()
    except Exception:
        cfg_hash = None
    manifest = {
        "analysis_mode": analysis_mode,
        "analysis_id": analysis_id,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "config_file_path": config_file_path,
        "config_sha1": cfg_hash,
        "existing_config_policy": existing_config_policy,
        "processed_subjects_policy": processed_subjects_policy,
    }
    # Version is optional at runtime (editable installs may not expose metadata).
    try:
        manifest["meg_qc_version"] = importlib.metadata.version("meg_qc")
    except Exception:
        manifest["meg_qc_version"] = None
    out = os.path.join(megqc_root, "profile_manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def get_list_of_raws_for_config(reuse_config_file_path: str):
    """
    Get the list of all raw files processed with the config file used before.

    Parameters
    ----------
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.

    Returns
    -------
    list_of_files : list
        List of all the raw files processed in this run, for this ds.
    config_desc : str
        Description entity of the config file used before.
    """

    # exchange ini to json:
    json_for_reused_config = reuse_config_file_path.replace('.ini', '.json')

    # check if the json file exists:
    if not os.path.isfile(json_for_reused_config):
        print('___MEGqc___: ',
              'No json file found for the config file used before. Can not add the new raw files to it.')
        return [], None

    print('___MEGqc___: ', 'json_for_reused_config', json_for_reused_config)

    try:
        with open(json_for_reused_config, 'r') as file:
            config_json = json.load(file)
    except json.JSONDecodeError as e:
        with open(json_for_reused_config, 'r') as file:
            content = file.read()
        print(f"Error decoding JSON: {e}")
        print(f"File content:\n{content}")
        # Handle the error appropriately by returning no previously processed files
        return [], None

    # from file name get desc entity to use it as a key in the json file:
    # after desc- and before the underscores:
    file_name = os.path.basename(reuse_config_file_path).split('.')[0]
    config_desc = file_name.split('desc-')[1].split('_')[0]

    # get what files already were in the config file
    list_of_files = config_json.get(config_desc, [])

    return list_of_files, config_desc


def add_raw_to_config_json(root_folder, reuse_config_file_path: str, all_taken_raw_files: List[str]):
    """
    Add the list of all taken raw files to the existing list of used settings in the config file.

    Expects that the config file .ini and the .json file (with the same name) are already saved as derivatives.

    To get corresponding json here use the easy way:
    just exchange ini to json in reuse file path (not using ANCPbids for it).
    The 'proper' way would be to:
    - query the desc entitiy of the reused config file
    - get the json file with the same desc entity
    This way will still assume that desc are exactly the same, so we use the easy way without ANCPbids d-tour.

    The function will also output the updated list of all taken raw files for this ds based on the users choice:
    rewrite or not the subjects that have already been processed with this config file.

    Parameters
    ----------
    root_folder : ancpbids folder-like object
        Folder-like node where the ``config`` folder should be created.
    reuse_config_file_path : str
        Path to the config file used for this ds conversion before.
    all_taken_raw_files : list
        List of all the raw files processed in this run, for this ds.

    Returns
    -------
    all_taken_raw_files : list
        Updated list of all the raw files processed in this run, for this ds.

    """

    list_of_files, config_desc = get_list_of_raws_for_config(reuse_config_file_path)

    # Continue to update the list with new files:
    list_of_files += all_taken_raw_files

    # sort and remove duplicates:
    list_of_files = sorted(list(set(list_of_files)))

    # overwrite the old json (premake ancp bids artifact):
    config_json = {config_desc: list_of_files}

    config_folder = root_folder.create_folder(name='config')
    # TODO: we dont need to create config folder again, already got it, how to get it?

    config_json_artifact = config_folder.create_artifact()
    config_json_artifact.content = lambda file_path, cont=config_json: json.dump(cont, open(file_path, 'w'), indent=4)
    config_json_artifact.add_entity('desc', config_desc)  # file name
    config_json_artifact.suffix = 'settings'
    config_json_artifact.extension = '.json'

    return all_taken_raw_files


def check_ds_paths(ds_paths: Union[List[str], str]):
    """
    Check if the given paths to the data sets exist.

    Parameters
    ----------
    ds_paths : list or str
        List of paths to the BIDS-conform data sets to run the QC on.

    Returns
    -------
    ds_paths : list
        List of paths to the BIDS-conform data sets to run the QC on.
    """

    # has to be a list, even if there is just one path:
    if isinstance(ds_paths, str):
        ds_paths = [ds_paths]

    # make sure all directories in the list exist:
    for ds_path in ds_paths:
        if not os.path.isdir(ds_path):
            raise ValueError(f'Given path to the dataset does not exist. Path: {ds_path}')

    return ds_paths


def check_config_saved_ask_user(dataset):
    """
    Check if there is already config file used for this ds:
    If yes - ask the user if he wants to use it again. If not - use default one.
    When no config found or user doesnt want to reuse - will return None.
    otherwise will return the path to one config file used for this ds before to reuse now.

    Parameters
    ----------
    dataset : ancpbids.Dataset
        Dataset object to work with.

    Returns
    -------
    config_file_path : str
        Path to the config file used for this ds conversion.
    """

    # if os.path.isfile(os.path.join(derivatives_path, 'config', 'UsedSettings.ini')):
    #     print('___MEGqc___: ', 'There is already a config file used for this data set. Do you want to use it again?')
    #     #ask user if he wants to use the same config file again

    try:
        entities = query_entities(dataset, scope='derivatives')
    except TypeError:
        # ``ancpbids.query.query_entities`` relies on ``query`` returning an iterable.
        # On Windows, ``query`` can return ``None`` when the derivatives folder does not
        # exist yet, raising a ``TypeError`` when ``query_entities`` tries to iterate over
        # the result.  In that situation there are no previous config files to reuse, so
        # we can safely treat the entity mapping as empty.
        entities = {}
    else:
        entities = entities or {}

    # print('___MEGqc___: ', 'entities', entities)

    # search if there is already a derivative with 'UsedSettings' in the name
    # if yes - ask the user if he wants to use it again. If not - use default one.
    used_settings_entity_list = []
    for key, entity_set in entities.items():
        if key == 'description':
            for ent in entity_set:
                if 'usedsettings' in ent.lower():
                    used_settings_entity_list.append(ent)

    used_setting_file_list = []
    for used_settings_entity in used_settings_entity_list:
        used_setting_file_list += sorted(list(
            dataset.query(suffix='meg', extension='.ini', desc=used_settings_entity, return_type='filename',
                          scope='derivatives')))

    reuse_config_file_path = None

    # Deprecated helper kept only for API compatibility.
    if used_setting_file_list:
        print(
            "___MEGqc___: Prompt-based config reuse is deprecated. "
            "Use existing_config_policy instead."
        )
    return None


def check_sub_list(sub_list: Union[List[str], str], dataset):
    """
    Check if the given subjects are in the data set.

    Parameters
    ----------
    sub_list : list or str
        List of subjects to run the QC on.
    dataset : ancpbids.Dataset
        Dataset object to work with.

    Returns
    -------
    sub_list : list
        Updated list of subjects to run the QC on.

    """

    available_subs = sorted(list(dataset.query_entities(scope='raw')['subject']))
    if sub_list == 'all':
        sub_list = available_subs
    elif isinstance(sub_list, str) and sub_list != 'all':
        sub_list = [sub_list]
        # check if this sub is available:
        if sub_list[0] not in available_subs:
            print('___MEGqc___: ',
                  'The subject you want to run the QC on is not in your data set. Check the subject ID.')
            return
    elif isinstance(sub_list, list):
        # if they are given as str - IDs:
        if all(isinstance(sub, str) for sub in sub_list):
            sub_list_missing = [sub for sub in sub_list if sub not in available_subs]
            sub_list = [sub for sub in sub_list if sub in available_subs]
            if sub_list_missing:
                print('___MEGqc___: ', 'Could NOT find these subs in your data set. Check the subject IDs:',
                      sub_list_missing)
                print('___MEGqc___: ', 'Requested subjects found in your data set:', sub_list,
                      'Only these subjects will be processed.')

        # if they are given as int - indexes:
        elif all(isinstance(sub, int) for sub in sub_list):
            sub_list = [available_subs[i] for i in sub_list]

    print('___MEGqc___: ', 'Requested sub_list to process: ', sub_list)

    return sub_list


def _run_metric_safe(metric_name: str, func, *args, n_outputs: int, empty_outputs: list, **kwargs):
    """Call *func* with *args*/*kwargs* and catch any exception.

    Parameters
    ----------
    metric_name : str
        Human-readable label used in log messages and error records.
    func : callable
        The metric function to call.
    *args :
        Positional arguments forwarded to *func*.
    n_outputs : int
        Expected number of return values from *func* (used for safety).
    empty_outputs : list
        Fallback values to return when *func* raises, one per return value.
        Must have length == *n_outputs*.
    **kwargs :
        Keyword arguments forwarded to *func*.

    Returns
    -------
    tuple
        ``(*outputs, error_info)`` where *outputs* are the metric results
        (or the empty fallbacks on failure) and *error_info* is ``None``
        on success or a dict with keys ``error_type``, ``error_message``,
        ``traceback``, and ``timestamp`` on failure.
    """
    import traceback as _traceback
    try:
        result = func(*args, **kwargs)
        # Normalise single-value returns to a tuple so the caller can always unpack
        if not isinstance(result, tuple):
            result = (result,)
        return (*result, None)
    except Exception as exc:
        tb_str = _traceback.format_exc()
        error_info = {
            "metric": metric_name,
            "error_type": type(exc).__qualname__,
            "error_message": str(exc),
            "traceback": tb_str,
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        print(
            f"___MEGqc___: Metric '{metric_name}' FAILED — subject will continue "
            f"without this metric.\n  {type(exc).__name__}: {exc}\n{tb_str}"
        )
        return (*empty_outputs, error_info)


def process_one_subject(
        sub: str,
        dataset,
        dataset_path: str,
        all_qc_params: dict,
        internal_qc_params: dict,
        derivatives_root: str,
        output_root: str,
        analysis_segments: Optional[List[str]] = None,
):
    """
    This function processes a single subject. It contains all the code that was
    originally inside the 'for sub in sub_list:' loop in 'make_derivative_meg_qc'.

    Parameters
    ----------
    sub : str
        Single subject ID string (e.g. '009').
    dataset : ancpbids.dataset
        BIDS-conform dataset loaded by ancpbids.
    dataset_path : str
        Path to the BIDS dataset.
    all_qc_params : dict
        QC parameters from user config file.
    internal_qc_params : dict
        Internal QC parameters that users do not change.
    derivatives_root : str
        Path to the derivatives directory where outputs should be written.
    output_root : str
        Base directory used when persisting derivatives (parent of the
        derivatives folder), allowing redirection outside the BIDS dataset.
    analysis_segments : list of str, optional
        Extra folder segments inserted between ``Meg_QC`` and ``calculation``.
        In legacy mode this is ``[]``; in profile mode this is
        ``["profiles", <analysis_id>]``.
    """

    # We replicate everything that was inside the loop.

    # CREATE DERIVATIVE FOR THIS SUBJECT
    derivative = dataset.create_derivative(name="Meg_QC")
    _ensure_derivative_dataset_description_filename(derivative)
    derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"

    print('___MEGqc___: ', 'Take SUB: ', sub)

    analysis_segments = analysis_segments or []
    # Keep derivative naming stable (Meg_QC) and insert profile folder(s)
    # underneath it so existing desc/suffix parsing remains unchanged.
    root_folder = derivative
    for seg in analysis_segments:
        root_folder = root_folder.create_folder(name=str(seg))

    calculation_folder = root_folder.create_folder(name='calculation')
    # subject_folder is created per-file inside the loop, under a modality
    # subfolder (meg/ or eeg/) to avoid filename collisions between modalities.
    _modality_subject_folders = {}  # cache: modality -> subject_folder

    # GET FILE LIST FOR THIS SUBJECT
    list_of_files, entities_per_file = get_files_list(
        sub, dataset_path, dataset,
        m_or_g_chosen=all_qc_params.get('default', {}).get('m_or_g_chosen')
    )

    if not list_of_files:
        print('___MEGqc___: ',
              'No files to work on. Check that given subjects are present in your data set.')
        return  # Stop if no files exist for this subject

    print('___MEGqc___: ', 'list_of_files to process:', list_of_files)
    print('___MEGqc___: ', 'entities_per_file to process:', entities_per_file)
    print('___MEGqc___: ', 'TOTAL files to process: ', len(list_of_files))

    # Keep track of all raw files processed for this subject (optional)
    all_taken_raw_files = [os.path.basename(f) for f in list_of_files]

    # Preassign in case nothing is processed
    raw = None

    # Counters, accumulators
    counter = 0
    avg_ecg = []
    avg_eog = []

    # LOOP OVER FIF FILES FOR THIS SUBJECT
    for file_ind, data_file in enumerate(list_of_files):  # e.g. [0:1] in your example

        print('___MEGqc___: ', 'Processing file: ', data_file)

        # Preassign strings with notes for the user (just as in your code)
        shielding_str, m_or_g_skipped_str, epoching_str = '', '', ''
        ecg_str, eog_str, head_str, muscle_str = '', '', '', ''
        pp_manual_str, pp_auto_str, std_str, psd_str = '', '', '', ''

        print('___MEGqc___: ', 'Starting initial processing...')
        start_time = time.time()

        # INITIAL PROCESSING
        (meg_system,
         dict_epochs_mg,
         chs_by_lobe,
         channels,
         raw_cropped_filtered,
         raw_cropped_filtered_resampled,
         raw_cropped,
         raw,
         info_derivs,
         stim_deriv,
         event_summary_deriv,
         shielding_str,
         epoching_str,
         sensors_derivs,
         m_or_g_chosen,
         m_or_g_skipped_str,
         lobes_color_coding_str,
         resample_str) = initial_processing(
            default_settings=all_qc_params['default'],
            filtering_settings=all_qc_params['Filtering'],
            epoching_params=all_qc_params['Epoching'],
            file_path=data_file,
            derivatives_root=derivatives_root,
            eeg_settings=all_qc_params.get('EEG_settings'),
        )

        print('___MEGqc___: ',
              "Finished initial processing. --- Execution %s seconds ---"
              % (time.time() - start_time))

        # PREDEFINE VARIABLES FOR QC
        noisy_freqs_global = None
        std_derivs, psd_derivs = [], []
        pp_manual_derivs, pp_auto_derivs = [], []
        ecg_derivs, eog_derivs = [], []
        head_derivs, muscle_derivs = [], []
        simple_metrics_psd, simple_metrics_std = [], []
        simple_metrics_pp_manual, simple_metrics_pp_auto = [], []
        simple_metrics_ecg, simple_metrics_eog = [], []
        simple_metrics_head, simple_metrics_muscle = [], []

        # Collects per-metric error dicts for this file; injected into the
        # report so users can see exactly which metric failed and why.
        metric_errors = []

        # 1) STD
        if all_qc_params['default']['run_STD'] is True:
            print('___MEGqc___: ', 'Starting STD...')
            start_time = time.time()
            std_derivs, simple_metrics_std, std_str, _err = _run_metric_safe(
                'STD',
                STD_meg_qc,
                all_qc_params['STD'],
                channels,
                chs_by_lobe,
                dict_epochs_mg,
                raw_cropped_filtered_resampled,
                m_or_g_chosen,
                n_outputs=3,
                empty_outputs=[[], [], '⚠ STD metric failed — see excluded_subjects_errors.json for details.'],
            )
            if _err:
                metric_errors.append(_err)
                std_str = f"⚠ STD metric failed: {_err['error_type']}: {_err['error_message']}"
            print('___MEGqc___: ',
                  "Finished STD. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 2) PSD
        if all_qc_params['default']['run_PSD'] is True:
            print('___MEGqc___: ', 'Starting PSD...')
            start_time = time.time()
            psd_derivs, simple_metrics_psd, psd_str, noisy_freqs_global, _err = _run_metric_safe(
                'PSD',
                PSD_meg_qc,
                all_qc_params['PSD'],
                internal_qc_params['PSD'],
                channels,
                chs_by_lobe,
                raw_cropped_filtered,
                m_or_g_chosen,
                n_outputs=4,
                empty_outputs=[[], [], '⚠ PSD metric failed — see excluded_subjects_errors.json for details.', None],
                helper_plots=False,
            )
            if _err:
                metric_errors.append(_err)
                psd_str = f"⚠ PSD metric failed: {_err['error_type']}: {_err['error_message']}"
            print('___MEGqc___: ',
                  "Finished PSD. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 3) Peak‑to‑Peak manual
        if all_qc_params['default']['run_PTP_manual'] is True:
            start_time = time.time()

            # choose the implementation ----------------------------------
            if all_qc_params['PTP_manual']['numba_version'] is True:
                print('___MEGqc___: ', 'Starting Peak‑to‑Peak manual (Numba)...')
                func = PP_manual_meg_qc_numba  #  accelerated version
            else:
                print('___MEGqc___: ', 'Starting Peak‑to‑Peak manual...')
                func = PP_manual_meg_qc  # standard version
            # -------------------------------------------------------------

            pp_manual_derivs, simple_metrics_pp_manual, pp_manual_str, _err = _run_metric_safe(
                'PTP_manual',
                func,
                all_qc_params['PTP_manual'],
                channels,
                chs_by_lobe,
                dict_epochs_mg,
                raw_cropped_filtered_resampled,
                m_or_g_chosen,
                n_outputs=3,
                empty_outputs=[[], [], '⚠ PTP manual metric failed — see excluded_subjects_errors.json for details.'],
            )
            if _err:
                metric_errors.append(_err)
                pp_manual_str = f"⚠ PTP manual metric failed: {_err['error_type']}: {_err['error_message']}"

            print('___MEGqc___: ',
                  "Finished Peak‑to‑Peak manual. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 4) Peak-to-Peak auto from MNE
        if all_qc_params['default']['run_PTP_auto_mne'] is True:
            print('___MEGqc___: ', 'Starting Peak-to-Peak auto...')
            start_time = time.time()
            pp_auto_derivs, bad_channels, pp_auto_str, _err = _run_metric_safe(
                'PTP_auto',
                PP_auto_meg_qc,
                all_qc_params['PTP_auto'],
                channels,
                raw_cropped_filtered_resampled,
                m_or_g_chosen,
                n_outputs=3,
                empty_outputs=[[], [], '⚠ PTP auto metric failed — see excluded_subjects_errors.json for details.'],
            )
            if _err:
                metric_errors.append(_err)
                pp_auto_str = f"⚠ PTP auto metric failed: {_err['error_type']}: {_err['error_message']}"
            print('___MEGqc___: ',
                  "Finished Peak-to-Peak auto. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # 5) ECG
        if all_qc_params['default']['run_ECG'] is True:
            print('___MEGqc___: ', 'Starting ECG...')
            start_time = time.time()
            ecg_derivs, simple_metrics_ecg, ecg_str, avg_objects_ecg, _err = _run_metric_safe(
                'ECG',
                ECG_meg_qc,
                all_qc_params['ECG'],
                internal_qc_params['ECG'],
                raw_cropped,
                channels,
                chs_by_lobe,
                m_or_g_chosen,
                n_outputs=4,
                empty_outputs=[[], [], '⚠ ECG metric failed — see excluded_subjects_errors.json for details.', []],
            )
            if _err:
                metric_errors.append(_err)
                ecg_str = f"⚠ ECG metric failed: {_err['error_type']}: {_err['error_message']}"
            else:
                avg_ecg += avg_objects_ecg
            print('___MEGqc___: ',
                  "Finished ECG. --- Execution %s seconds ---"
                  % (time.time() - start_time))


        # 6) EOG
        if all_qc_params['default']['run_EOG'] is True:
            print('___MEGqc___: ', 'Starting EOG...')
            start_time = time.time()
            eog_derivs, simple_metrics_eog, eog_str, avg_objects_eog, _err = _run_metric_safe(
                'EOG',
                EOG_meg_qc,
                all_qc_params['EOG'],
                internal_qc_params['EOG'],
                raw_cropped,
                channels,
                chs_by_lobe,
                m_or_g_chosen,
                n_outputs=4,
                empty_outputs=[[], [], '⚠ EOG metric failed — see excluded_subjects_errors.json for details.', []],
            )
            if _err:
                metric_errors.append(_err)
                eog_str = f"⚠ EOG metric failed: {_err['error_type']}: {_err['error_message']}"
            else:
                avg_eog += avg_objects_eog
            print('___MEGqc___: ',
                  "Finished EOG. --- Execution %s seconds ---"
                  % (time.time() - start_time))


        # 7) Head movement artifacts
        if all_qc_params['default']['run_Head'] is True and meg_system != 'EEG':
            print('___MEGqc___: ', 'Starting Head movement calculation...')
            start_time = time.time()
            head_derivs, simple_metrics_head, head_str, df_head_pos, head_pos, _err = _run_metric_safe(
                'Head',
                HEAD_movement_meg_qc,
                raw_cropped,
                n_outputs=5,
                empty_outputs=[[], [], '⚠ Head metric failed — see excluded_subjects_errors.json for details.', None, None],
            )
            if _err:
                metric_errors.append(_err)
                head_str = f"⚠ Head metric failed: {_err['error_type']}: {_err['error_message']}"
            print('___MEGqc___: ',
                  "Finished Head movement calculation. --- Execution %s seconds ---"
                  % (time.time() - start_time))
        elif all_qc_params['default']['run_Head'] is True and meg_system == 'EEG':
            head_str = 'Head motion metric is not available for EEG data (requires MEG cHPI coils).'
            print(f'___MEGqc___: {head_str}')

        # 8) Muscle artifacts
        if all_qc_params['default']['run_Muscle'] is True:
            print('___MEGqc___: ', 'Starting Muscle artifacts calculation...')
            start_time = time.time()
            muscle_derivs, simple_metrics_muscle, muscle_str, scores_muscle_all3, _err = _run_metric_safe(
                'Muscle',
                MUSCLE_meg_qc,
                all_qc_params['Muscle'],
                all_qc_params['PSD'],
                internal_qc_params['PSD'],
                channels,
                raw_cropped_filtered,
                noisy_freqs_global,
                m_or_g_chosen,
                derivatives_root,
                n_outputs=4,
                empty_outputs=[[], [], '⚠ Muscle metric failed — see excluded_subjects_errors.json for details.', None],
                attach_dummy=True,
                cut_dummy=True,
            )
            if _err:
                metric_errors.append(_err)
                muscle_str = f"⚠ Muscle metric failed: {_err['error_type']}: {_err['error_message']}"
            else:
                # Store the total number of events analyzed so we can later express
                # the number of detected artifacts as a percentage.  The first
                # derivative contains a TSV table where each row corresponds to one
                # event that was evaluated during muscle detection.
                if muscle_derivs:
                    total_events_for_muscle = muscle_derivs[0].content.shape[0]
                    simple_metrics_muscle["total_number_of_events"] = int(
                        total_events_for_muscle
                    )
            print('___MEGqc___: ',
                  "Finished Muscle artifacts calculation. --- Execution %s seconds ---"
                  % (time.time() - start_time))

        # REPORT STRINGS
        report_strings = {
            'INITIAL_INFO': (m_or_g_skipped_str + resample_str + epoching_str +
                             shielding_str + lobes_color_coding_str),
            'STD': std_str,
            'PSD': psd_str,
            'PTP_MANUAL': pp_manual_str,
            'PTP_AUTO': pp_auto_str,
            'ECG': ecg_str,
            'EOG': eog_str,
            'HEAD': head_str,
            'MUSCLE': muscle_str,
            'STIMULUS': epoching_str + ('<p>If the data was cropped for this calculation, '
                        'the stimulus data is also cropped.</p>'
                        if all_qc_params['default'].get('crop_tmax') else '')
        }

        # Embed per-metric error summaries into the report strings so that
        # the HTML report clearly communicates which metrics failed and why,
        # even when the subject overall succeeded (partial failure scenario).
        if metric_errors:
            failed_names = ', '.join(e['metric'] for e in metric_errors)
            report_strings['METRIC_ERRORS'] = (
                f"⚠ {len(metric_errors)} metric(s) failed for this recording "
                f"({failed_names}). The subject report is partial. "
                f"Full tracebacks are in excluded_subjects_errors.json."
            )

        report_str_derivs = [QC_derivative(report_strings, 'ReportStrings', 'json')]

        # ORGANIZE QC DERIVATIVES
        QC_derivs = {
            'Raw info': info_derivs,
            'Stimulus channels': stim_deriv,
            'Event summary': event_summary_deriv,
            'Report_strings': report_str_derivs,
            'Sensors locations': sensors_derivs,
            'Standard deviation of the data': std_derivs,
            'Frequency spectrum': psd_derivs,
            'Peak-to-Peak manual': pp_manual_derivs,
            'Peak-to-Peak auto from MNE': pp_auto_derivs,
            'ECG': ecg_derivs,
            'EOG': eog_derivs,
            'Head movement artifacts': head_derivs,
            'High frequency (Muscle) artifacts': muscle_derivs
        }

        QC_simple = {
            'STD': simple_metrics_std,
            'PSD': simple_metrics_psd,
            'PTP_MANUAL': simple_metrics_pp_manual,
            'PTP_AUTO': simple_metrics_pp_auto,
            'ECG': simple_metrics_ecg,
            'EOG': simple_metrics_eog,
            'HEAD': simple_metrics_head,
            'MUSCLE': simple_metrics_muscle
        }

        QC_derivs['Simple_metrics'] = [QC_derivative(QC_simple, 'SimpleMetrics', 'json')]

        # SAVE DERIVATIVES (EXCEPT MATPLOTLIB, PLOTLY, REPORT)
        # Determine the BIDS suffix based on the modality of the current file
        _bids_suffix = 'eeg' if meg_system == 'EEG' else 'meg'

        # Create (or reuse) a modality subfolder: calculation/meg/sub-XXX or
        # calculation/eeg/sub-XXX to keep MEG and EEG derivatives separate.
        if _bids_suffix not in _modality_subject_folders:
            modality_folder = calculation_folder.create_folder(name=_bids_suffix)
            _modality_subject_folders[_bids_suffix] = modality_folder.create_folder(
                type_=dataset.get_schema().Subject,
                name='sub-' + sub,
            )
        subject_folder = _modality_subject_folders[_bids_suffix]

        for section in (sec for sec in QC_derivs.values() if sec):
            for deriv in (
                    d for d in section
                    if d.content_type not in ['matplotlib', 'plotly', 'report']
            ):
                meg_artifact = subject_folder.create_artifact(raw=entities_per_file[file_ind])
                counter += 1
                print('___MEGqc___: ', 'counter of subject_folder.create_artifact', counter)

                meg_artifact.add_entity('desc', deriv.name)  # file name
                meg_artifact.suffix = _bids_suffix
                meg_artifact.extension = '.html'

                if deriv.content_type == 'df':
                    meg_artifact.extension = '.tsv'
                    # Write the index only when it carries meaningful data
                    # (e.g. channel names in epoch matrices).  A plain
                    # RangeIndex is just row numbers and creates an
                    # unwanted ``Unnamed: 0`` column on re-read.
                    def _df_writer(file_path, cont=deriv.content):
                        import pandas as _pd
                        write_idx = not isinstance(cont.index, _pd.RangeIndex)
                        cont.to_csv(file_path, sep='\t', index=write_idx)
                    meg_artifact.content = _df_writer

                elif deriv.content_type == 'json':
                    meg_artifact.extension = '.json'

                    def json_writer(file_path, cont=deriv.content):
                        with open(file_path, "w") as file_wrapper:
                            json.dump(cont, file_wrapper, indent=4)

                    meg_artifact.content = json_writer

                elif deriv.content_type == 'info':
                    meg_artifact.extension = '.fif'
                    meg_artifact.content = lambda file_path, cont=deriv.content: mne.io.write_info(
                        file_path, cont
                    )
                else:
                    print('___MEGqc___: ', meg_artifact.name)
                    meg_artifact.content = 'dummy text'
                    meg_artifact.extension = '.txt'

        # CLEAN UP TEMP FILES
        try:
            remove_fif_and_splits(raw_cropped)
            remove_fif_and_splits(raw_cropped_filtered)
            remove_fif_and_splits(raw_cropped_filtered_resampled)

            # Delete heavy objects to free memory between files.
            # Use a safe approach: delete only variables that exist.
            for _varname in ('meg_system', 'dict_epochs_mg', 'chs_by_lobe',
                             'channels', 'raw_cropped_filtered',
                             'raw_cropped_filtered_resampled', 'raw_cropped',
                             'info_derivs', 'stim_deriv', 'event_summary_deriv',
                             'shielding_str', 'epoching_str', 'sensors_derivs',
                             'm_or_g_chosen', 'm_or_g_skipped_str',
                             'lobes_color_coding_str', 'resample_str'):
                locals().pop(_varname, None)
            gc.collect()
            print('REMOVING TRASH: SUCCEEDED')
        except Exception:
            print('REMOVING TRASH: FAILED')

    # WRITE DERIVATIVE
    with temporary_dataset_base(dataset, output_root):
        ancpbids.write_derivative(dataset, derivative)

    # Removes intermediate trash objects — guard against NameError when no
    # files were processed (meg_artifact would never have been assigned).
    try:
        del meg_artifact
    except NameError:
        pass
    del derivative
    gc.collect()

    # Check if raw is None => means we never processed a file
    try:
        if raw is None:
            print('___MEGqc___: ', 'No data files could be processed for subject:', sub)
            return None, metric_errors
    except Exception:
        print('___MEGqc___: ', 'No data files could be processed for subject:', sub)

    # Return raw file list plus any per-metric errors collected during processing.
    # metric_errors is [] when all metrics succeeded.
    return all_taken_raw_files, metric_errors


def process_one_subject_safe(
        sub: str,
        dataset,
        dataset_path: str,
        all_qc_params: dict,
        internal_qc_params: dict,
        derivatives_root: str,
        output_root: str,
        analysis_segments: Optional[List[str]] = None):
    """Wrapper around :func:`process_one_subject` that catches errors.

    Parameters are identical to :func:`process_one_subject`.

    Returns
    -------
    tuple
        ``(sub, files, error_info)`` where:

        * ``files`` is the list of processed raw filenames on success, or
          ``None`` when the subject failed entirely (initial processing or
          an unexpected error).
        * ``error_info`` is ``None`` when the subject completed without any
          issue.  When the subject **failed entirely**, it is a dict with
          keys ``error_type``, ``error_message``, ``traceback``, and
          ``timestamp``.  When the subject **completed partially** (one or
          more metrics failed but overall processing continued), it is a dict
          with key ``metric_errors`` containing the list of per-metric error
          dicts — each with keys ``metric``, ``error_type``,
          ``error_message``, ``traceback``, and ``timestamp``.
    """
    import traceback as _traceback
    try:
        result = process_one_subject(
            sub=sub,
            dataset=dataset,
            dataset_path=dataset_path,
            all_qc_params=all_qc_params,
            internal_qc_params=internal_qc_params,
            derivatives_root=derivatives_root,
            output_root=output_root,
            analysis_segments=analysis_segments,
        )
        # process_one_subject now returns (files, metric_errors)
        if result is None:
            files, metric_errors = None, []
        else:
            files, metric_errors = result

        if metric_errors:
            # Subject succeeded overall but some metrics failed — report as
            # partial failure so the caller can write it to the error log.
            failed_names = ', '.join(e['metric'] for e in metric_errors)
            print(
                f"___MEGqc___: Subject {sub} completed with "
                f"{len(metric_errors)} metric failure(s): {failed_names}"
            )
            error_info = {"metric_errors": metric_errors}
        else:
            error_info = None

        return sub, files, error_info

    except Exception as e:  # Catch any error so the parallel job continues
        tb_str = _traceback.format_exc()
        error_info = {
            "error_type": type(e).__qualname__,
            "error_message": str(e),
            "traceback": tb_str,
            "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        print(
            f"___MEGqc___: Error processing subject {sub}:\n"
            f"  {type(e).__name__}: {e}\n"
            f"{tb_str}"
        )
        return sub, None, error_info


def _parse_count_percent(val: str):
    """Return ``(count, percent)`` from strings like ``"10 (5.0%)"``."""
    if not isinstance(val, str):
        return val, None
    try:
        if "(" in val and "%" in val:
            count_str, rest = val.split("(", 1)
            count = float(count_str.strip())
            percent = float(rest.strip().strip(")% "))
            return count, percent
        if val.endswith("%"):
            return None, float(val.strip("%"))
        return float(val), None
    except Exception:
        return None, None


def _parse_percent(val: str):
    """Parse a percentage string like ``"10.5%"``."""
    if isinstance(val, str):
        try:
            return float(val.strip().strip("%"))
        except Exception:
            return None
    return float(val)


def flatten_summary_metrics(js: dict) -> dict:
    """Flatten one GlobalSummaryReport JSON into numeric columns."""
    row = {}
    if js.get("GQI") is not None:
        row["GQI"] = js.get("GQI")

    def _sensor_key(sensor_type_str):
        """Map human-readable sensor names to short keys."""
        if sensor_type_str == "MAGNETOMETERS":
            return "mag"
        elif sensor_type_str == "GRADIOMETERS":
            return "grad"
        elif sensor_type_str == "EEG CHANNELS":
            return "eeg"
        return sensor_type_str.lower().replace(" ", "_")

    for item in js.get("STD_time_series", []):
        metric = item.get("Metric", "").replace(" ", "_").lower()
        for col_name, short in [("MAGNETOMETERS", "mag"), ("GRADIOMETERS", "grad"), ("EEG CHANNELS", "eeg")]:
            num, pct = _parse_count_percent(item.get(col_name, ""))
            row[f"STD_ts_{metric}_{short}_num"] = num
            row[f"STD_ts_{metric}_{short}_percentage"] = pct

    for item in js.get("PTP_time_series", []):
        metric = item.get("Metric", "").replace(" ", "_").lower()
        for col_name, short in [("MAGNETOMETERS", "mag"), ("GRADIOMETERS", "grad"), ("EEG CHANNELS", "eeg")]:
            num, pct = _parse_count_percent(item.get(col_name, ""))
            row[f"PTP_ts_{metric}_{short}_num"] = num
            row[f"PTP_ts_{metric}_{short}_percentage"] = pct

    for item in js.get("STD_epoch_summary", []):
        sensor = _sensor_key(item.get("Sensor Type", ""))
        num_noisy, pct_noisy = _parse_count_percent(item.get("Noisy Epochs", ""))
        num_flat, pct_flat = _parse_count_percent(item.get("Flat Epochs", ""))
        row[f"STD_ep_{sensor}_noisy_num"] = num_noisy
        row[f"STD_ep_{sensor}_noisy_percentage"] = pct_noisy
        row[f"STD_ep_{sensor}_flat_num"] = num_flat
        row[f"STD_ep_{sensor}_flat_percentage"] = pct_flat

    for item in js.get("PTP_epoch_summary", []):
        sensor = _sensor_key(item.get("Sensor Type", ""))
        num_noisy, pct_noisy = _parse_count_percent(item.get("Noisy Epochs", ""))
        num_flat, pct_flat = _parse_count_percent(item.get("Flat Epochs", ""))
        row[f"PTP_ep_{sensor}_noisy_num"] = num_noisy
        row[f"PTP_ep_{sensor}_noisy_percentage"] = pct_noisy
        row[f"PTP_ep_{sensor}_flat_num"] = num_flat
        row[f"PTP_ep_{sensor}_flat_percentage"] = pct_flat

    for item in js.get("ECG_correlation_summary", []):
        sensor = _sensor_key(item.get("Sensor Type", ""))
        num, pct = _parse_count_percent(item.get("# |High Correlations| > 0.8", ""))
        total = item.get("Total Channels")
        row[f"ECG_{sensor}_high_corr_num"] = num
        row[f"ECG_{sensor}_high_corr_percentage"] = pct
        row[f"ECG_{sensor}_total_channels"] = total

    for item in js.get("EOG_correlation_summary", []):
        sensor = _sensor_key(item.get("Sensor Type", ""))
        num, pct = _parse_count_percent(item.get("# |High Correlations| > 0.8", ""))
        total = item.get("Total Channels")
        row[f"EOG_{sensor}_high_corr_num"] = num
        row[f"EOG_{sensor}_high_corr_percentage"] = pct
        row[f"EOG_{sensor}_total_channels"] = total

    for item in js.get("PSD_noise_summary", []):
        row["PSD_noise_mag_percentage"] = _parse_percent(item.get("MAGNETOMETERS", "0"))
        row["PSD_noise_grad_percentage"] = _parse_percent(item.get("GRADIOMETERS", "0"))

    muscle = js.get("Muscle_events", {})
    row["Muscle_events_num"] = muscle.get("# Muscle Events")
    row["Muscle_events_total"] = muscle.get("total_number_of_events")

    for key, val in js.get("GQI_penalties", {}).items():
        row[f"GQI_penalty_{key}"] = val

    for key, val in js.get("GQI_metrics", {}).items():
        row[f"GQI_{key}"] = val

    for key, val in js.get("parameters", {}).items():
        row[f"param_{key}"] = val

    return row


def make_derivative_meg_qc(
        default_config_file_path: str,
        internal_config_file_path: str,
        ds_paths: Union[List[str], str],
        sub_list: Union[List[str], str] = 'all',
        n_jobs: int = 5,  # Number of parallel jobs
        derivatives_base: Optional[str] = None,
        analysis_mode: str = "legacy",
        analysis_id: Optional[str] = None,
        existing_config_policy: str = "provided",
        processed_subjects_policy: str = "skip",
        interactive_prompts: bool = False,
        keep_temp_on_error: bool = False,
):
    """Run MEGqc calculation for one or more datasets.

    Parameters
    ----------
    analysis_mode
        ``legacy`` writes to ``derivatives/Meg_QC``.
        ``new``/``reuse``/``latest`` write under
        ``derivatives/Meg_QC/profiles/<analysis_id>``.
    analysis_id
        Profile identifier used with ``new`` or ``reuse``.
    existing_config_policy
        Policy for choosing config when snapshots already exist in profile.
    processed_subjects_policy
        Policy when requested subjects already appear in selected snapshot.
    interactive_prompts
        Deprecated (kept for API compatibility). Prompt policies are disabled.
    keep_temp_on_error
        If ``True``, temporary preprocessed FIF files are kept when an exception
        occurs while processing a dataset to aid debugging.
    """
    start_time = time.time()

    ds_paths = check_ds_paths(ds_paths)
    internal_qc_params = get_internal_config_params(internal_config_file_path)

    requested_sub_list = sub_list

    for dataset_path in ds_paths:
        print('___MEGqc___: ', 'DS path:', dataset_path)
        dataset = ancpbids.load_dataset(dataset_path, DatasetOptions(lazy_loading=True))
        (
            output_root,
            derivatives_root,
            megqc_root,
            resolved_analysis_id,
            analysis_segments,
        ) = resolve_analysis_root(
            dataset_path=dataset_path,
            external_derivatives_root=derivatives_base,
            analysis_mode=analysis_mode,
            analysis_id=analysis_id,
            create_if_missing=True,
        )
        config_dir = _config_dir_from_megqc_root(megqc_root)
        os.makedirs(config_dir, exist_ok=True)

        # Ensure temporary intermediates are cleaned deterministically. On
        # failures, keep_temp_on_error allows retaining them for debugging.
        dataset_succeeded = False
        try:
            config_file_path = _resolve_config_by_policy(
                default_config_file_path=default_config_file_path,
                config_dir=config_dir,
                existing_config_policy=existing_config_policy,
                interactive_prompts=interactive_prompts,
            )
            print('___MEGqc___: ', 'Using config file: ', config_file_path)

            all_qc_params = get_all_config_params(config_file_path)
            if all_qc_params is None:
                return

            # Determine which subjects to run
            dataset_sub_list = check_sub_list(requested_sub_list, dataset)
            if dataset_sub_list is None:
                print('___MEGqc___: ', 'No valid subjects to process for this dataset.')
                continue
            overlap = _subjects_overlap_from_config(config_file_path, dataset_sub_list)
            dataset_sub_list = _apply_processed_subjects_policy(
                requested_subs=dataset_sub_list,
                overlap_subs=overlap,
                processed_subjects_policy=processed_subjects_policy,
                interactive_prompts=interactive_prompts,
            )

            # Parallel execution over subjects
            # Each subject is processed by process_one_subject_safe() in parallel
            # with n_jobs specifying how many workers to run simultaneously
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_one_subject_safe)(
                    sub=sub,
                    dataset=dataset,
                    dataset_path=dataset_path,
                    all_qc_params=all_qc_params,
                    internal_qc_params=internal_qc_params,
                    derivatives_root=megqc_root,
                    output_root=output_root,
                    analysis_segments=analysis_segments,
                )
                for sub in dataset_sub_list
            )

        # for sub in sub_list:
        #     process_one_subject(
        #         sub=sub,
        #         dataset=dataset,
        #         dataset_path=dataset_path,
        #         all_qc_params=all_qc_params,
        #         internal_qc_params=internal_qc_params
        #     )
        # Optionally, you can handle the returned values here, e.g.,:
        # global_all_taken_raw_files = []
        # global_avg_ecg = []
        # global_avg_eog = []
        # for res in results:
        #     if res is not None:
        #         taken_files, ecg_data, eog_data, raw_obj = res
        #         global_all_taken_raw_files += taken_files
        #         global_avg_ecg += ecg_data
        #         global_avg_eog += eog_data

            # Collect results and log subjects that failed
            # results is a list of (sub, files, error_info) 3-tuples.
            # files is None for total failures; error_info carries either a
            # whole-subject error dict or {"metric_errors": [...]} for partial
            # failures where the subject completed but some metrics did not.
            excluded_subjects = [sub for sub, files, _err in results if files is None]
            failed_subjects_errors = {
                sub: err
                for sub, files, err in results
                if files is None and err is not None
            }
            # Partial failures: subject succeeded overall but ≥1 metric failed
            partial_subjects_errors = {
                sub: err
                for sub, files, err in results
                if files is not None and err is not None and "metric_errors" in err
            }

            # Save config file used for this run as a derivative:
            all_subs_raw_files = []
            for sub, files, _err in results:
                if files is not None:
                    all_subs_raw_files.extend(files)

            derivative = dataset.create_derivative(name="Meg_QC")
            _ensure_derivative_dataset_description_filename(derivative)
            derivative.dataset_description.GeneratedBy.Name = "MEG QC Pipeline"
            root_folder = derivative
            for seg in analysis_segments:
                root_folder = root_folder.create_folder(name=str(seg))

            # Always persist a timestamped config snapshot to make sensitivity
            # analyses and profile provenance explicit and auditable.
            create_config_artifact(root_folder, config_file_path, 'UsedSettings', all_subs_raw_files)

            # Write the pipeline-level derivative to disk
            with temporary_dataset_base(dataset, output_root):
                ancpbids.write_derivative(dataset, derivative)

            _write_profile_manifest(
                megqc_root=megqc_root,
                analysis_mode=analysis_mode,
                analysis_id=resolved_analysis_id,
                config_file_path=config_file_path,
                existing_config_policy=existing_config_policy,
                processed_subjects_policy=processed_subjects_policy,
            )

            # Save list of excluded subjects
            if excluded_subjects:
                excl_path = os.path.join(megqc_root, 'excluded_subjects')
                os.makedirs(os.path.dirname(excl_path), exist_ok=True)
                # Plain text list – kept for backward compatibility
                with open(excl_path, 'w', encoding='utf-8') as f:
                    for sub in excluded_subjects:
                        f.write(str(sub) + '\n')

            # Write structured JSON error log for ALL failure types:
            #   • Total failures  → subject completely excluded
            #   • Partial failures → subject processed but ≥1 metric skipped
            has_any_errors = excluded_subjects or partial_subjects_errors
            if has_any_errors:
                excl_json_path = os.path.join(megqc_root, 'excluded_subjects_errors.json')
                error_log = []

                # --- Total failures ---
                for sub in excluded_subjects:
                    entry = {"subject": sub, "failure_type": "total"}
                    err = failed_subjects_errors.get(sub)
                    if err:
                        entry.update(err)
                    else:
                        entry["error_type"] = "NoDataError"
                        entry["error_message"] = (
                            "Subject returned no results. Possible causes: "
                            "no raw files found, or all files were skipped."
                        )
                        entry["traceback"] = None
                        entry["timestamp"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    error_log.append(entry)

                # --- Partial failures (metric-level) ---
                for sub, err in partial_subjects_errors.items():
                    entry = {
                        "subject": sub,
                        "failure_type": "partial",
                        "metric_errors": err.get("metric_errors", []),
                    }
                    error_log.append(entry)

                with open(excl_json_path, 'w', encoding='utf-8') as f:
                    json.dump(error_log, f, indent=4)

                summary_parts = []
                if excluded_subjects:
                    summary_parts.append(f"{len(excluded_subjects)} total failure(s): {excluded_subjects}")
                if partial_subjects_errors:
                    summary_parts.append(
                        f"{len(partial_subjects_errors)} partial failure(s): {list(partial_subjects_errors.keys())}"
                    )
                print(
                    f"___MEGqc___: Subject errors detected — "
                    + " | ".join(summary_parts)
                    + f"\n  Detailed errors → {excl_json_path}"
                )

            # Generate Global Quality Index reports and group table
            try:
                generate_gqi_summary(dataset_path, megqc_root, config_file_path)
            except Exception as e:
                print("___MEGqc___: Failed to create global quality reports", e)

            dataset_succeeded = True
        finally:
            if dataset_succeeded or not keep_temp_on_error:
                delete_temp_folder(megqc_root)
            else:
                print(
                    "___MEGqc___: keep_temp_on_error=True, preserving temporary "
                    f"files under {os.path.join(megqc_root, '.tmp')}"
                )

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print(f"CALCULATION MODULE FINISHED. Elapsed time: {elapsed_seconds:.2f} seconds.")

    return
