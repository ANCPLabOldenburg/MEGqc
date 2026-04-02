"""
qc_viewer_annotation_manager.py – Parse MEGqc derivative files and provide annotation data.

Loads TSV / JSON derivative files produced by MEGqc and exposes a uniform
API for the time-series viewer to overlay coloured highlights.

Integrated into meg_qc package as part of the QC Viewer module.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Data structures                                                    #
# ------------------------------------------------------------------ #
@dataclass
class TimeInterval:
    """A time interval (in seconds) with optional score."""
    start: float
    end: float
    score: float = 0.0


@dataclass
class EventMarker:
    """A stimulus event at a specific time with an integer event ID."""
    time: float
    event_id: int
    channel: str = ""
    label: str = ""     # Optional trial_type label from events.tsv
    source: str = ""    # "stim", "bids_events", "raw"


@dataclass
class EpochAnnotation:
    """Boolean channel × epoch matrix for noisy/flat classification."""
    channel_names: List[str]
    epoch_indices: List[int]
    matrix: np.ndarray  # bool, shape (n_channels, n_epochs)
    metric: str = ""
    ch_type: str = ""


@dataclass
class ChannelFlag:
    """A channel flagged by a metric, with its value."""
    name: str
    value: float = 0.0


@dataclass
class AnnotationSet:
    """All annotations available for one MEG recording."""
    # Channel-level flags  (metric → list of channels)
    std_noisy_channels: Dict[str, List[ChannelFlag]] = field(default_factory=dict)
    std_flat_channels: Dict[str, List[ChannelFlag]] = field(default_factory=dict)
    ptp_noisy_channels: Dict[str, List[ChannelFlag]] = field(default_factory=dict)
    ptp_flat_channels: Dict[str, List[ChannelFlag]] = field(default_factory=dict)
    psd_noisy_channels: Dict[str, List[ChannelFlag]] = field(default_factory=dict)
    ecg_affected_channels: List[str] = field(default_factory=list)
    eog_affected_channels: List[str] = field(default_factory=list)

    # Ranked channels (sorted by |correlation| descending) for tiered display
    ecg_ranked_channels: List[ChannelFlag] = field(default_factory=list)
    eog_ranked_channels: List[ChannelFlag] = field(default_factory=list)

    # Epoch-level boolean matrices
    noisy_epochs_std: Dict[str, EpochAnnotation] = field(default_factory=dict)
    flat_epochs_std: Dict[str, EpochAnnotation] = field(default_factory=dict)
    noisy_epochs_ptp: Dict[str, EpochAnnotation] = field(default_factory=dict)
    flat_epochs_ptp: Dict[str, EpochAnnotation] = field(default_factory=dict)

    # Muscle artefact intervals
    muscle_intervals: List[TimeInterval] = field(default_factory=list)
    muscle_scores_times: Optional[np.ndarray] = None
    muscle_scores: Optional[np.ndarray] = None

    # Stimulus / event markers
    events: List[EventMarker] = field(default_factory=list)
    stim_channels: List[str] = field(default_factory=list)

    # BIDS events.tsv info
    bids_event_onsets: List[float] = field(default_factory=list)
    bids_event_ids: List[int] = field(default_factory=list)
    bids_id_to_trial_type: Dict[str, str] = field(default_factory=dict)

    # ECG/EOG detected event times (peak locations from ECG/EOG analysis)
    ecg_event_times: List[float] = field(default_factory=list)
    ecg_event_rate: float = 0.0
    eog_event_times: List[float] = field(default_factory=list)
    eog_event_rate: float = 0.0

    # Recording metadata
    sfreq: float = 0.0
    duration: float = 0.0
    epoch_duration: float = 0.0
    n_epochs: int = 0


# ------------------------------------------------------------------ #
# Loader                                                             #
# ------------------------------------------------------------------ #
class AnnotationManager:
    """Load and manage MEGqc annotations for a given recording."""

    def __init__(self):
        self.annotations: Optional[AnnotationSet] = None
        self._calc_dir: Optional[Path] = None
        self._prefix: str = ""

    # ---- public API ------------------------------------------------ #

    def load_for_recording(self, calc_dir: str, file_prefix: str) -> AnnotationSet:
        """Load all available annotations for a recording.

        Parameters
        ----------
        calc_dir : str
            Path to the ``calculation/sub-XXX/`` directory.
        file_prefix : str
            BIDS entity prefix, e.g.
            ``sub-009_ses-1_task-induction_run-1``
        """
        self._calc_dir = Path(calc_dir)
        self._prefix = file_prefix
        self.annotations = AnnotationSet()

        self._load_simple_metrics()
        self._load_epoch_matrices()
        self._load_muscle()
        self._load_ecg_eog_channels()
        self._load_ecg_eog_events()
        self._load_stimulus()
        self._load_event_summary()

        return self.annotations

    def auto_detect_recordings(self, calc_dir: str) -> List[str]:
        """Scan a calculation dir and return distinct recording prefixes."""
        d = Path(calc_dir)
        if not d.is_dir():
            return []
        prefixes = set()
        for f in d.iterdir():
            m = re.match(r"^(sub-[^_]+(?:_ses-[^_]+)?(?:_task-[^_]+)?(?:_run-[^_]+)?)_desc-", f.name)
            if m:
                prefixes.add(m.group(1))
        return sorted(prefixes)

    def get_flagged_channels(self, metric: str, max_channels: int = 0) -> Set[str]:
        """Return the set of channel names flagged by *metric*."""
        if self.annotations is None:
            return set()
        a = self.annotations
        mapping = {
            "std_noisy": a.std_noisy_channels,
            "std_flat": a.std_flat_channels,
            "ptp_noisy": a.ptp_noisy_channels,
            "ptp_flat": a.ptp_flat_channels,
            "psd_noisy": a.psd_noisy_channels,
        }
        if metric in mapping:
            channels = set()
            for ch_type_flags in mapping[metric].values():
                channels.update(cf.name for cf in ch_type_flags)
            return channels
        if metric == "ecg":
            ranked = a.ecg_ranked_channels if a.ecg_ranked_channels else []
            if max_channels > 0 and ranked:
                return {cf.name for cf in ranked[:max_channels]}
            return set(a.ecg_affected_channels)
        if metric == "eog":
            ranked = a.eog_ranked_channels if a.eog_ranked_channels else []
            if max_channels > 0 and ranked:
                return {cf.name for cf in ranked[:max_channels]}
            return set(a.eog_affected_channels)
        return set()

    def get_epoch_matrix(self, metric: str, ch_type: str) -> Optional[EpochAnnotation]:
        """Return the epoch annotation matrix for *metric* and *ch_type*."""
        if self.annotations is None:
            return None
        stores = {
            "noisy_std": self.annotations.noisy_epochs_std,
            "flat_std": self.annotations.flat_epochs_std,
            "noisy_ptp": self.annotations.noisy_epochs_ptp,
            "flat_ptp": self.annotations.flat_epochs_ptp,
        }
        store = stores.get(metric, {})
        return store.get(ch_type)

    def get_muscle_intervals(self) -> List[TimeInterval]:
        if self.annotations:
            return self.annotations.muscle_intervals
        return []

    def get_available_metrics(self) -> List[str]:
        """Return list of annotation categories that have data."""
        if self.annotations is None:
            return []
        avail = []
        a = self.annotations
        if any(a.std_noisy_channels.values()):
            avail.append("std_noisy")
        if any(a.std_flat_channels.values()):
            avail.append("std_flat")
        if any(a.ptp_noisy_channels.values()):
            avail.append("ptp_noisy")
        if any(a.ptp_flat_channels.values()):
            avail.append("ptp_flat")
        if any(a.psd_noisy_channels.values()):
            avail.append("psd_noisy")
        if a.ecg_affected_channels:
            avail.append("ecg")
        if a.eog_affected_channels:
            avail.append("eog")
        if a.noisy_epochs_std:
            avail.append("noisy_std_epochs")
        if a.flat_epochs_std:
            avail.append("flat_std_epochs")
        if a.noisy_epochs_ptp:
            avail.append("noisy_ptp_epochs")
        if a.flat_epochs_ptp:
            avail.append("flat_ptp_epochs")
        if a.muscle_intervals:
            avail.append("muscle")
        if a.ecg_event_times:
            avail.append("ecg_events")
        if a.eog_event_times:
            avail.append("eog_events")
        if a.events:
            avail.append("events")
        if a.bids_event_onsets:
            avail.append("bids_events")
        return avail

    def get_events(self, stim_channel: str = "", include_bids: bool = True) -> List[EventMarker]:
        """Return event markers, optionally filtered by channel."""
        if self.annotations is None:
            return []
        result = []
        if stim_channel:
            result = [e for e in self.annotations.events
                      if e.channel == stim_channel or e.source == stim_channel]
        else:
            result = list(self.annotations.events)
        # Include BIDS events.tsv events
        if include_bids and self.annotations.bids_event_onsets:
            id_map = self.annotations.bids_id_to_trial_type
            for onset, eid in zip(self.annotations.bids_event_onsets,
                                  self.annotations.bids_event_ids):
                label = id_map.get(str(eid), "")
                result.append(EventMarker(
                    time=onset, event_id=eid, channel="events.tsv",
                    label=label, source="bids_events",
                ))
        result.sort(key=lambda e: e.time)
        return result

    # ---- private loaders ------------------------------------------- #

    def _fpath(self, desc: str, ext: str = ".tsv") -> Optional[Path]:
        """Build the full path for a derivative file, return None if missing.

        Tries ``_meg`` suffix first, then ``_eeg`` to support both modalities.
        """
        for suffix in ("meg", "eeg"):
            fname = f"{self._prefix}_desc-{desc}_{suffix}{ext}"
            p = self._calc_dir / fname
            if p.exists():
                return p
        return None

    def _load_simple_metrics(self):
        """Parse SimpleMetrics JSON for channel-level flags."""
        p = self._fpath("SimpleMetrics", ".json")
        if not p:
            return
        with open(p) as f:
            data = json.load(f)

        self._extract_channel_flags(data, "STD", "STD_all_time_series",
                                    self.annotations.std_noisy_channels,
                                    self.annotations.std_flat_channels)
        for sub_key in ("ptp_manual_all", "PtP_manual_all_time_series",
                        "ptp_manual_all_time_series"):
            self._extract_channel_flags(data, "PTP_MANUAL", sub_key,
                                        self.annotations.ptp_noisy_channels,
                                        self.annotations.ptp_flat_channels)
        for sub_key in ("ptp_auto_all", "PtP_auto_all_time_series",
                        "ptp_auto_all_time_series"):
            self._extract_channel_flags(data, "PTP_AUTO", sub_key,
                                        self.annotations.ptp_noisy_channels,
                                        self.annotations.ptp_flat_channels)
        self._extract_psd_flags(data)
        self._extract_ecg_eog_from_json(data)

    def _extract_channel_flags(self, data: dict, top_key: str, sub_key: str,
                               noisy_store: Dict, flat_store: Dict):
        top = data.get(top_key, {})
        if not isinstance(top, dict):
            return
        section = top.get(sub_key, {})
        if not isinstance(section, dict):
            return
        for ch_type in ("mag", "grad", "eeg"):
            info = section.get(ch_type)
            if not isinstance(info, dict):
                continue
            details = info.get("details")
            if not isinstance(details, dict):
                continue
            noisy = details.get("noisy_ch", {})
            flat = details.get("flat_ch", {})
            if noisy and isinstance(noisy, dict):
                noisy_store[ch_type] = [ChannelFlag(name=k, value=v if isinstance(v, (int, float)) else 0.0) for k, v in noisy.items()]
            if flat and isinstance(flat, dict):
                flat_store[ch_type] = [ChannelFlag(name=k, value=v if isinstance(v, (int, float)) else 0.0) for k, v in flat.items()]

    def _extract_psd_flags(self, data: dict):
        psd_section = data.get("PSD", {})
        if isinstance(psd_section, dict):
            for ch_type in ("mag", "grad", "eeg"):
                for sub_key in ("PSD_local", "PSD_global"):
                    sub = psd_section.get(sub_key)
                    if not isinstance(sub, dict):
                        continue
                    ch_info = sub.get(ch_type)
                    if not isinstance(ch_info, dict):
                        continue
                    details = ch_info.get("details")
                    if not isinstance(details, dict):
                        continue
                    noisy_names = [
                        ch_name for ch_name, ch_detail in details.items()
                        if isinstance(ch_detail, dict) and len(ch_detail) > 0
                    ]
                    if noisy_names:
                        existing = self.annotations.psd_noisy_channels.get(ch_type, [])
                        existing_names = {cf.name for cf in existing}
                        for n in noisy_names:
                            if n not in existing_names:
                                existing.append(ChannelFlag(name=n))
                        self.annotations.psd_noisy_channels[ch_type] = existing

        for ch_type, desc in [("grad", "PSDnoiseGrad"), ("mag", "PSDnoiseMag"), ("eeg", "PSDnoiseEeg")]:
            p = self._fpath(desc)
            if p:
                try:
                    df = pd.read_csv(p, sep="\t")
                    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
                    if "Name" in df.columns:
                        names = df["Name"].dropna().tolist()
                        existing = self.annotations.psd_noisy_channels.get(ch_type, [])
                        existing_names = {cf.name for cf in existing}
                        for n in names:
                            if n not in existing_names:
                                existing.append(ChannelFlag(name=n))
                        self.annotations.psd_noisy_channels[ch_type] = existing
                except Exception:
                    pass

    def _extract_ecg_eog_from_json(self, data: dict):
        ecg_sec = data.get("ECG", {})
        if isinstance(ecg_sec, dict):
            ecg_flags: list = []
            for sub_key in ecg_sec:
                sub = ecg_sec[sub_key]
                if not isinstance(sub, dict):
                    continue
                for ch_type in ("mag", "grad", "eeg"):
                    ch_info = sub.get(ch_type)
                    if not isinstance(ch_info, dict):
                        continue
                    details = ch_info.get("details")
                    if not isinstance(details, dict):
                        continue
                    for ch_name, score_data in details.items():
                        if isinstance(score_data, (list, tuple)) and len(score_data) > 0:
                            score = abs(float(score_data[0]))
                        elif isinstance(score_data, (int, float)):
                            score = abs(float(score_data))
                        else:
                            score = 0.0
                        ecg_flags.append(ChannelFlag(name=ch_name, value=score))
            seen = set()
            unique_flags = []
            for cf in sorted(ecg_flags, key=lambda x: -x.value):
                if cf.name not in seen:
                    seen.add(cf.name)
                    unique_flags.append(cf)
            self.annotations.ecg_ranked_channels = unique_flags
            self.annotations.ecg_affected_channels = [cf.name for cf in unique_flags]

        eog_sec = data.get("EOG", {})
        if isinstance(eog_sec, dict):
            eog_flags: list = []
            for sub_key in eog_sec:
                sub = eog_sec[sub_key]
                if not isinstance(sub, dict):
                    continue
                for ch_type in ("mag", "grad", "eeg"):
                    ch_info = sub.get(ch_type)
                    if not isinstance(ch_info, dict):
                        continue
                    details = ch_info.get("details")
                    if not isinstance(details, dict):
                        continue
                    for ch_name, score_data in details.items():
                        if isinstance(score_data, (list, tuple)) and len(score_data) > 0:
                            score = abs(float(score_data[0]))
                        elif isinstance(score_data, (int, float)):
                            score = abs(float(score_data))
                        else:
                            score = 0.0
                        eog_flags.append(ChannelFlag(name=ch_name, value=score))
            seen = set()
            unique_flags = []
            for cf in sorted(eog_flags, key=lambda x: -x.value):
                if cf.name not in seen:
                    seen.add(cf.name)
                    unique_flags.append(cf)
            self.annotations.eog_ranked_channels = unique_flags
            self.annotations.eog_affected_channels = [cf.name for cf in unique_flags]

    def _load_ecg_eog_channels(self):
        p = self._fpath("Sensors")
        if not p:
            return
        try:
            df = pd.read_csv(p, sep="\t")
            df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
        except Exception:
            return

        self._merge_ecg_eog_from_sensors(
            df, metric="ecg",
            corr_col="ecg_corr_coeff",
            amp_col="ecg_amplitude_ratio",
            sim_col="ecg_similarity_score",
            ranked_attr="ecg_ranked_channels",
            affected_attr="ecg_affected_channels",
        )
        self._merge_ecg_eog_from_sensors(
            df, metric="eog",
            corr_col="eog_corr_coeff",
            amp_col="eog_amplitude_ratio",
            sim_col="eog_similarity_score",
            ranked_attr="eog_ranked_channels",
            affected_attr="eog_affected_channels",
        )

    def _merge_ecg_eog_from_sensors(self, df: pd.DataFrame, *,
                                     metric: str,
                                     corr_col: str, amp_col: str, sim_col: str,
                                     ranked_attr: str, affected_attr: str):
        if "Name" not in df.columns:
            return

        has_corr = corr_col in df.columns
        has_amp = amp_col in df.columns
        has_sim = sim_col in df.columns
        if not (has_corr or has_amp or has_sim):
            return

        existing_ranked: list = getattr(self.annotations, ranked_attr, [])
        existing_map = {cf.name: cf.value for cf in existing_ranked}

        for _, row in df.iterrows():
            ch_name = row.get("Name")
            if not ch_name or pd.isna(ch_name):
                continue

            parts = []
            weights = []
            if has_corr and pd.notna(row.get(corr_col)):
                try:
                    parts.append(abs(float(row[corr_col])))
                    weights.append(0.4)
                except (ValueError, TypeError):
                    pass
            if has_amp and pd.notna(row.get(amp_col)):
                try:
                    parts.append(float(row[amp_col]))
                    weights.append(0.3)
                except (ValueError, TypeError):
                    pass
            if has_sim and pd.notna(row.get(sim_col)):
                try:
                    parts.append(float(row[sim_col]))
                    weights.append(0.3)
                except (ValueError, TypeError):
                    pass

            if not parts:
                continue
            total_w = sum(weights)
            composite = sum(p * w / total_w for p, w in zip(parts, weights))

            if ch_name in existing_map:
                existing_map[ch_name] = max(existing_map[ch_name], composite)
            else:
                existing_map[ch_name] = composite

        new_ranked = [ChannelFlag(name=n, value=v) for n, v in existing_map.items()]
        new_ranked.sort(key=lambda x: -x.value)
        setattr(self.annotations, ranked_attr, new_ranked)
        setattr(self.annotations, affected_attr, [cf.name for cf in new_ranked])

    def _load_ecg_eog_events(self):
        """Load ECG/EOG detected event (peak) times from ECGchannel/EOGchannel TSVs.

        These TSVs contain columns like ``event_indexes`` and ``fs``.
        We convert sample indices to times using the sampling frequency.
        Also extracts ``events_rate_per_min`` when available.
        """
        for metric, desc, times_attr, rate_attr in [
            ("ecg", "ECGchannel", "ecg_event_times", "ecg_event_rate"),
            ("eog", "EOGchannel", "eog_event_times", "eog_event_rate"),
        ]:
            p = self._fpath(desc)
            if not p:
                continue
            try:
                df = pd.read_csv(p, sep="\t")
                df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
            except Exception:
                continue

            if "event_indexes" not in df.columns:
                continue

            # Get sampling frequency
            sfreq = 1.0
            if "fs" in df.columns:
                fs_vals = pd.to_numeric(df["fs"], errors="coerce").dropna()
                if len(fs_vals) > 0:
                    sfreq = float(fs_vals.iloc[0])

            # Convert sample indices to times
            indices = pd.to_numeric(df["event_indexes"], errors="coerce").dropna().values
            times = (indices / sfreq).tolist() if sfreq > 0 else []
            setattr(self.annotations, times_attr, times)

            # Extract rate
            if "events_rate_per_min" in df.columns:
                rate_vals = pd.to_numeric(df["events_rate_per_min"], errors="coerce").dropna()
                if len(rate_vals) > 0:
                    setattr(self.annotations, rate_attr, float(rate_vals.iloc[0]))

    def _load_epoch_matrices(self):
        mappings = [
            ("Noisy_epochs_on_std_base_mag", self.annotations.noisy_epochs_std, "noisy_std", "mag"),
            ("Noisy_epochs_on_std_base_grad", self.annotations.noisy_epochs_std, "noisy_std", "grad"),
            ("Noisy_epochs_on_std_base_eeg", self.annotations.noisy_epochs_std, "noisy_std", "eeg"),
            ("Flat_epochs_on_std_base_mag", self.annotations.flat_epochs_std, "flat_std", "mag"),
            ("Flat_epochs_on_std_base_grad", self.annotations.flat_epochs_std, "flat_std", "grad"),
            ("Flat_epochs_on_std_base_eeg", self.annotations.flat_epochs_std, "flat_std", "eeg"),
            ("Noisy_epochs_on_ptp_base_mag", self.annotations.noisy_epochs_ptp, "noisy_ptp", "mag"),
            ("Noisy_epochs_on_ptp_base_grad", self.annotations.noisy_epochs_ptp, "noisy_ptp", "grad"),
            ("Noisy_epochs_on_ptp_base_eeg", self.annotations.noisy_epochs_ptp, "noisy_ptp", "eeg"),
            ("Flat_epochs_on_ptp_base_mag", self.annotations.flat_epochs_ptp, "flat_ptp", "mag"),
            ("Flat_epochs_on_ptp_base_grad", self.annotations.flat_epochs_ptp, "flat_ptp", "grad"),
            ("Flat_epochs_on_ptp_base_eeg", self.annotations.flat_epochs_ptp, "flat_ptp", "eeg"),
        ]
        for desc, store, metric, ch_type in mappings:
            p = self._fpath(desc)
            if not p:
                continue
            try:
                # Try old format first: channel names as row index (index_col=0)
                df = pd.read_csv(p, sep="\t", index_col=0)
                ch_names = df.index.tolist()
                epoch_cols = [c for c in df.columns if str(c).isdigit()]

                # Detect new format: if index_col=0 ate a data column and
                # channel names look like booleans, re-read without index_col
                if not epoch_cols or (
                    ch_names and str(ch_names[0]).strip().lower() in ("true", "false")
                ):
                    df = pd.read_csv(p, sep="\t")
                    df = df.drop(
                        columns=[c for c in df.columns if str(c).startswith("Unnamed:")],
                        errors="ignore",
                    )
                    epoch_cols = [c for c in df.columns if str(c).isdigit()]
                    if not epoch_cols:
                        continue
                    # New format has no channel-name column; synthesise names
                    ch_names = [f"ch_{i}" for i in range(len(df))]
                    # If a 'mean' column exists, drop it from epoch columns
                    epoch_cols = [c for c in epoch_cols if c != "mean"]

                if not epoch_cols:
                    continue
                epoch_indices = [int(c) for c in epoch_cols]
                mat = df[epoch_cols].values.astype(str)
                bool_mat = np.char.lower(mat) == "true"
                store[ch_type] = EpochAnnotation(
                    channel_names=ch_names,
                    epoch_indices=epoch_indices,
                    matrix=bool_mat,
                    metric=metric,
                    ch_type=ch_type,
                )
            except Exception:
                pass

    def _load_muscle(self):
        p = self._fpath("Muscle")
        if not p:
            return
        try:
            df = pd.read_csv(p, sep="\t")
            df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
        except Exception:
            return

        if "data_times" in df.columns and "scores_muscle" in df.columns:
            times = pd.to_numeric(df["data_times"], errors="coerce").dropna().values
            scores = pd.to_numeric(df["scores_muscle"], errors="coerce").dropna().values
            n = min(len(times), len(scores))
            self.annotations.muscle_scores_times = times[:n]
            self.annotations.muscle_scores = scores[:n]

        if "high_scores_muscle_times" in df.columns and "high_scores_muscle" in df.columns:
            ht = pd.to_numeric(df["high_scores_muscle_times"], errors="coerce").dropna().values
            hs = pd.to_numeric(df["high_scores_muscle"], errors="coerce").dropna().values
            n = min(len(ht), len(hs))
            for i in range(n):
                self.annotations.muscle_intervals.append(
                    TimeInterval(start=ht[i] - 0.5, end=ht[i] + 0.5, score=hs[i])
                )

    def _load_stimulus(self):
        """Load stimulus / event markers from the stimulus TSV."""
        p = self._fpath("stimulus")
        if not p:
            return
        try:
            df = pd.read_csv(p, sep="\t")
            # Drop stale index column from old derivatives saved with index=True
            df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed:")], errors="ignore")
        except Exception:
            return

        if "time" not in df.columns:
            return

        times = pd.to_numeric(df["time"], errors="coerce").values
        stim_cols = [c for c in df.columns if c.startswith("STI")]
        if not stim_cols:
            return

        self.annotations.stim_channels = stim_cols

        for col in stim_cols:
            vals = pd.to_numeric(df[col], errors="coerce").values
            if len(vals) < 2:
                continue
            unique = np.unique(vals[~np.isnan(vals)])
            if len(unique) > 100:
                continue
            diff = np.diff(vals)
            change_idx = np.where(diff != 0)[0] + 1
            for idx in change_idx:
                if idx < len(times) and not np.isnan(vals[idx]):
                    event_val = int(vals[idx])
                    if event_val != 0:
                        self.annotations.events.append(
                            EventMarker(time=times[idx], event_id=event_val,
                                        channel=col, source="stim")
                        )
        self.annotations.events.sort(key=lambda e: e.time)

    def _load_event_summary(self):
        """Load EventSummary JSON with bids_events_info for events.tsv data.

        The EventSummary JSON has structure:
        {
          "bids_events_info": {
            "event_onsets_s": [7.282, 8.041, ...],
            "event_ids": [2, 100, ...],
            "id_to_trial_type": {"2": "standard", ...},
            ...
          },
          "stim_channel_event_counts": {"STI101": {...}, ...},
          "sfreq": 1000.0,
          ...
        }
        """
        p = self._fpath("EventSummary", ".json")
        if not p:
            return
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception:
            return

        # Extract sfreq
        sfreq = data.get("sfreq", 0.0)
        if sfreq and self.annotations.sfreq == 0:
            self.annotations.sfreq = float(sfreq)

        # Extract bids_events_info
        bids_info = data.get("bids_events_info", {})
        if not isinstance(bids_info, dict):
            return

        onsets = bids_info.get("event_onsets_s", [])
        ids = bids_info.get("event_ids", [])
        id_map = bids_info.get("id_to_trial_type", {})

        if onsets and ids:
            n = min(len(onsets), len(ids))
            self.annotations.bids_event_onsets = [float(o) for o in onsets[:n]]
            self.annotations.bids_event_ids = [int(i) for i in ids[:n]]
            self.annotations.bids_id_to_trial_type = {
                str(k): str(v) for k, v in id_map.items()
            } if id_map else {}

        # Also merge stim channel event counts for reference
        stim_counts = data.get("stim_channel_event_counts", {})
        if isinstance(stim_counts, dict):
            for ch_name in stim_counts:
                if ch_name not in self.annotations.stim_channels:
                    counts = stim_counts[ch_name]
                    if isinstance(counts, dict) and counts:
                        self.annotations.stim_channels.append(ch_name)

