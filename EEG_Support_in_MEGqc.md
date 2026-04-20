# EEG Support in the MEGqc Workflow

MEGqc was originally designed for MEG data quality control, but it now supports **standalone EEG datasets** as well as **EEG channels embedded inside MEG recordings**. This document explains how EEG data flows through every stage of the pipeline.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Supported EEG File Formats](#2-supported-eeg-file-formats)
3. [Configuration](#3-configuration)
4. [BIDS Dataset Discovery](#4-bids-dataset-discovery)
5. [File Loading & System Detection](#5-file-loading--system-detection)
6. [Channel Separation](#6-channel-separation)
7. [BIDS Channel-Type Correction](#7-bids-channel-type-correction)
8. [EEG Montage Application](#8-eeg-montage-application)
9. [EEG Re-Referencing](#9-eeg-re-referencing)
10. [Lobe / Region Assignment](#10-lobe--region-assignment)
11. [QC Metrics & EEG](#11-qc-metrics--eeg)
12. [Derivatives & BIDS Output](#12-derivatives--bids-output)
13. [Reports](#13-reports)
14. [Embedded EEG in MEG Recordings](#14-embedded-eeg-in-meg-recordings)
15. [Summary Diagram](#15-summary-diagram)

---

## 1. Overview

The pipeline classifies every recording into one of two **modalities**:

| Modality | `meg_system` value | Channel types processed |
|----------|--------------------|------------------------|
| MEG      | `'Triux'`, `'CTF'`, `'OTHER'` | `mag`, `grad` (and optionally `eeg` if embedded) |
| EEG      | `'EEG'`            | `eeg` only |

The modality is determined at load time and controls which metrics run, how channels are labelled, and where derivatives are saved.

**Key source files:**

| File | Role |
|------|------|
| `meg_qc/calculation/initial_meg_qc.py` | Loading, channel separation, montage, reference, lobe assignment |
| `meg_qc/calculation/meg_qc_pipeline.py` | Orchestration: file discovery, metric dispatch, derivative saving |
| `meg_qc/calculation/metrics/*.py` | Individual QC metrics (STD, PSD, PTP, ECG, EOG, Muscle, Head) |
| `meg_qc/plotting/meg_qc_plots.py` | Per-subject HTML report generation |
| `meg_qc/plotting/meg_qc_group_plots.py` | Group-level QA report |
| `meg_qc/plotting/meg_qc_group_qc_plots.py` | Group-level QC report |
| `meg_qc/settings/settings.ini` | User configuration (`[GENERAL]`, `[EEG]`, `[Muscle]`) |

---

## 2. Supported EEG File Formats

The `load_data()` function in `initial_meg_qc.py` handles these EEG formats via MNE readers:

| Extension | Format | MNE Reader |
|-----------|--------|------------|
| `.edf`    | European Data Format | `mne.io.read_raw_edf()` |
| `.bdf`    | BioSemi Data Format | `mne.io.read_raw_bdf()` |
| `.vhdr`   | BrainVision | `mne.io.read_raw_brainvision()` |
| `.set`    | EEGLAB | `mne.io.read_raw_eeglab()` |
| `.mff`    | EGI / MFF (directory) | `mne.io.read_raw_egi()` |
| `.cnt`    | Neuroscan CNT | `mne.io.read_raw_cnt()` |
| `.fif`    | FIF (with `_eeg` BIDS suffix) | `mne.io.read_raw_fif()` |
| `.ds`     | CTF (with `_eeg` BIDS suffix) | `mne.io.read_raw_ctf()` |

All native EEG formats (EDF, BDF, BrainVision, EEGLAB, EGI, Neuroscan) automatically set `meg_system = 'EEG'` and `modality = 'eeg'`.

For FIF and CTF files, the BIDS suffix in the filename is the primary classifier:
- `*_eeg.fif` → classified as EEG
- `*_meg.fif` → classified as MEG (even if it contains EEG channels)

A **safeguard** also checks: if a `*_meg.fif` or `*_meg.ds` file contains zero mag/grad channels, it is reclassified as EEG.

---

## 3. Configuration

### `settings.ini` — `[GENERAL]` section

```ini
[GENERAL]
# Which channel types to process:
# For MEG data: mag, grad (or both)
# For EEG data: eeg
# For simultaneous MEG+EEG: mag, grad, eeg
ch_types = mag, grad, eeg
```

Setting `ch_types = eeg` tells the pipeline to look for and process EEG recordings. When set to `mag, grad, eeg` (the default), the pipeline processes whichever channel types are present in each file.

### `settings.ini` — `[EEG]` section

```ini
[EEG]
# EEG re-referencing method:
#   'average' = common average reference (recommended default)
#   'REST'    = Reference Electrode Standardisation Technique
#   'none'    = skip re-referencing
reference_method = average

# EEG electrode montage for topographic plots:
#   'auto'         = attempt to detect from channel names
#   Standard names = standard_1020, standard_1010, standard_1005, biosemi64, etc.
montage = auto
```

### `settings.ini` — `[Muscle]` section

```ini
[Muscle]
# MEG muscle detection band (default MNE: 110-140 Hz)
muscle_freqs = 110, 140
# EEG muscle detection band (lower frequencies: 20-100 Hz)
muscle_freqs_eeg = 20, 100
```

EEG muscle artifacts appear at lower frequencies than MEG, so a separate band is used.

---

## 4. BIDS Dataset Discovery

**Function:** `get_files_list()` in `meg_qc_pipeline.py`

The pipeline uses **ancpbids** to discover files by BIDS suffix:

```
dataset.query(suffix='meg', ...)   →  MEG recordings
dataset.query(suffix='eeg', ...)   →  EEG recordings
```

Selection logic:
1. If user requested `ch_types = eeg` only → process only EEG files.
2. If user requested `ch_types = mag, grad` only → process only MEG files.
3. If user requested `ch_types = mag, grad, eeg` → process both.
4. If a multimodal dataset has both MEG and EEG but `ch_types` is unspecified → defaults to MEG only (legacy behaviour), with a console message suggesting to add `eeg`.

A **glob fallback** exists for EEG files when ancpbids returns nothing (searches for `*_eeg.{edf,bdf,vhdr,set,cnt}` in the expected BIDS paths).

---

## 5. File Loading & System Detection

**Function:** `load_data()` in `initial_meg_qc.py`

```
load_data(file_path)
    → returns (raw, shielding_str, meg_system, modality)
```

The function sets `meg_system` and `modality` based on:

| Condition | `meg_system` | `modality` |
|-----------|-------------|------------|
| `.fif` with BIDS suffix `_eeg` | `'EEG'` | `'eeg'` |
| `.fif` with suffix `_meg` + has mag/grad channels | `'Triux'` | `'meg'` |
| `.fif` with suffix `_meg` + **no** mag/grad channels | `'EEG'` | `'eeg'` |
| `.ds` with BIDS suffix `_eeg` | `'EEG'` | `'eeg'` |
| `.ds` with suffix `_meg` + has mag/grad channels | `'CTF'` | `'meg'` |
| `.edf`, `.bdf`, `.vhdr`, `.set`, `.mff`, `.cnt` | `'EEG'` | `'eeg'` |

The Triux system is further refined (by checking active shielding info) when applicable. The `shielding_str` is empty for EEG data.

---

## 6. Channel Separation

**Function:** `choose_channels()` in `initial_meg_qc.py`

```python
channels = {'mag': [], 'grad': [], 'eeg': []}
for ch_idx, ch_name in enumerate(raw.info['ch_names']):
    ch_type = mne.channel_type(raw.info, ch_idx)
    if ch_type in channels:
        channels[ch_type].append(ch_name)
```

This separates all channels into three buckets. For pure EEG recordings, `mag` and `grad` will be empty lists. For MEG recordings with embedded EEG, all three may be populated.

The constant `SUPPORTED_CH_TYPES = {'mag', 'grad', 'eeg'}` at module level defines the canonical set.

---

## 7. BIDS Channel-Type Correction

**Function:** `_apply_bids_channel_types()` in `initial_meg_qc.py`

Many EEG recordings contain non-EEG channels (EOG, EMG, ECG, MISC) that MNE may auto-detect as EEG. The BIDS `*_channels.tsv` sidecar explicitly labels each channel.

This function:
1. Locates the companion `*_channels.tsv` file by parsing the BIDS path.
2. Reads the `name` and `type` columns.
3. Re-types channels in the raw object using `raw.set_channel_types()`.

This correction runs **before** `choose_channels()` to ensure non-EEG channels are not mis-classified as EEG.

---

## 8. EEG Montage Application

**Function:** `apply_eeg_montage()` in `initial_meg_qc.py`

Only runs when `meg_system == 'EEG'` and EEG settings are provided. Assigns 3D electrode positions needed for topographic plots.

### Auto-detection mode (`montage = auto`)

Two-pass approach:

1. **Pass 1 — Direct match:** Try `standard_1005`, `standard_1010`, `standard_1020` montages. If ≥ 50% of EEG channel names match a montage, apply it.

2. **Pass 2 — Strip reference suffixes:** Many referential recordings name channels like `Fp1-M2`, `F3-A1`, `C3-Ref`. The helper `_strip_eeg_reference_suffix()` removes these suffixes using regex:
   ```python
   re.sub(r'[-_](?:M[12]|A[12]|Ref|LE|RE|AVG|REF|...)$', '', name)
   ```
   After stripping, it retries montage matching. If successful, it renames channels in the raw object and applies the montage.

### Explicit montage mode

When a specific montage name is given (e.g., `standard_1020`, `biosemi64`), it applies it directly via `mne.channels.make_standard_montage()`. If it fails, it also tries stripping reference suffixes before giving up.

---

## 9. EEG Re-Referencing

**Function:** `apply_eeg_reference()` in `initial_meg_qc.py`

Only runs when `meg_system == 'EEG'`. Three options:

| `reference_method` | Action |
|---------------------|--------|
| `'average'` | Common average reference via `raw.set_eeg_reference('average', projection=True)` |
| `'REST'` | Reference Electrode Standardisation Technique via `raw.set_eeg_reference('REST')` (requires montage with 3D positions) |
| `'none'` | Keep original reference unchanged |

---

## 10. Lobe / Region Assignment

**Function:** `add_EEG_lobes()` in `initial_meg_qc.py`

For `meg_system == 'EEG'`, channels are assigned brain-region labels based on standard 10-20 / 10-10 / 10-05 electrode naming conventions.

### Classification rules

- **Prefix → Region:** First 1–3 letters determine the region (Fp → Frontal, F → Frontal, C → Central, P → Parietal, O → Occipital, T → Temporal, etc.)
- **Laterality:** Trailing odd digit → Left hemisphere, even digit → Right hemisphere, `z`/`Z` → Midline.
- **Special electrodes:** A1/A2, M1/M2 → Reference; T3/T4/T5/T6 → Temporal (old 10-20 naming).

### Resulting lobe labels

Each channel gets a `lobe` label like `'Left Frontal'`, `'Right Parietal'`, `'Central'`, `'Occipital'`, etc., and a corresponding `lobe_color` for plotting.

Color-coded labels:

| Region | Left colour | Right colour |
|--------|------------|--------------|
| Frontal | `#1f77b4` | `#ff7f0e` |
| Temporal | `#2ca02c` | `#9467bd` |
| Parietal | `#e377c2` | `#d62728` |
| Occipital | `#bcbd22` | `#17becf` |
| Central | `#8c564b` | — |
| Reference | `#7f7f7f` | — |

This is analogous to how `add_Triux_lobes()` and `add_CTF_lobes()` work for MEG systems.

---

## 11. QC Metrics & EEG

All QC metrics iterate over the user-selected channel types (`m_or_g_chosen`). For pure EEG recordings, this is `['eeg']`. Each metric handles EEG transparently because the data is always an MNE `Raw`/`Epochs` object.

### Metric-by-metric behaviour

| Metric | EEG Support | Notes |
|--------|-------------|-------|
| **STD** (Standard Deviation) | ✅ Full | Computes per-epoch STD for each EEG channel. |
| **PSD** (Power Spectral Density) | ✅ Full | Brain-wave bands (delta, theta, alpha, beta, gamma) work identically. |
| **PTP Manual** (Peak-to-Peak) | ✅ Full | Epoch-wise peak-to-peak amplitude for each EEG channel. |
| **ECG** (Cardiac Artifacts) | ✅ Full | Detects ECG events from dedicated ECG channels or reconstructs from EEG. Computes artifact influence on each EEG channel. |
| **EOG** (Ocular Artifacts) | ✅ Full | Detects blink/saccade events from EOG channels. Computes artifact influence on each EEG channel. |
| **Muscle** (High-Frequency Artifacts) | ✅ Full | Uses **20–100 Hz** band for EEG (vs 110–140 Hz for MEG) — configured via `muscle_freqs_eeg` in settings. |
| **Head Motion** | ❌ Skipped | Requires MEG cHPI coils. When `meg_system == 'EEG'`, the pipeline prints: _"Head motion metric is not available for EEG data (requires MEG cHPI coils)."_ |

### ECG channel detection for EEG data

The `get_ECG_data_choose_method()` function in `ECG_EOG_meg_qc.py` searches for ECG channels by name pattern. In EEG recordings, a channel named `ECG`, `EKG`, or similar may be present. If the channel is currently typed as `eeg`, the function re-types it to `ecg` using `raw.set_channel_types()`. If the channel is part of an SSP projector (e.g., "Average EEG reference"), those projectors are removed first to avoid errors.

---

## 12. Derivatives & BIDS Output

Derivatives are saved with a BIDS-compliant suffix based on modality:

```python
_bids_suffix = 'eeg' if meg_system == 'EEG' else 'meg'
```

This means:
- MEG derivatives: `sub-009_ses-1_task-X_desc-STDmag_meg.tsv`
- EEG derivatives: `sub-009_ses-1_task-X_desc-STDeeg_eeg.tsv`

Derivatives are stored under:
```
<dataset>/derivatives/meg_qc/calculation/eeg/sub-XXX/   ← EEG
<dataset>/derivatives/meg_qc/calculation/meg/sub-XXX/   ← MEG
```

Intermediate filtered/resampled files (stored in `.tmp/`) are always saved as FIF regardless of the original EEG format, since MNE cannot write back to native EEG formats (EDF, BDF, etc.).

---

## 13. Reports

### Per-Subject Reports

Generated by `meg_qc_plots.py`. The report organises plots by channel type in tabs:

- **MAG** tab — magnetometer plots
- **GRAD** tab — gradiometer plots
- **EEG** tab — EEG channel plots

For pure EEG recordings, only the EEG tab appears. For MEG recordings with embedded EEG, all applicable tabs are shown.

### Group Reports (QA and QC)

Generated by `meg_qc_group_plots.py` and `meg_qc_group_qc_plots.py`. Tab order:

```python
tab_order = ["Combined (mag+grad)", "MAG", "GRAD", "EEG"]
```

Only tabs with actual data are shown. The group reports can split MEG and EEG data into separate report files:

```python
# MEG-only report
meg_html = _build_report_html(...)

# EEG-only report (only if dataset has genuine EEG recordings)
if has_eeg:
    eeg_html = _build_report_html(...)
```

Modality detection in group reports uses path-based inference:
- Files under `calculation/eeg/` → `modality = 'eeg'`
- Files with `_eeg` in the name → `modality = 'eeg'`

---

## 14. Embedded EEG in MEG Recordings

When an MEG recording (Triux or CTF) also contains EEG electrodes, these are **not** treated as a separate EEG recording. Instead:

1. `load_data()` classifies it as MEG (`meg_system = 'Triux'` or `'CTF'`).
2. `choose_channels()` separates out the `eeg` channels alongside `mag` and `grad`.
3. If `ch_types` includes `eeg`, EEG channels are processed by all metrics in their own `eeg` pass.
4. The per-subject report includes an **EEG tab** inside the MEG report, with a blue info banner:

   > **EEG channels detected in MEG recording**
   >
   > This MEG recording also contains EEG electrode channels. [...]

5. The `embedded_eeg_note` flag on the report builder controls whether this banner appears.

**Important:** For embedded EEG, the EEG montage and reference functions are **not** applied (they only run when `meg_system == 'EEG'`). The embedded EEG channels use whatever positions MNE reads from the FIF/CTF file.

---

## 15. Summary Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     settings.ini                             │
│  ch_types = mag, grad, eeg                                   │
│  [EEG] montage = auto, reference_method = average            │
│  [Muscle] muscle_freqs_eeg = 20, 100                         │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│             get_files_list()  (meg_qc_pipeline.py)           │
│  ancpbids query: suffix='meg' + suffix='eeg'                 │
│  Glob fallback for EEG: *.edf, *.bdf, *.vhdr, *.set, *.cnt │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│             load_data()  (initial_meg_qc.py)                 │
│  Detects format → sets meg_system & modality                 │
│  Safeguard: _meg.fif with 0 MEG channels → reclassify EEG   │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│         initial_processing()  (initial_meg_qc.py)            │
│                                                              │
│  1. _apply_bids_channel_types()  ← fix mis-typed channels   │
│  2. choose_channels()            ← separate mag/grad/eeg     │
│  3. IF meg_system == 'EEG':                                  │
│       a. apply_eeg_montage()     ← 3D positions for topomaps│
│       b. apply_eeg_reference()   ← avg / REST / none         │
│  4. assign_channels_properties() ← lobe labels & colours     │
│       └─ add_EEG_lobes()        ← 10-20 naming rules        │
│  5. Filter, crop, resample, epoch                            │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│        process_one_subject()  (meg_qc_pipeline.py)           │
│                                                              │
│  For each metric in [STD, PSD, PTP, ECG, EOG, Head, Muscle]:│
│    • Head: SKIPPED if meg_system == 'EEG'                    │
│    • Muscle: uses muscle_freqs_eeg (20-100 Hz) for EEG      │
│    • All others: run on 'eeg' channels normally              │
│                                                              │
│  Save derivatives with _eeg suffix & under eeg/ folder       │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│              Report Generation                               │
│                                                              │
│  Per-subject:  EEG tab in subject HTML report                │
│  Group QA:     EEG tab in group QA report                    │
│  Group QC:     Separate EEG report file if EEG data exists   │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: How to Run MEGqc on EEG Data

1. Organise your data in BIDS format:
   ```
   my_dataset/
     sub-01/
       eeg/
         sub-01_task-rest_eeg.set
         sub-01_task-rest_channels.tsv   ← recommended
   ```

2. Set `ch_types = eeg` in `settings.ini`.

3. Optionally configure `[EEG]` section (montage, reference).

4. Run:
   ```bash
   megqc                  # GUI
   # or
   run-megqc              # CLI
   ```

The pipeline will auto-detect EEG files, apply montage + reference, run all applicable metrics (skipping Head motion), and generate an HTML report with an EEG tab.

