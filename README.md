# MEGqc

MEGqc is an open-source, BIDS-aligned toolbox for automated **MEG quality assessment (QA)** and explicit **quality control (QC)** summarization.

It is designed for large cohorts and reproducible workflows, and provides both:
- interactive HTML reports for human inspection, and
- machine-readable derivatives for downstream automation.

## What MEGqc Provides

- **QA-first quality profiling** of raw MEG signal quality before exclusion decisions.
- **Multi-metric coverage** including:
  - standard deviation (STD),
  - peak-to-peak amplitude (PtP),
  - power spectral density (PSD),
  - ECG/EOG-related contamination,
  - high-frequency muscle burden,
  - optional head-motion summaries.
- **Multi-scale reporting** across recording, channel, epoch, subject, dataset group, and multi-sample comparisons.
- **QC support layer** with configurable module-level criteria and a Global Quality Index (GQI).
- **Reproducible execution** with profile-aware outputs and saved settings provenance.
- **Three usage modes**: CLI, GUI, and programmatic dispatchers.

## Requirements

- **Python 3.10**
- MEG data organized according to **BIDS/MEG-BIDS**.

## Installation

### Option 1: Installer-based (recommended for most users)

Download the installer bundle from the [MEGqc releases](https://github.com/ANCPLabOldenburg/MEGqc/raw/main/installers/installers.zip) and follow the platform-specific instructions in the [installation guide](https://ancplaboldenburg.github.io/megqc_documentation/installation/gui.html).

### Option 2: CLI-based (Conda + pip)

```bash
conda create -n megqc-py310 python=3.10 pip -y
conda activate megqc-py310
pip install meg-qc
```

For detailed installation instructions, see the [CLI installation guide](https://ancplaboldenburg.github.io/megqc_documentation/installation/cli.html).

## Quick Start (CLI)

1. Export default config:

```bash
get-megqc-config --target_directory ./config
```

2. Run QA/QC calculation:

```bash
run-megqc --inputdata /path/to/bids_dataset --config ./config/settings.ini
```

3. Build plotting reports:

```bash
run-megqc-plotting --inputdata /path/to/bids_dataset
```

4. Recompute GQI summaries (optional):

```bash
globalqualityindex --inputdata /path/to/bids_dataset
```

5. Run full pipeline in one command (calculation + plotting):

```bash
run-megqc --inputdata /path/to/bids_dataset --config ./config/settings.ini --run-all
```

## Launch GUI

```bash
megqc
```

The GUI uses the same backend logic as CLI dispatchers and writes the same derivative/report outputs.

## Typical Outputs

MEGqc writes outputs under BIDS derivatives (default):

- `derivatives/Meg_QC/calculation/` — metric tables + JSON summaries
- `derivatives/Meg_QC/reports/` — interactive HTML reports
- `derivatives/Meg_QC/summary_reports/` — QC summaries including GQI artifacts

## Documentation

- **Installation (Installer-based):** [https://ancplaboldenburg.github.io/megqc_documentation/installation/gui.html](https://ancplaboldenburg.github.io/megqc_documentation/installation/gui.html)
- **Installation (CLI-based):** [https://ancplaboldenburg.github.io/megqc_documentation/installation/cli.html](https://ancplaboldenburg.github.io/megqc_documentation/installation/cli.html)
- **Tutorial:** [https://ancplaboldenburg.github.io/megqc_documentation/book/tutorial.html](https://ancplaboldenburg.github.io/megqc_documentation/book/tutorial.html)
- **HTML Reports guide:** [https://ancplaboldenburg.github.io/megqc_documentation/book/report.html](https://ancplaboldenburg.github.io/megqc_documentation/book/report.html)
- **Full documentation:** [https://ancplaboldenburg.github.io/megqc_documentation/](https://ancplaboldenburg.github.io/megqc_documentation/)

## Source Code

[https://github.com/ANCPLabOldenburg/MEGqc](https://github.com/ANCPLabOldenburg/MEGqc)

## License

MIT License.
