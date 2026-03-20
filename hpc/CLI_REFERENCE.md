# MEGqc — HPC and CBRAIN CLI Reference

> **Assumes** `MEGqc.sif` is already built.
> See [`BUILD_APPTAINER_IMAGE.md`](BUILD_APPTAINER_IMAGE.md) if you have not built the image yet.

This reference documents all MEGqc CLI workflows for HPC environments.
It was developed as part of the MEGqc integration into
**CBRAIN** ([https://cbrain.ca/](https://cbrain.ca/)), the distributed computing
platform at the Montreal Neurological Institute (MNI) / McGill University.
All commands work unmodified on any Linux HPC scheduler (Slurm, PBS, SGE, bare-metal).

---

## Path conventions

| Host path | Container mount | Purpose |
|-----------|----------------|---------|
| `./settings.ini` | `/mnt_config/settings.ini` | Configuration file |
| `/path/to/bids_dataset` | `/mnt_IN` | Input BIDS dataset |
| `./outputs` | `/mnt_OUT` | Output derivatives (external, keeps source dataset clean) |

> **Tip:** without `--derivatives_output` the results land inside the BIDS
> dataset at `<dataset>/derivatives/Meg_QC/`. Using `--derivatives_output`
> (mapped to `/mnt_OUT`) keeps the source dataset untouched — recommended on CBRAIN.

---

## 0. Before you start — export the config

```bash
apptainer exec \
  --containall \
  --bind "$PWD":/mnt_config \
  MEGqc.sif \
  get-megqc-config --target_directory /mnt_config
```

Edit `settings.ini` as needed. Useful settings for quick testing:

```ini
[GENERAL]
data_crop_tmin = 0
data_crop_tmax = 10   # crop to 10 s for fast iteration
```

---

## 1. Full profiled run — recommended workflow

Runs calculation + all QA and QC reports in one shot.
Results are stored under a timestamped profile directory.

```bash
mkdir -p outputs

apptainer run \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --analysis_mode new \
  --run-all --all
```

The analysis profile ID is auto-generated (`YYYYMMDD_HHMMSS`).
Results: `outputs/mnt_IN/derivatives/Meg_QC/profiles/<analysis_id>/`

---

## 2. Single-subject test run

Fastest way to verify the container works on your data:

```bash
apptainer run \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --subs 009 \
  --analysis_mode new \
  --run-all --qa-subject
```

`--qa-subject` generates only per-subject HTML reports (no group statistics),
which is faster and sufficient for a smoke test.

---

## 3. Multiple specific subjects

```bash
apptainer run \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --subs 009 012 015 \
  --analysis_mode new \
  --run-all --all
```

---

## 4. Parallel multi-subject run

```bash
apptainer run \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --n_jobs 4 \
  --analysis_mode new \
  --run-all --all
```

> **RAM guide:** 8 GB = 1 job / 16 GB = 2 jobs / 32 GB = 6 jobs / 64 GB = 16 jobs

On CBRAIN or large Slurm allocations, increase `--n_jobs` proportionally
to take full advantage of the available cores. See the Slurm template in
section 14 for a practical example.

---

## 5. Multiple datasets in one call

```bash
apptainer run \
  --containall \
  --bind /path/to/ds_1:/mnt_ds1 \
  --bind /path/to/ds_camcan:/mnt_ds2 \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_ds1 /mnt_ds2 \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --analysis_mode new \
  --run-all --all
```

---

## 6. Per-dataset config and subject overrides

```bash
apptainer run \
  --containall \
  --bind /path/to/ds_1:/mnt_ds1 \
  --bind /path/to/ds_camcan:/mnt_ds2 \
  --bind "$PWD/settings_ds1.ini":/mnt_cfg1/settings.ini \
  --bind "$PWD/settings_ds2.ini":/mnt_cfg2/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_ds1 /mnt_ds2 \
  --config_per_dataset \
      /mnt_ds1::/mnt_cfg1/settings.ini \
      /mnt_ds2::/mnt_cfg2/settings.ini \
  --subs_per_dataset \
      /mnt_ds1::009,012,015 \
      /mnt_ds2::all \
  --derivatives_output /mnt_OUT \
  --analysis_mode new \
  --run-all --all
```

---

## 7. Analysis modes

| Mode | When to use |
|------|-------------|
| `new` | Default — creates a new timestamped profile |
| `reuse` | Re-run plotting on an existing profile (requires `--analysis_id`) |
| `latest` | Auto-resolves the most recently modified profile |

### Reuse an existing profile (regenerate or add reports)

```bash
apptainer run \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --analysis_mode reuse \
  --analysis_id 20260318_094626 \
  --run-all --all
```

---

## 8. run-megqc-plotting — plotting only (no recalculation)

Useful when calculation is already done and you only want to regenerate
reports with different settings or an updated configuration.

```bash
apptainer exec \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  run-megqc-plotting \
  --inputdata /mnt_IN \
  --derivatives_output /mnt_OUT \
  --analysis_mode latest \
  --all
```

### Subject reports only

```bash
apptainer exec \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  run-megqc-plotting \
  --inputdata /mnt_IN \
  --derivatives_output /mnt_OUT \
  --analysis_mode latest \
  --qa-subject
```

### Group reports only

```bash
apptainer exec \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  run-megqc-plotting \
  --inputdata /mnt_IN \
  --derivatives_output /mnt_OUT \
  --analysis_mode latest \
  --qa-group
```

### QC group and multisample reports

```bash
apptainer exec \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  run-megqc-plotting \
  --inputdata /mnt_IN \
  --derivatives_output /mnt_OUT \
  --analysis_mode latest \
  --qc-group --qc-multisample
```

---

## 9. globalqualityindex — GQI summary

```bash
apptainer exec \
  --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  globalqualityindex \
  --inputdata /mnt_IN \
  --derivatives_output /mnt_OUT \
  --analysis_mode latest
```

---

## 10. Plotting flags reference

| Flag | Scope | Reports generated |
|------|-------|-------------------|
| `--qa-subject` | Per subject | Subject-level HTML reports |
| `--qa-group` | Per dataset | Group-level QA summary |
| `--qa-multisample` | Across datasets | Cross-dataset QA (2+ datasets required) |
| `--qa-all` | All QA | Subject + Group + Multisample |
| `--qc-group` | Per dataset | Group-level QC summary |
| `--qc-multisample` | Across datasets | Cross-dataset QC (2+ datasets required) |
| `--qc-all` | All QC | Group + Multisample |
| `--all` | Everything | All QA + all QC |

> Add `--run-all` to `run-megqc` to invoke plotting immediately after
> calculation finishes. The scope of reports is determined by the pairing flag:
> `--run-all --all` = all reports, `--run-all --qa-subject` = subject reports only.
> Without a pairing flag, `--run-all` alone has no effect.

---

## 11. Subject / processed data policies

### Skip already-processed subjects (default)

```bash
  --processed_subjects_policy skip
```

### Force reprocessing of all subjects

```bash
  --processed_subjects_policy rerun
```

### Use the saved config from the profile (instead of the one you provide)

```bash
  --existing_config_policy latest_saved
```

---

## 12. Error handling and debugging

### Keep temp files on error

```bash
  --keep-temp-on-error
```

Intermediate `.tmp` files are preserved in the profile folder for inspection.

### Run entirely headless (explicit override)

These environment variables are already set by default inside the container
(`MEGQC.def` `%environment` block), but you can set them explicitly if needed:

```bash
apptainer exec \
  --containall \
  --env QT_QPA_PLATFORM=offscreen \
  --env MPLBACKEND=Agg \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  run-megqc \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --analysis_mode new \
  --run-all --all
```

---

## 13. Launch the GUI (desktop only)

> The GUI is **not available on headless HPC or CBRAIN nodes**.
> All other commands in this file use `--containall` for strict isolation.
> The GUI command intentionally does not.

```bash
apptainer exec \
  --env QT_QPA_PLATFORM=xcb \
  --bind /tmp/.X11-unix:/tmp/.X11-unix \
  --bind /etc/machine-id:/etc/machine-id:ro \
  --bind /run/user/$(id -u):/run/user/$(id -u) \
  --bind "${XAUTHORITY:-$HOME/.Xauthority}":/tmp/.Xauthority:ro \
  --bind "$HOME":"$HOME" \
  MEGqc.sif \
  megqc
```

Without `--containall`, Apptainer inherits the full host environment
automatically: `$DISPLAY`, `$XDG_RUNTIME_DIR`, `$DBUS_SESSION_BUS_ADDRESS`,
and `$XAUTHORITY` are all passed in with no extra flags. Python isolation
is still guaranteed by the hardcoded shebang `#!/opt/conda/bin/python3.10`.

---

## 14. CBRAIN / Slurm job script template

The following script is the recommended starting point for submitting MEGqc
jobs on CBRAIN or any Slurm-based HPC cluster.

```bash
#!/bin/bash
#SBATCH --job-name=megqc
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

module load apptainer   # or: module load singularity

BIDS=/scratch/$USER/bids_dataset
SIF=/scratch/$USER/containers/MEGqc.sif
CONFIG=/scratch/$USER/containers/settings.ini
OUT=/scratch/$USER/megqc_results

mkdir -p "$OUT"

apptainer run \
  --containall \
  --bind "${BIDS}":/mnt_IN \
  --bind "${CONFIG}":/mnt_config/settings.ini \
  --bind "${OUT}":/mnt_OUT \
  "${SIF}" \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --n_jobs 6 \
  --analysis_mode new \
  --run-all --all
```

Adjust `--mem`, `--cpus-per-task`, and `--n_jobs` according to your allocation
and cohort size. A practical guide:

| RAM allocation | Recommended `--n_jobs` |
|----------------|------------------------|
| 8 GB | 1 |
| 16 GB | 2 |
| 32 GB | 6 |
| 64 GB | 16 |
| 128 GB+ | 32+ |

---

*Built and tested on Ubuntu 22.04 LTS · Apptainer 1.4.5 · meg_qc 0.7.8*
*Maintainer: karel.mauricio.lopez.vilaret@uni-oldenburg.de*
*Original CBRAIN integration: Alexandre Pastor-Bernier, MNI/McGill, March 2026*

