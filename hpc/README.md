# MEGqc — HPC and CBRAIN Deployment

This folder contains everything needed to run MEGqc on high-performance computing (HPC)
infrastructure using [Apptainer](https://apptainer.org/) (formerly Singularity) containers.

---

## CBRAIN collaboration

This containerization effort is developed in collaboration with the
**Montreal Neurological Institute (MNI) / McGill University** as part of the
integration of MEGqc into [CBRAIN](https://cbrain.ca/), a web-enabled distributed
computing platform that provides an accessible interface for running neuroimaging
workflows on HPC and cloud resources across multiple sites.

The Apptainer image and the workflows documented here are the **official deployment
model for MEGqc on CBRAIN**. The same image and CLI commands work on any Linux HPC
system (Slurm, PBS, SGE, and bare-metal) without modification.

---

## Contents

| File | Description |
|------|-------------|
| [`MEGQC.def`](MEGQC.def) | Apptainer definition file — the single source of truth for building the container image |
| [`BUILD_APPTAINER_IMAGE.md`](BUILD_APPTAINER_IMAGE.md) | Step-by-step guide: prerequisites, build, GUI launch, image distribution (Sylabs, GHCR, CBRAIN) |
| [`CLI_REFERENCE.md`](CLI_REFERENCE.md) | Full CLI reference: all run modes, flags, analysis modes, and a ready-to-use CBRAIN/Slurm job script template |

---

## Quick start (three commands)

```bash
# 1. Build the image
apptainer build MEGqc.sif hpc/MEGQC.def

# 2. Export the default settings.ini
apptainer exec --containall --bind "$PWD":/mnt_config MEGqc.sif \
  get-megqc-config --target_directory /mnt_config

# 3. Run full analysis on a BIDS dataset
apptainer run --containall \
  --bind /path/to/bids_dataset:/mnt_IN \
  --bind "$PWD/settings.ini":/mnt_config/settings.ini \
  --bind "$PWD/outputs":/mnt_OUT \
  MEGqc.sif \
  --inputdata /mnt_IN \
  --config /mnt_config/settings.ini \
  --derivatives_output /mnt_OUT \
  --analysis_mode new --run-all --all
```

See [`BUILD_APPTAINER_IMAGE.md`](BUILD_APPTAINER_IMAGE.md) and
[`CLI_REFERENCE.md`](CLI_REFERENCE.md) for full documentation.

---

## Compatibility

| Platform | Supported |
|----------|-----------|
| Linux HPC (x86) — Slurm, PBS, SGE | Yes |
| CBRAIN (MNI/McGill) | Yes — primary deployment target |
| macOS / Windows (local desktop) | Build on remote Linux; run GUI locally via X11 forwarding |
| Headless nodes (no display) | Yes — `QT_QPA_PLATFORM=offscreen` is the baked-in default |

---

## Attribution

The original Singularity definition file and the CBRAIN integration workflow were
initiated by **Alexandre Pastor-Bernier** (MNI / McGill University,
alexandre.pastor@mcgill.ca) in March 2026. The production definition file
(`MEGQC.def`), Qt/headless fixes, OS library dependencies, and the full tutorial
documentation were developed by the **Applied Neurocognitive Psychology Lab (ANCPLab),
University of Oldenburg**, in collaboration with MNI/McGill.

---

*Maintainer: karel.mauricio.lopez.vilaret@uni-oldenburg.de*

