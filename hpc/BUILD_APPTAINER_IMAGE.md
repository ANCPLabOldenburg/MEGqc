# MEGqc — Apptainer Image Build Guide

> **Goal:** Build a single `MEGqc.sif` Apptainer image from `MEGQC.def` that
> runs `meg_qc` on any Linux HPC system — including **CBRAIN** (MNI/McGill),
> Slurm, PBS, and bare-metal servers — without root privileges.
> The same image also supports launching the MEGqc GUI on a local desktop.

This guide and the accompanying `MEGQC.def` are the official deployment artifacts
for MEGqc on **CBRAIN** ([https://cbrain.ca/](https://cbrain.ca/)), developed in
collaboration with the Montreal Neurological Institute (MNI) / McGill University.

---

## 1. Prerequisites

### 1.1 Operating system

Apptainer runs on **Linux only**.
macOS / Windows users must work on a remote Linux machine, VM, or WSL 2.

### 1.2 Install Apptainer >= 1.1

**Ubuntu / Debian** (recommended: install the official release `.deb`):

```bash
# Get latest release tag
TAG=$(curl -fsSL https://api.github.com/repos/apptainer/apptainer/releases/latest \
      | grep -o '"tag_name": *"[^"]*"' | head -1 | tr -d '"tag_name": ')

wget "https://github.com/apptainer/apptainer/releases/download/${TAG}/apptainer_${TAG#v}_amd64.deb"
sudo apt-get install -y "./apptainer_${TAG#v}_amd64.deb"
```

**Rocky Linux / RHEL / CentOS**:

```bash
sudo dnf install -y apptainer
```

**Verify**:

```bash
apptainer --version   # must be >= 1.1
```

### 1.3 Disk space and memory

| Resource | Minimum |
|----------|---------|
| Free disk (build) | ~5 GB (cache + image) |
| Final image | ~1 GB |
| RAM (build) | 4 GB |
| RAM (run) | 8 GB (16 GB recommended) |

---

## 2. Get the definition file

The only file needed to build the image is **`MEGQC.def`**, located at:

```
MEGqc/
└── hpc/
    ├── MEGQC.def                       <- the only file needed at build time
    ├── BUILD_APPTAINER_IMAGE.md        <- this file
    └── CLI_REFERENCE.md
```

Clone the MEGqc repository or download `MEGQC.def` directly:

```bash
# Clone the full repository
git clone https://github.com/ANCPLabOldenburg/MEGqc.git
cd MEGqc/hpc

# Or download only the definition file
curl -O https://raw.githubusercontent.com/ANCPLabOldenburg/MEGqc/main/hpc/MEGQC.def
```

---

## 3. Build the image

```bash
cd MEGqc/hpc
apptainer build MEGqc.sif MEGQC.def
```

**What happens inside `MEGQC.def`:**

| Step | Action |
|------|--------|
| Bootstrap | Pulls `continuumio/miniconda3` (Debian amd64) from Docker Hub |
| Python | Installs Python 3.10 via conda |
| OS libs | Installs all required shared libraries (OpenGL, Qt6/XCB, EGL, D-Bus ...) |
| meg_qc | `pip install "meg_qc==0.7.8"` |
| Smoke tests | Runs all 4 CLI entry points with `--help` — build fails fast if broken |
| Environment | Sets `QT_QPA_PLATFORM=offscreen` + `MPLBACKEND=Agg` (headless HPC default) |
| Runscript | `apptainer run MEGqc.sif ...` maps directly to `run-megqc ...` |

Expected output (abbreviated):

```
INFO:    Starting build...
INFO:    Fetching OCI image...
...
+ run-megqc --help
+ run-megqc-plotting --help
+ globalqualityindex --help
+ get-megqc-config --help
INFO:    Build complete: MEGqc.sif
```

Build time: 5-15 minutes depending on network speed.
Final image size: ~1 GB.

---

## 4. Verify the image

```bash
# Check version label
apptainer inspect MEGqc.sif | grep Version

# Quick CLI smoke tests
apptainer run MEGqc.sif --help
apptainer exec MEGqc.sif run-megqc-plotting --help
apptainer exec MEGqc.sif globalqualityindex --help
apptainer exec MEGqc.sif get-megqc-config --help
```

---

## 5. Export the default settings.ini

```bash
apptainer exec \
  --containall \
  --bind "$PWD":/mnt_config \
  MEGqc.sif \
  get-megqc-config --target_directory /mnt_config
```

This writes `settings.ini` to the current directory.
Edit it freely — the container image is not modified.

---

## 6. Launch the GUI (desktop only)

> The GUI requires a real display. It is **not available on headless HPC or CBRAIN nodes**.
> Use the CLI commands in [`CLI_REFERENCE.md`](CLI_REFERENCE.md) for unattended runs.

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

**Why these flags:**

| Flag | CLI / HPC | GUI (desktop) |
|------|-----------|---------------|
| `--containall` | Required — strict isolation | Not used — GUI needs host filesystem access |
| `--bind $HOME:$HOME` | Not needed | Required — allows the file browser to open datasets |
| Host env inherited | Blocked by `--containall` | `$DISPLAY`, `$XDG_RUNTIME_DIR`, `$DBUS_SESSION_BUS_ADDRESS`, `$XAUTHORITY` all passed automatically |
| Python isolation | Via `--containall` | Via hardcoded shebang `#!/opt/conda/bin/python3.10` |

---

## 7. Distribute the image

### Deploy to an HPC / CBRAIN server

```bash
scp MEGqc.sif user@hpc-server:/scratch/user/containers/
```

### Push to Sylabs Cloud Library

```bash
apptainer push MEGqc.sif library://your_sylabs_user/default/meg_qc:0.7.8
```

### Push to GitHub Container Registry (GHCR)

```bash
echo "$GITHUB_TOKEN" | apptainer remote login --username your_gh_user \
  --password-stdin oras://ghcr.io
apptainer push MEGqc.sif oras://ghcr.io/your_gh_user/meg_qc:0.7.8
```

---

## 8. Rebuild and update tips

### Rebuild from scratch (ignore cache)

```bash
apptainer build --force MEGqc.sif MEGQC.def
```

### Debug in sandbox mode (writable filesystem)

```bash
apptainer build --sandbox MEGqc_sandbox/ MEGQC.def
apptainer shell --fakeroot --writable MEGqc_sandbox/
# inspect, fix, and test interactively, then:
apptainer build MEGqc_fixed.sif MEGqc_sandbox/
```

### Change the meg_qc version

Edit two lines in `MEGQC.def`:

```
%labels
    Version     "0.7.9"          <- update version label

%post
    pip install "meg_qc==0.7.9"  <- update pip install
```

Then rebuild.

---

## 9. Key definition file sections explained

```apptainer
Bootstrap: docker
From: continuumio/miniconda3       # Debian-based base image
```

```apptainer
%environment
    export QT_QPA_PLATFORM=offscreen   # headless default for CBRAIN / HPC
    export MPLBACKEND=Agg              # matplotlib headless
```

Without `QT_QPA_PLATFORM=offscreen` the container crashes on HPC nodes with:

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
ImportError: libEGL.so.1: cannot open shared object file
```

This was the bug first encountered during the CBRAIN integration effort by
Alexandre Pastor-Bernier (MNI/McGill). It is permanently fixed in `MEGQC.def`.

---

## 10. Files produced by this workflow

After a successful run, outputs are written to whatever `--derivatives_output`
folder you specify on the host. See [`CLI_REFERENCE.md`](CLI_REFERENCE.md)
for the full path conventions and output structure.

---

*Built and tested on Ubuntu 22.04 LTS · Apptainer 1.4.5 · meg_qc 0.7.8*
*Maintainer: karel.mauricio.lopez.vilaret@uni-oldenburg.de*
*Original CBRAIN integration: Alexandre Pastor-Bernier, MNI/McGill, March 2026*

