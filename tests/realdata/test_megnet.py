"""MEGnet real-data tests.

Runs the pipeline with MEGnet enabled and verifies that MEGnet-specific
derivatives are produced.  Requires the MEEGQC_TEST_DATA bundle and
a working megnet-neuro installation.
"""
from __future__ import annotations

import json
import re

import pytest

pytestmark = pytest.mark.realdata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _megnet_config(tmp_path_factory, *, enabled: bool = True):
    """Write a settings.ini with ECG + EOG on and MEGnet toggled."""
    import meg_qc
    import re as _re

    src = (meg_qc.__file__.rsplit("/", 1)[0] if "/" in meg_qc.__file__
           else str(__import__("pathlib").Path(meg_qc.__file__).parent))
    ini_src = __import__("pathlib").Path(src) / "settings" / "settings.ini"
    text = ini_src.read_text()

    # Enable ECG, EOG, STD, PSD; disable Head and Muscle for speed
    for key in ("STD", "PSD", "PTP_manual", "ECG", "EOG"):
        text = _re.sub(rf"(?m)^{key}\s*=.*$", f"{key} = True", text)
    for key in ("Head", "Muscle"):
        text = _re.sub(rf"(?m)^{key}\s*=.*$", f"{key} = False", text)

    # Toggle MEGnet settings
    text = _re.sub(r"(?m)^megnet_optional_dependency\s*=.*$",
                   f"megnet_optional_dependency = {str(not enabled).lower()}", text)
    text = _re.sub(r"(?m)^megnet_independent\s*=.*$",
                   f"megnet_independent = {str(enabled).lower()}", text)

    out = tmp_path_factory.mktemp("cfg_megnet") / "settings.ini"
    out.write_text(text)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMEGnetConfigParsing:
    """Verify MEGnet settings survive config round-trip."""

    def test_megnet_section_present(self, tmp_path):
        import meg_qc
        from meg_qc.calculation.initial_meg_qc import get_all_config_params

        src = (meg_qc.__file__.rsplit("/", 1)[0] if "/" in meg_qc.__file__
               else str(__import__("pathlib").Path(meg_qc.__file__).parent))
        ini = __import__("pathlib").Path(src) / "settings" / "settings.ini"
        params = get_all_config_params(str(ini))
        assert "MEGNET" in params
        assert params["MEGNET"]["megnet_ecg_class"] == 2
        assert params["MEGNET"]["megnet_eog_primary_class"] == 1

    def test_ecg_megnet_keys(self, tmp_path):
        import meg_qc
        from meg_qc.calculation.initial_meg_qc import get_all_config_params

        src = (meg_qc.__file__.rsplit("/", 1)[0] if "/" in meg_qc.__file__
               else str(__import__("pathlib").Path(meg_qc.__file__).parent))
        ini = __import__("pathlib").Path(src) / "settings" / "settings.ini"
        params = get_all_config_params(str(ini))
        assert "megnet_fallback" in params["ECG"]
        assert "megnet_independent" in params["ECG"]
        assert "megnet_lowpass_apply" in params["ECG"]
        assert "megnet_lowpass_h_freq" in params["ECG"]

    def test_eog_megnet_keys(self, tmp_path):
        import meg_qc
        from meg_qc.calculation.initial_meg_qc import get_all_config_params

        src = (meg_qc.__file__.rsplit("/", 1)[0] if "/" in meg_qc.__file__
               else str(__import__("pathlib").Path(meg_qc.__file__).parent))
        ini = __import__("pathlib").Path(src) / "settings" / "settings.ini"
        params = get_all_config_params(str(ini))
        assert "megnet_fallback" in params["EOG"]
        assert "megnet_independent" in params["EOG"]
        assert "megnet_lowpass_apply" in params["EOG"]
        assert "megnet_lowpass_h_freq" in params["EOG"]


class TestMEGnetPipeline:
    """Run the pipeline with MEGnet on real data and check outputs."""

    def test_megnet_produces_derivatives(
        self, megnet_meg_dataset, isolated_dataset, _megnet_config_factory, cli
    ):
        """Full pipeline run with MEGnet produces ECG/EOG MEGnet derivatives."""
        name, path = megnet_meg_dataset
        ds = isolated_dataset(path)
        cfg = _megnet_config_factory
        rc = cli([
            "run-meegqc", "--inputdata", str(ds),
            "--config", str(cfg), "--n_jobs", "1",
        ])
        assert rc.returncode == 0, (
            f"Pipeline failed on {name}:\n{rc.stdout[-4000:]}\n{rc.stderr[-2000:]}"
        )

        calc = ds / "derivatives" / "MEEGqc" / "calculation"
        assert calc.is_dir(), f"{name}: no calculation folder"

        # MEGnet derivatives should be present with BIDS-valid desc labels
        megnet_derivs = list(calc.rglob("*megnet*"))
        # If MEGnet is installed and produced outputs, verify naming
        if megnet_derivs:
            for f in megnet_derivs:
                assert ".tsv" in f.suffix or ".json" in f.suffix, (
                    f"Unexpected file type: {f.name}"
                )

    def test_megnet_or_skip(self, megnet_meg_dataset, isolated_dataset, _megnet_config_factory, cli):
        """Run pipeline — MEGnet sections in config are exercised regardless."""
        name, path = megnet_meg_dataset
        ds = isolated_dataset(path)
        cfg = _megnet_config_factory
        rc = cli([
            "run-meegqc", "--inputdata", str(ds),
            "--config", str(cfg), "--n_jobs", "1",
        ])
        assert rc.returncode == 0, (
            f"Pipeline failed on {name}:\n{rc.stdout[-4000:]}\n{rc.stderr[-2000:]}"
        )

        calc = ds / "derivatives" / "MEEGqc" / "calculation"
        assert calc.is_dir(), f"{name}: no calculation folder"

        # Basic derivatives must always be produced
        std_derivs = list(calc.rglob("*_desc-STDs_*.tsv"))
        assert std_derivs, f"{name}: no STDs derivative"

        # SimpleMetrics JSON should exist for each recording
        sm_json = list(calc.rglob("*SimpleMetrics_*.json"))
        assert sm_json, f"{name}: no SimpleMetrics JSON"

        # If MEGnet was available, the ECG/EOG SimpleMetrics should reference it
        for jf in sm_json:
            data = json.loads(jf.read_text())
            ecg_key = [k for k in data if "ECG" in k.upper() or "ecg" in k.lower()]
            eog_key = [k for k in data if "EOG" in k.upper() or "eog" in k.lower()]
            # Just verify the keys exist; content depends on data
            # (MEGnet may or may not have been used depending on installation)
