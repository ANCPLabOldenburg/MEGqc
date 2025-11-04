"""Integration test covering MEG-QC calculation and plotting pipelines."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

try:  # pragma: no cover - dependency availability handled at runtime
    import ancpbids  # type: ignore
except ImportError:  # pragma: no cover
    ancpbids = None
    make_derivative_meg_qc = None
    make_plots_meg_qc = None
else:  # pragma: no cover - imported only when dependencies are available
    from meg_qc.calculation.meg_qc_pipeline import make_derivative_meg_qc
    from meg_qc.plotting.meg_qc_plots import make_plots_meg_qc


class TestMegPipeline(unittest.TestCase):
    """Run the MEG-QC calculation and plotting pipeline on the test dataset."""

    maxDiff = None

    def setUp(self) -> None:
        if ancpbids is None or make_derivative_meg_qc is None:
            self.skipTest("ancpbids is required to run the MEG-QC pipeline")

        self.repo_root = Path(__file__).resolve().parents[1]
        self.template_dataset = (
            self.repo_root / "tests" / "data" / "meg_datasets" / "ds_meg1"
        )
        if not self.template_dataset.exists():
            self.fail(f"Template dataset not found: {self.template_dataset}")

        self.config_file_path = self.repo_root / "meg_qc" / "settings" / "settings.ini"
        self.internal_config_file_path = (
            self.repo_root / "meg_qc" / "settings" / "settings_internal.ini"
        )

        # Work in an isolated temp copy of the dataset
        self.temp_dir = Path(tempfile.mkdtemp()).resolve()
        self.data_directory = (self.temp_dir / "ds_meg1").resolve()
        shutil.copytree(self.template_dataset, self.data_directory)

        # --- WINDOWS FIX ---
        # Ensure 'derivatives/' exists before any ancpbids query(scope='derivatives')
        # Some ancpbids versions return None if this folder is missing.
        (self.data_directory / "derivatives").mkdir(exist_ok=True)

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, onerror=self._handle_remove_readonly)

    def test_meg_pipeline_calculation_and_plotting(self) -> None:
        """The calculation stage feeds into the plotting stage successfully."""

        make_derivative_meg_qc(
            str(self.config_file_path),
            str(self.internal_config_file_path),
            str(self.data_directory),
            ["009"],
            n_jobs=1,
        )

        derivatives_root = self.data_directory / "derivatives" / "Meg_QC"
        calculation_dir = derivatives_root / "calculation"
        self.assertTrue(
            calculation_dir.exists(),
            "Calculation step should create derivatives/Meg_QC/calculation",
        )

        make_plots_meg_qc(str(self.data_directory), n_jobs=1)

        reports_root = derivatives_root / "reports"
        subject_reports = reports_root / "sub-009"
        self.assertTrue(
            subject_reports.exists(),
            "Plotting step should create subject-specific reports",
        )
        self.assertTrue(
            any(subject_reports.rglob("*.html")),
            "Expected HTML reports were not generated",
        )

    @staticmethod
    def _handle_remove_readonly(func, path, exc_info):
        """Ensure read-only files can be removed during cleanup on Windows."""
        del exc_info  # unused but part of the interface
        os.chmod(path, stat.S_IWRITE)
        func(path)


if __name__ == "__main__":
    unittest.main()
