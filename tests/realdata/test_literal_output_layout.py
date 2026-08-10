"""End-to-end coverage for literal external output."""

import pytest

pytestmark = pytest.mark.realdata


def test_literal_output_calculation_and_plotting(
    one_meg,
    isolated_dataset,
    fast_config,
    cli,
    tmp_path,
):
    dataset = isolated_dataset(one_meg[1])
    output = tmp_path / "quality-control"
    common = [
        "--inputdata",
        str(dataset),
        "--derivatives_output",
        str(output),
        "--output_layout",
        "literal",
    ]

    calculation = cli(
        ["run-meegqc", *common, "--config", str(fast_config), "--n_jobs", "1"]
    )
    assert calculation.returncode == 0, calculation.stdout[-3000:]

    calculation_root = output / "MEEGqc" / "calculation"
    assert calculation_root.is_dir()
    assert list(calculation_root.rglob("*_desc-STDs_*.tsv"))
    assert not (output / dataset.name).exists()
    assert not (output / "derivatives").exists()

    plotting = cli(["run-meegqc-plotting", *common, "--qa-subject"])
    assert plotting.returncode == 0, plotting.stdout[-3000:]
    reports = output / "MEEGqc" / "reports"
    assert list(reports.glob("*/sub-*/*subjectQaReport*.html"))
