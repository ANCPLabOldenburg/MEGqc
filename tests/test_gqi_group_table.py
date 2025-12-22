import csv
import json
import sys
import types

mne_stub = types.SimpleNamespace()
mne_stub.io = types.SimpleNamespace(Raw=object)
sys.modules["mne"] = mne_stub
sys.modules.setdefault("plotly", types.SimpleNamespace())

import meg_qc.calculation.metrics.summary_report_GQI as gqi_module


def test_group_table_includes_task_column(tmp_path, monkeypatch):
    derivatives_root = tmp_path / "derivatives"
    calc_dir = derivatives_root / "Meg_QC" / "calculation" / "sub-01"
    calc_dir.mkdir(parents=True)

    # Create a SimpleMetrics file that carries a task label in its name
    simple_file = calc_dir / "sub-01_task-rest_desc-SimpleMetrics_meg.json"
    simple_file.write_text("{}")

    config_path = tmp_path / "config.ini"
    config_path.write_text("[GlobalQualityIndex]\ncompute_gqi = true\n")

    # Simplify the summary generation to focus on group table construction
    monkeypatch.setattr(
        gqi_module, "get_all_config_params", lambda _: {"GlobalQualityIndex": {}}
    )

    def fake_create_summary_report(json_file, html_output, json_output, gqi_params):
        del json_file, html_output, gqi_params
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump({"dummy": True}, f)

    monkeypatch.setattr(gqi_module, "create_summary_report", fake_create_summary_report)

    dummy_pipeline = types.SimpleNamespace(flatten_summary_metrics=lambda js: {"GQI": 1})
    sys.modules["meg_qc.calculation.meg_qc_pipeline"] = dummy_pipeline

    gqi_module.generate_gqi_summary(
        str(tmp_path / "dataset"), str(derivatives_root), str(config_path)
    )

    tsv_path = (
        derivatives_root
        / "Meg_QC"
        / "summary_reports"
        / "group_metrics"
        / "Global_Quality_Index_attempt_1.tsv"
    )

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert reader.fieldnames[0] == "task"
    assert rows[0]["task"] == "rest"
