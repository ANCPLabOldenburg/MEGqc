"""Headless GUI tests (offscreen). Cover construction, the profile-mode controls,
and the settings-dialog grey-out system."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _clean_run_settings(qapp):
    """Isolate the persisted parallel (run/) settings so tests are deterministic
    and never leave values in the real QSettings store."""
    from PyQt6.QtCore import QSettings
    s = QSettings("ANCP", "MEEGqc")
    s.remove("run")
    s.sync()
    yield
    s.remove("run")
    s.sync()


@pytest.fixture
def main_window(qapp):
    from meg_qc.miscellaneous.GUI.megqcGUI import MainWindow
    w = MainWindow()
    yield w
    w.close()


def test_window_constructs(main_window):
    assert main_window.isVisible() in (True, False)  # constructed without raising


def test_analysis_mode_options_are_canonical(main_window):
    cmb = main_window.cmb_analysis_mode
    data = [cmb.itemData(i) for i in range(cmb.count())]
    assert data == ["non-profile", "new-profile", "reuse-profile", "latest-profile"]


def test_non_profile_disables_profile_controls(main_window):
    w = main_window
    w._set_analysis_mode("non-profile")
    assert not w.edit_analysis_id.isEnabled()
    assert not w.btn_load_profiles.isEnabled()
    assert not w.btn_refresh_profiles.isEnabled()

    w._set_analysis_mode("reuse-profile")
    assert w.edit_analysis_id.isEnabled()
    assert w.btn_load_profiles.isEnabled()
    assert w.btn_refresh_profiles.isEnabled()


def test_collect_returns_canonical_mode(main_window):
    main_window._set_analysis_mode("new-profile")
    mode, analysis_id, cfg_policy, sub_policy = main_window._collect_analysis_profile_settings()
    assert mode == "new-profile"
    assert cfg_policy in ("provided", "latest_saved", "stop")
    assert sub_policy in ("skip", "rerun", "stop")


@pytest.fixture
def settings_dialog(qapp):
    import os.path as p
    import meg_qc
    from meg_qc.miscellaneous.GUI.megqcGUI import SettingsEditorDialog
    ini = p.join(p.dirname(meg_qc.__file__), "settings", "settings.ini")
    dlg = SettingsEditorDialog(ini, ini, "test")
    yield dlg
    dlg.close()


def test_metric_switch_greys_whole_section(settings_dialog, qapp):
    """Turning a [GENERAL] metric off greys that metric's whole section box."""
    dlg = settings_dialog
    std = dlg.widgets[("GENERAL", "STD")][0]
    std.setChecked(False); qapp.processEvents()
    assert not dlg.section_boxes["STD"].isEnabled()
    std.setChecked(True); qapp.processEvents()
    assert dlg.section_boxes["STD"].isEnabled()
    # Head defaults to False -> its section is greyed on open.
    assert not dlg.section_boxes["Head_movement"].isEnabled()


def test_compute_gqi_greys_rows_but_keeps_switch(settings_dialog, qapp):
    """compute_gqi=off greys the other GQI rows (widget + label) but the switch stays live."""
    dlg = settings_dialog
    gqi = dlg.widgets[("GlobalQualityIndex", "compute_gqi")][0]
    others = [k for (s, k) in dlg.widgets if s == "GlobalQualityIndex" and k != "compute_gqi"]
    gqi.setChecked(False); qapp.processEvents()
    assert gqi.isEnabled()
    for k in others:
        assert not dlg.widgets[("GlobalQualityIndex", k)][0].isEnabled()
        assert not dlg.labels[("GlobalQualityIndex", k)].isEnabled()


def test_njobs_default_is_safe_and_all_cores_toggles(main_window, qapp):
    """Calc/plot jobs default to 1 (not -1); the 'All cores' checkbox sets -1
    and restores the previous value when unticked."""
    w = main_window
    assert w.jobs.value() == 1
    assert w.plot_jobs.value() == 1
    assert not w.chk_all_cores.isChecked()

    # Use a value within the CPU-capped range: the spinbox max is os.cpu_count()
    # (few cores on CI runners), so a hard-coded 4 would clamp to the maximum.
    n = w.jobs.maximum()
    w.jobs.setValue(n); qapp.processEvents()
    w.chk_all_cores.setChecked(True); qapp.processEvents()
    assert w.jobs.value() == -1 and not w.jobs.isEnabled()
    w.chk_all_cores.setChecked(False); qapp.processEvents()
    assert w.jobs.value() == n and w.jobs.isEnabled()

    w.chk_plot_all_cores.setChecked(True); qapp.processEvents()
    assert w.plot_jobs.value() == -1 and not w.plot_jobs.isEnabled()
    w.chk_plot_all_cores.setChecked(False); qapp.processEvents()
    assert w.plot_jobs.value() == 1 and w.plot_jobs.isEnabled()


def test_reset_parallel_defaults_and_persistence(qapp):
    """Reset-defaults restores 1 core; the parallel settings persist across a
    fresh window."""
    from meg_qc.miscellaneous.GUI.megqcGUI import MainWindow
    w = MainWindow()
    try:
        # Value within the CPU-capped range (spinbox max is os.cpu_count()).
        n = w.jobs.maximum()
        w.jobs.setValue(n); w.chk_plot_all_cores.setChecked(True); qapp.processEvents()
        w2 = MainWindow()  # restores from QSettings
        try:
            assert w2.jobs.value() == n
            assert w2.chk_plot_all_cores.isChecked() and w2.plot_jobs.value() == -1
            w2._reset_parallel_defaults(); qapp.processEvents()
            assert w2.jobs.value() == 1 and not w2.chk_all_cores.isChecked()
            assert w2.plot_jobs.value() == 1 and not w2.chk_plot_all_cores.isChecked()
        finally:
            w2.close()
    finally:
        w.close()


def test_greyout_dims_label_not_just_field(settings_dialog, qapp):
    """The label dims alongside the field (the colour fix)."""
    dlg = settings_dialog
    cmb = dlg.widgets[("Epoching", "epoching_strategy")][0]
    cmb.setCurrentText("fixed"); qapp.processEvents()
    assert not dlg.widgets[("Epoching", "event_dur")][0].isEnabled()
    assert not dlg.labels[("Epoching", "event_dur")].isEnabled()
    assert dlg.widgets[("Epoching", "fixed_epoch_duration")][0].isEnabled()
    cmb.setCurrentText("events"); qapp.processEvents()
    assert dlg.widgets[("Epoching", "event_dur")][0].isEnabled()
    assert not dlg.widgets[("Epoching", "fixed_epoch_duration")][0].isEnabled()



# ── parallelism ──────────────────────────────────────────────────────────
# The old "Hungry jobs" checkbox is gone: parallelism is per recording, so
# --n_jobs already spreads one subject's runs across workers and there is
# nothing left to opt into.

def test_no_hungry_job_control_remains(main_window):
    assert not hasattr(main_window, "chk_hungry_job")


def test_calc_jobs_control_still_present(main_window):
    """--n_jobs is now the only parallelism control, so it must stay wired."""
    assert hasattr(main_window, "jobs")
    assert main_window.jobs.value() >= 1
