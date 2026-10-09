import pytest

import poseidon
from poseidon import provenance


def test_source_snapshot_prevents_claiming_new_disk_code_as_loaded(monkeypatch):
    baseline = dict(poseidon._SOURCE_SNAPSHOT)
    changed = dict(baseline, **{"core.py": "0" * 64})
    monkeypatch.setattr(provenance, "capture_sources", lambda: changed)
    state = provenance.source_status()
    assert state["restart_required"] and not state["source_current"]
    assert state["changed_source_files"] == ["core.py"]
    with pytest.raises(provenance.StaleSourceError, match="Restart"):
        provenance.assert_source_current()


def test_missing_import_snapshot_fails_closed(monkeypatch):
    monkeypatch.delattr(poseidon, "_SOURCE_SNAPSHOT")
    with pytest.raises(provenance.StaleSourceError):
        provenance.assert_source_current()


def test_capture_and_return_are_detached(monkeypatch, tmp_path):
    (tmp_path / "one.py").write_text("x=1\n", encoding="utf-8")
    first = provenance.capture_sources(tmp_path)
    (tmp_path / "one.py").write_text("x=2\n", encoding="utf-8")
    assert provenance.capture_sources(tmp_path) != first
    monkeypatch.setattr(provenance, "capture_sources", lambda: dict(poseidon._SOURCE_SNAPSHOT))
    returned = provenance.assert_source_current()
    returned["core.py"] = "changed"
    assert poseidon._SOURCE_SNAPSHOT["core.py"] != "changed"
