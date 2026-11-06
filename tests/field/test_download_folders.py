"""phase10_download.pending(): which product folders still need a job (s2-only option for the held-out water test)."""
import os

import phase10_download as D


def test_pending_skips_complete_and_respects_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "raw_dir", lambda a, y: str(tmp_path))
    os.makedirs(tmp_path / "s1")
    open(tmp_path / "s1" / ".complete", "w").write("5 files\n")
    assert D.pending("dekli", 2017) == ["s2", "s1_desc"]
    assert D.pending("dekli", 2017, folders=("s2",)) == ["s2"]
    os.makedirs(tmp_path / "s2")
    open(tmp_path / "s2" / ".complete", "w").write("5 files\n")
    assert D.pending("dekli", 2017, folders=("s2",)) == []
