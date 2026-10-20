import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "05_platform"))
import build_site as B


def _src(tmp):
    src = tmp / "docs"
    (src / "data").mkdir(parents=True)
    (src / "field" / "pilot_2024").mkdir(parents=True)
    (src / "index.html").write_text("<html>app</html>")
    (src / "data" / "meta.json").write_text("{}")
    (src / "field" / "index.json").write_text("{}")
    return src


def test_build_site_copies_app_data_and_field(tmp_path):
    src, site = _src(tmp_path), tmp_path / "site"
    (site / "crop-outlook").mkdir(parents=True)
    (site / "crop-outlook" / "old.html").write_text("old")
    r = B.build_site(str(src), str(site), backup_root=str(tmp_path / "bk"))
    co = site / "crop-outlook"
    assert (co / "index.html").exists() and (co / "data" / "meta.json").exists() and (co / "field" / "index.json").exists()
    assert not (co / "old.html").exists()                                   # replaced, not merged
    assert r["files"] == 3


def test_build_site_makes_a_backup(tmp_path):
    src, site = _src(tmp_path), tmp_path / "site"
    (site / "crop-outlook").mkdir(parents=True)
    (site / "crop-outlook" / "old.html").write_text("old")
    r = B.build_site(str(src), str(site), backup_root=str(tmp_path / "bk"))
    assert (Path(r["backup"]) / "crop-outlook" / "old.html").read_text() == "old"


def test_headers_cache_field_files(tmp_path):
    src, site = _src(tmp_path), tmp_path / "site"
    B.build_site(str(src), str(site), backup_root=str(tmp_path / "bk"))
    h = (site / "_headers").read_text()
    assert "/crop-outlook/field/*" in h and "max-age" in h


def test_build_site_refuses_incomplete_source_and_keeps_the_live_folder(tmp_path):
    # review 2026-10-20: docs/data is git-ignored; a fresh checkout has no district data
    import pytest
    src, site = _src(tmp_path), tmp_path / "site"
    (src / "data" / "meta.json").unlink()
    (site / "crop-outlook").mkdir(parents=True)
    (site / "crop-outlook" / "live.html").write_text("live")
    with pytest.raises(FileNotFoundError):
        B.build_site(str(src), str(site), backup_root=str(tmp_path / "bk"))
    assert (site / "crop-outlook" / "live.html").read_text() == "live"


def test_service_worker_is_never_cached(tmp_path):
    # the installable app (PWA): sw.js must always be re-checked so updates reach installed apps
    src, site = _src(tmp_path), tmp_path / "site"
    B.build_site(str(src), str(site), backup_root=str(tmp_path / "bk"))
    h = (site / "_headers").read_text()
    assert "/crop-outlook/sw.js" in h and "no-cache" in h


def test_manifest_and_icons_for_the_installable_app():
    import json
    docs = Path(__file__).resolve().parents[2] / "docs"
    m = json.load(open(docs / "manifest.webmanifest", encoding="utf-8"))
    assert m["display"] == "standalone" and m["start_url"] and m["name"]
    sizes = {i["sizes"] for i in m["icons"]}
    assert {"192x192", "512x512"} <= sizes
    for i in m["icons"]:
        assert (docs / i["src"]).exists(), i["src"]
    assert (docs / "sw.js").exists()
