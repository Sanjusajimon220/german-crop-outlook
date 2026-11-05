"""retry(): waits and retries on rate limits (429) and temporary server errors (5xx); other errors raise at once."""
import pytest

import phase7_step2_download as D


def _flaky(msg, fails):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= fails:
            raise RuntimeError(msg)
        return "ok"
    return fn, calls


@pytest.mark.parametrize("msg", ["[429] Too Many Requests", "[500] ", "[502] Bad Gateway", "[503] Service Unavailable"])
def test_retries_transient(monkeypatch, msg):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fn, calls = _flaky(msg, 2)
    assert D.retry(fn) == "ok" and calls["n"] == 3


def test_other_errors_raise_at_once(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fn, calls = _flaky("[404] Not Found", 1)
    with pytest.raises(RuntimeError):
        D.retry(fn)
    assert calls["n"] == 1
