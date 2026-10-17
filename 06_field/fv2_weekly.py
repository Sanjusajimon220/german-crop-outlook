"""Weekly pixel series per area-season with the tested design (plan 2026-10-17, Task 2).

Per pixel and Monday: measured (clear S2 value within +-3 days, nearest) > straight line (gap <= 30 d) > radar V2
re-analysis (gap > 30 d; line instead if the area test did not adopt the design) > live V2 after the last clear image.
Run `python fv2_weekly.py all` from D:\vista (resumable).
"""
import numpy as np

SRC_NONE, SRC_MEASURED, SRC_LINE, SRC_RADAR, SRC_AFTER = 0, 1, 2, 3, 4
MEASURED_DAYS = 3
LINE_MAX_GAP = 30


def assign_week(days, values, sd, week_day, models=None):
    """One pixel, one week. days / values / sd: its clear S2 values (ascending days). Returns (value, unc, src, gap).
    models: optional callables {"re": f(a, b) -> value, "live": f(a) -> value} for the radar cases (indices into days);
    without them the radar cases fall back to the straight line / last value (used by the pure tests)."""
    if len(days) == 0:
        return np.nan, np.nan, SRC_NONE, 0
    d = np.abs(days - week_day)
    k = int(np.argmin(d))
    if d[k] <= MEASURED_DAYS:
        return float(values[k]), float(sd[k]), SRC_MEASURED, 0
    before, after = np.flatnonzero(days < week_day), np.flatnonzero(days > week_day)
    if len(before) == 0:
        return np.nan, np.nan, SRC_NONE, 0
    a = before[-1]
    if len(after) == 0:
        v = models["live"](a) if models else values[a]
        return float(v), np.nan, SRC_AFTER, int(week_day - days[a])
    b = after[0]
    gap = int(days[b] - days[a])
    line = values[a] + (week_day - days[a]) / gap * (values[b] - values[a])
    if gap <= LINE_MAX_GAP:
        return float(line), np.nan, SRC_LINE, gap
    v = models["re"](a, b) if models else line
    return float(v), np.nan, SRC_RADAR, gap
