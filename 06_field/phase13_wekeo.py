r"""WEkEO (HDA) access for the HR-VPP benchmark: PPI seasonal trajectories + QFLAG, VPP season-1 parameters.

The owner's WEkEO API token (profile page) is read from D:\vista\credentials\wekeo.json (plain text or JSON) and never printed. `search` only lists
files and sizes; downloads need the owner's OK.
"""
import json
import re
import sys
import time
from collections import Counter

import requests

CRED = r"D:\vista\credentials\wekeo.json"
BBOX = [5.963, 50.589, 6.693, 51.101]                                   # W S E N, all four study areas
START, END = "2020-10-01T00:00:00.000Z", "2025-08-31T23:59:59.999Z"
QUERIES = {"PPI": ("EO:EEA:DAT:CLMS_HRVPP_ST", {"productType": "PPI"}),
           "QFLAG": ("EO:EEA:DAT:CLMS_HRVPP_ST", {"productType": "QFLAG"}),
           **{p: ("EO:EEA:DAT:CLMS_HRVPP_VPP", {"productType": p, "productGroupId": "s1"}) for p in ("SOSD", "MAXD", "EOSD", "QFLAG2")}}


HDA = "https://gateway.prod.wekeo2.eu/hda-broker/api/v1/dataaccess"


def read_token(text):
    """The token from the file: plain text (with or without 'Bearer ') or JSON with a token-like key."""
    text = text.strip()
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            text = next(str(v) for k, v in d.items() if "token" in k.lower() or k.lower() in ("key", "api_key"))
        elif isinstance(d, str):
            text = d
    except (ValueError, StopIteration):
        pass
    m = re.search(r"Bearer\s+([A-Za-z0-9._~+/=-]+)", text)        # the whole curl example from the profile page also works
    return (m.group(1) if m else text).strip().strip('"')


_CACHE = {}


def login_token(user, password):
    """A fresh access token from the WEkEO username / password (cached 50 min; never shown)."""
    if _CACHE.get("t", 0) > time.time():
        return _CACHE["token"]
    try:
        r = requests.post("https://gateway.prod.wekeo2.eu/hda-broker/gettoken", json={"username": user, "password": password}, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"login request failed ({type(e).__name__})") from None
    if r.status_code >= 400:
        raise RuntimeError(f"login {r.status_code}")
    _CACHE.update(token=r.json()["access_token"], t=time.time() + 3000)
    return _CACHE["token"]


def session():
    text = open(CRED, encoding="utf-8-sig").read()
    try:
        d = json.loads(text)
    except ValueError:
        d = None
    tok = login_token(d["user"], d["password"]) if isinstance(d, dict) and "user" in d and "password" in d else read_token(text)
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "Authorization": "Bearer " + tok})
    return s


def search(s, q):
    out, start = [], 0
    while True:
        try:
            r = s.post(f"{HDA}/search", json=dict(q, startIndex=start), timeout=180)
        except requests.RequestException as e:                      # never show the request (it carries the token)
            raise RuntimeError(f"search request failed ({type(e).__name__})") from None
        if r.status_code >= 400:
            raise RuntimeError(f"search {r.status_code}: {r.text[:300]}")
        d = r.json(); feats = d.get("features", [])
        out += feats
        total = d.get("properties", {}).get("totalResults", len(out))
        if not feats or len(out) >= total:
            return out
        start += len(feats)


def query(dataset, extra):
    return {"dataset_id": dataset, "bbox": BBOX, "startdate": START, "enddate": END, "itemsPerPage": 200, "startIndex": 0, **extra}


if __name__ == "__main__" and sys.argv[1:2] == ["search"]:
    c = session()
    for name, (ds, extra) in QUERIES.items():
        try:
            res = search(c, query(ds, extra))
        except Exception as e:
            print(f"== {name}: search failed: {str(e)[:300]}"); continue
        sizes = [x.get("properties", {}).get("size") or 0 for x in res]
        ids = [x.get("id", "") for x in res]
        tiles = Counter(t for i in ids for t in i.split("_") if len(t) == 6 and t[0].isdigit() and t[2].isalpha())
        print(f"== {name}: {len(res)} files, {sum(sizes) / 1e9:.2f} GB; tiles {dict(tiles)}")
        for i in ids[:2]: print("   e.g.", i)
