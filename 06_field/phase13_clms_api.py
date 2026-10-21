"""Copernicus Land Monitoring Service (CLMS) download API for the HR-VPP / LAI 300 m benchmark (project log 2026-10-17/19).

The owner's service key is read from D:\\vista\\credentials\\clms_service_key.json; the key and the access token are never
printed, logged or written anywhere else. Run `python phase13_clms_api.py search` (read-only: lists the products, how they are
offered and their limits) before any download request.
"""
import calendar
import json
import sys
import time

import jwt
import requests

API = "https://land.copernicus.eu/api"
KEY = r"D:\vista\credentials\clms_service_key.json"
BOX = {"W": 5.963, "S": 50.589, "E": 6.693, "N": 51.101}                       # all four study areas
QUERIES = {"ppi": "Plant Phenology Index Seasonal Trajectories", "ppi_qflag": "Seasonal Trajectories Quality Flag",
           "vpp": "Vegetation Phenology and Productivity Parameters", "lai300": "Leaf Area Index 300m"}


def token():
    k = json.load(open(KEY, encoding="utf-8"))
    now = int(time.time())
    grant = jwt.encode({"iss": k["client_id"], "sub": k["user_id"], "aud": k["token_uri"], "iat": now, "exp": now + 3600},
                       k["private_key"].encode("utf-8"), algorithm="RS256")
    r = requests.post(k["token_uri"], headers={"Accept": "application/json"},
                      data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": grant}, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def session():
    s = requests.Session()
    s.headers.update({"Accept": "application/json", "Authorization": f"Bearer {token()}"})
    return s


def search(s, text):
    r = s.get(f"{API}/@search", params={"portal_type": "DataSet", "SearchableText": text, "b_size": 25,
                                        "metadata_fields": ["UID", "dataset_full_format", "dataset_download_information",
                                                            "download_limit_temporal_extent", "title"]}, timeout=120)
    r.raise_for_status()
    return r.json().get("items", [])


if __name__ == "__main__" and sys.argv[1:2] == ["search"]:
    s = session()
    print("signed in (token not shown)")
    for k, q in QUERIES.items():
        print(f"\n== {k}: '{q}'")
        for it in search(s, q)[:8]:
            info = (it.get("dataset_download_information") or {}).get("items", [])
            print(f"- {it.get('title')} | UID {it.get('UID')} | format {it.get('dataset_full_format')} | "
                  f"limit {it.get('download_limit_temporal_extent')}")
            for d in info:
                print(f"    download option: {d.get('@id')} | {d.get('name') or d.get('collection')} | {d.get('full_format') or ''} | {d.get('full_source') or ''}")


LAI300 = {"DatasetID": "a8126fddc2c749cfbdfffaa5cf96e1c2", "DatasetDownloadInformationID": "460c3597-85bc-430a-9a27-309189cc829f"}


def ms(day, end=False):
    """Milliseconds since epoch at 00:00 UTC of day (end=True: 23:59:59.999 UTC)."""
    return calendar.timegm(time.strptime(day, "%Y-%m-%d")) * 1000 + (86399999 if end else 0)


def request_lai300(s, start, end):
    """One CLMS data request (<= 31 days) for LAI 300 m v2 clipped to BOX; returns the TaskIds."""
    body = {"Datasets": [dict(LAI300, OutputFormat="Geotiff", OutputGCS="EPSG:4326",
                              BoundingBox=[BOX["N"], BOX["E"], BOX["S"], BOX["W"]],
                              TemporalFilter={"StartDate": ms(start), "EndDate": ms(end, end=True)})]}
    r = s.post(f"{API}/@datarequest_post", json=body, headers={"Content-Type": "application/json"}, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"datarequest_post {r.status_code}: {r.text[:500]}")
    return r.json()
