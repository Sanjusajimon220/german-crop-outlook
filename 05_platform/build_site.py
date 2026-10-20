"""Builds the combined Crop Outlook (district + field level) into the Groundtruth-earth site folder (plan 2026-10-17, Task 6).

build_site(src, site) replaces <site>/crop-outlook/ with a copy of <src> (the app, district data in data/, field data in
field/), after a dated backup of the old folder, and makes sure <site>/_headers caches the field files.
The owner deploys <site> to Netlify. Run `python build_site.py [src] [site]`.
"""
import os
import shutil
import sys
import time

HEADERS = {   # path -> rule; field files carry ?v=<data version>, sw.js is always re-checked (installable app updates)
    "/crop-outlook/field/*": "\n/crop-outlook/field/*\n  Cache-Control: public, max-age=604800\n",
    "/crop-outlook/sw.js": "\n/crop-outlook/sw.js\n  Cache-Control: no-cache\n",
    "/crop-outlook/manifest.webmanifest": "\n/crop-outlook/manifest.webmanifest\n  Content-Type: application/manifest+json\n",
}


def build_site(src="D:/german-crop-outlook-field/docs", site="D:/groundtruth-earth-site", backup_root="D:/groundtruth-earth-site-backup"):
    for need in ("index.html", os.path.join("data", "meta.json"), os.path.join("field", "index.json")):
        if not os.path.exists(os.path.join(src, need)):                 # check BEFORE touching the live folder
            raise FileNotFoundError(f"source incomplete, live site left unchanged: {need} missing in {src}")
    dest = os.path.join(site, "crop-outlook")
    backup = None
    if os.path.exists(dest):
        backup = os.path.join(f"{backup_root}-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copytree(dest, os.path.join(backup, "crop-outlook"))
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("package.json", "*.part", "*.part.npz"))
    hp = os.path.join(site, "_headers")
    old = open(hp, encoding="utf-8").read() if os.path.exists(hp) else ""
    missing = [rule for path, rule in HEADERS.items() if path not in old]
    if missing:
        with open(hp, "a", encoding="utf-8") as f:
            f.write("".join(missing))
    n, size = 0, 0
    for root, _, files in os.walk(dest):
        for f in files:
            n += 1
            size += os.path.getsize(os.path.join(root, f))
    return {"files": n, "mb": round(size / 1e6, 1), "backup": backup}


if __name__ == "__main__":
    print(build_site(*sys.argv[1:3]))
