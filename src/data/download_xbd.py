"""
xBD dataset acquisition checklist.

xView2 requires manual registration + download (no public direct-download API,
per their terms of use). This script does NOT auto-download anything - it
verifies your local setup and gives you the exact next steps.

Run this after you've registered at https://xview2.org and downloaded Tier 1.
"""

import os
import sys

XBD_ROOT = "data/raw/xbd"
FLOOD_KEYWORDS = ["flood", "flooding", "hurricane-harvey", "hurricane-matthew", "midwest-flooding"]


def check_setup():
    print("=== xBD Dataset Setup Checklist ===\n")

    if not os.path.exists(XBD_ROOT):
        print(f"[ ] MISSING: {XBD_ROOT} does not exist.")
        print("    1. Register at https://xview2.org")
        print("    2. Download the 'Tier 1' training set")
        print(f"    3. Extract it into {XBD_ROOT}/images/ and {XBD_ROOT}/labels/")
        return False

    images_dir = os.path.join(XBD_ROOT, "images")
    labels_dir = os.path.join(XBD_ROOT, "labels")

    ok = True
    for d in [images_dir, labels_dir]:
        if not os.path.isdir(d):
            print(f"[ ] MISSING: {d}")
            ok = False
        else:
            n_files = len(os.listdir(d))
            print(f"[x] Found {d} ({n_files} files)")

    if ok:
        print("\nNext step: filter labels for flood-related disaster types.")
        print(f"Look for these keywords in label JSON metadata: {FLOOD_KEYWORDS}")
        print("See src/data/preprocessing.py -> filter_flood_events()")

    return ok


if __name__ == "__main__":
    success = check_setup()
    sys.exit(0 if success else 1)
