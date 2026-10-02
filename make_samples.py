import argparse, json, random, re
from pathlib import Path
from PIL import Image

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]

def centroid(wkt):
    pts = [tuple(map(float, p.split())) for p in re.findall(r"(-?[\d.]+ -?[\d.]+)", wkt)]
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        c = x0 * y1 - x1 * y0
        a += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    if abs(a) < 1e-9:
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return cx / (3 * a), cy / (3 * a)

ap = argparse.ArgumentParser()
ap.add_argument("--xbd-root", required=True, help="folder containing the xBD labels and images")
ap.add_argument("--event", default="", help="only use files starting with this, e.g. midwest-flooding")
ap.add_argument("--patch", type=int, default=224, help="window size in pixels; must match your preprocessing")
ap.add_argument("--per-class", type=int, default=3)
ap.add_argument("--out", default="samples")
a = ap.parse_args()

random.seed(0)
root = Path(a.xbd_root)
found = {c: [] for c in CLASSES}
for lab in root.rglob("*_post_disaster.json"):
    if not lab.name.startswith(a.event):
        continue
    post_img = next(root.rglob(lab.stem + ".png"), None)
    pre_img = next(root.rglob(lab.stem.replace("post", "pre") + ".png"), None)
    if not (post_img and pre_img):
        continue
    for f in json.load(open(lab))["features"]["xy"]:
        p = f["properties"]
        if p.get("feature_type") == "building" and p.get("subtype") in found:
            found[p["subtype"]].append((pre_img, post_img, f["wkt"], p["uid"]))

h = a.patch // 2
for cls, items in found.items():
    random.shuffle(items)
    kept = 0
    for pre_img, post_img, wkt, uid in items:
        if kept == a.per_class:
            break
        cx, cy = centroid(wkt)
        pre, post = Image.open(pre_img).convert("RGB"), Image.open(post_img).convert("RGB")
        if cx < h or cy < h or cx + h > pre.width or cy + h > pre.height:
            continue
        box = (int(cx - h), int(cy - h), int(cx + h), int(cy + h))
        d = Path(a.out) / cls
        d.mkdir(parents=True, exist_ok=True)
        pre.crop(box).resize((224, 224)).save(d / f"{uid[:8]}_before.png")
        post.crop(box).resize((224, 224)).save(d / f"{uid[:8]}_after.png")
        kept += 1
    print(cls, "pairs saved:", kept)