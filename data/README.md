# Dataset Acquisition

## 1. xBD Dataset (for damage classification training)

xBD is the dataset behind the xView2 challenge: pre/post-disaster satellite image pairs
with per-building damage labels (no-damage / minor / major / destroyed) across 19 disaster
events worldwide (not flood-specific to Nepal, but flood events ARE included, e.g. Midwest
US floods, Hurricane Harvey/Matthew flooding).

**Steps:**
1. Register at https://xview2.org (free, academic use)
2. Download the "Tier 1" training set (smaller, ~2.5GB — good starting point for Colab)
3. Place under `data/raw/xbd/` with structure:
   ```
   data/raw/xbd/
   ├── images/       # pre_disaster and post_disaster .png tiles
   └── labels/       # .json files with building polygons + damage labels
   ```
4. Filter to flood-related disaster types first (search the label metadata for
   `"disaster_type": "flooding"` or similar) so your model specializes on flood damage
   patterns rather than generic disaster damage (earthquake rubble looks different from
   flood damage).

Run `src/data/download_xbd.py` for a checklist script (manual download + registration
is required by xView2's terms, this isn't a direct API pull).

## 2. Sentinel-1 SAR Imagery (for flood extent segmentation + your Nepal case study)

**Steps:**
1. Create a free account at https://dataspace.copernicus.eu (replaces the old SciHub)
2. OR use Google Earth Engine (recommended — no need to download raw files, process in the cloud):
   - Sign up for GEE access: https://earthengine.google.com/signup
   - Use `src/data/download_sentinel.py`, which queries GEE for Sentinel-1 GRD scenes
     over the Rasuwa/Nuwakot/Dhading AOI, for dates before and after 2026-08-26.
3. Export pre-flood and post-flood VV+VH band composites as GeoTIFFs.

## 3. Building Footprints (optional, improves per-building damage aggregation)

- Microsoft Global ML Building Footprints: https://github.com/microsoft/GlobalMLBuildingFootprints
- OR OpenStreetMap via `osmnx` Python package for the Nepal AOI

## 4. Ground truth for your case study validation

- **NDRRMA Situation Reports (PRIMARY SOURCE - use this first)** — NDRRMA publishes formal
  PDF situation reports at `ndrrma.gov.np/mediafiles/rasuwa/` (filenames follow a pattern like
  `Rasuwa_Flood_SitRep_Temp_ENG_<number>_<date>.pdf`). These contain detailed, dated official
  figures: deaths/missing/injured by district, infrastructure damage, and even NDRRMA's own
  satellite-based building exposure estimate for this event (4,689 buildings) — extremely
  useful as a direct comparison point for your model's output. Check for later-numbered
  situation reports (#02, #03, etc.) for updated figures, since numbers were revised for weeks.
  Run `src/data/ndrrma_sitrep.py` to automate this: it brute-force-discovers whichever
  (number, date) PDF is newest, downloads it, and regex-parses ~17 figures (deaths, missing,
  injured, rescued, bridges, houses damaged, satellite building exposure, schools, hydropower
  capacity affected, etc.) into `reports/ndrrma_sitrep_latest.json`. Numbers were still climbing
  as of early September, so re-run it periodically rather than trusting one cached copy. See the
  module docstring for the network-access caveat if you're running this from a sandboxed
  environment with restricted egress.
- **BIPAD Portal (bipadportal.gov.np)** — the Incident and Damage & Loss modules did NOT
  reflect this event's true scale at the per-district level when checked (likely because a
  disaster this large was tracked via centralized Situation Reports instead) — treat BIPAD's
  UI as secondary/supplementary, not authoritative, for this specific event. No documented
  public bulk API was found either — see `src/data/bipad_client.py` for a stub if you want to
  try finding one via browser DevTools for other/smaller events.
- This is **validation data, not training data** — it gives you numbers to compare against,
  not imagery or per-building labels to train on.
- Other sources: international news (AFP/Reuters/AP), Wikipedia's event page (cross-check
  final figures, since they were updated over several weeks).
