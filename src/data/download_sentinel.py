"""
Pull Sentinel-1 SAR imagery for the Nepal case study AOI via Google Earth Engine.

Prereqs:
    pip install earthengine-api
    earthengine authenticate   (run once, follow the browser login flow)

This queries pre-flood and post-flood Sentinel-1 GRD scenes over the
Rasuwa/Nuwakot/Dhading area around the August 26, 2026 flood event, and
exports them as GeoTIFFs to your Google Drive (accessible from Colab).
"""

import ee

# Approximate bounding box covering Rasuwa/Nuwakot/Dhading districts, Nepal.
# Refine this with exact district boundaries (e.g. via GADM shapefiles) for
# a more precise AOI before your final case study run.
AOI_COORDS = [
    [85.05, 27.85],
    [85.55, 27.85],
    [85.55, 28.25],
    [85.05, 28.25],
    [85.05, 27.85],
]

EVENT_DATE = "2026-08-26"
PRE_WINDOW = ("2026-08-01", "2026-08-25")
POST_WINDOW = ("2026-08-27", "2026-09-10")


def initialize_ee(project_id: str = None):
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


def get_sentinel1_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


def export_composite(image: ee.Image, aoi: ee.Geometry, description: str, folder: str = "flood_project"):
    task = ee.batch.Export.image.toDrive(
        image=image.clip(aoi),
        description=description,
        folder=folder,
        fileNamePrefix=description,
        scale=10,          # Sentinel-1 GRD native resolution
        region=aoi,
        maxPixels=1e13,
    )
    task.start()
    print(f"Export task started: {description}. Check Earth Engine Tasks tab or Drive folder '{folder}'.")
    return task


def main(project_id: str = None):
    initialize_ee(project_id)
    aoi = ee.Geometry.Polygon([AOI_COORDS])

    pre_collection = get_sentinel1_collection(aoi, *PRE_WINDOW)
    post_collection = get_sentinel1_collection(aoi, *POST_WINDOW)

    pre_composite = pre_collection.median()
    post_composite = post_collection.median()

    export_composite(pre_composite, aoi, "nepal_flood_pre_20260826")
    export_composite(post_composite, aoi, "nepal_flood_post_20260826")


if __name__ == "__main__":
    # Pass your GEE cloud project ID if required by your account setup.
    main(project_id=None)
