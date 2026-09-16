"""
Tests for src/data/preprocessing.py's XBDDamageDataset.

Uses synthetic fixtures (tiny generated PNGs + hand-written label JSON in xBD's
real schema) rather than needing the actual xBD dataset, so these run fast and
without any external data dependency.
"""

import json
import os

import pytest
from PIL import Image

from src.data.preprocessing import DAMAGE_LABEL_MAP, XBDDamageDataset, filter_flood_events


@pytest.fixture
def xbd_fixture(tmp_path):
    """
    Builds a minimal synthetic xBD-style dataset with three buildings placed at
    a top-left corner, the center, and a bottom-right corner of a 500x500 tile -
    the corner placements are what exposed the edge-crop black-padding bug.
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    base_id = "test-flooding_00000001"
    Image.new("RGB", (500, 500), color=(100, 150, 100)).save(
        images_dir / f"{base_id}_pre_disaster.png"
    )
    Image.new("RGB", (500, 500), color=(80, 60, 60)).save(
        images_dir / f"{base_id}_post_disaster.png"
    )

    label = {
        "features": {
            "xy": [
                {
                    "properties": {"subtype": "major-damage"},
                    "wkt": "POLYGON ((10 10, 30 10, 30 30, 10 30, 10 10))",  # near top-left corner
                },
                {
                    "properties": {"subtype": "no-damage"},
                    "wkt": "POLYGON ((250 250, 270 250, 270 270, 250 270, 250 250))",  # center
                },
                {
                    "properties": {"subtype": "destroyed"},
                    "wkt": "POLYGON ((490 490, 499 490, 499 499, 490 499, 490 490))",  # bottom-right corner
                },
            ]
        }
    }
    with open(labels_dir / f"{base_id}_post_disaster.json", "w") as f:
        json.dump(label, f)

    # A non-flood-event label file, to confirm filter_flood_events actually filters.
    with open(labels_dir / "hurricane-michael_00000002_post_disaster.json", "w") as f:
        json.dump({"features": {"xy": []}}, f)

    return str(images_dir), str(labels_dir)


def test_filter_flood_events_excludes_non_flood_disasters(xbd_fixture):
    _, labels_dir = xbd_fixture
    flood_files = filter_flood_events(labels_dir)
    assert flood_files == ["test-flooding_00000001_post_disaster.json"]


def test_dataset_builds_correct_number_of_samples(xbd_fixture):
    images_dir, labels_dir = xbd_fixture
    ds = XBDDamageDataset(images_dir=images_dir, labels_dir=labels_dir, patch_size=224)
    assert len(ds) == 3


def test_dataset_maps_damage_labels_correctly(xbd_fixture):
    images_dir, labels_dir = xbd_fixture
    ds = XBDDamageDataset(images_dir=images_dir, labels_dir=labels_dir, patch_size=224)
    labels = sorted(ann["damage_label"] for _, ann in ds.samples)
    assert labels == [
        DAMAGE_LABEL_MAP["no-damage"],
        DAMAGE_LABEL_MAP["major-damage"],
        DAMAGE_LABEL_MAP["destroyed"],
    ]


def test_getitem_returns_correctly_shaped_tensors(xbd_fixture):
    images_dir, labels_dir = xbd_fixture
    ds = XBDDamageDataset(images_dir=images_dir, labels_dir=labels_dir, patch_size=224)
    for i in range(len(ds)):
        pre, post, label = ds[i]
        assert pre.shape == (3, 224, 224)
        assert post.shape == (3, 224, 224)
        assert label.dtype.is_floating_point is False  # should be a long/int label


def test_edge_case_buildings_get_full_real_content_not_black_padding(xbd_fixture):
    """
    Regression test: buildings within half a patch-size of a tile edge used to
    get a crop box that ran outside the image, which PIL silently pads with
    black rather than erroring - for the corner cases here (patch_size=224,
    building 10-20px from a 500x500 tile's edge), that meant roughly 60-70%
    of the "building" patch was black filler instead of real imagery.

    _crop_patch() now shifts the box to stay inside the image bounds instead
    of letting it run off the edge, so every patch should be 100% real content.
    """
    images_dir, labels_dir = xbd_fixture
    ds = XBDDamageDataset(images_dir=images_dir, labels_dir=labels_dir, patch_size=224)
    pre_image = Image.open(os.path.join(images_dir, "test-flooding_00000001_pre_disaster.png"))

    for _, ann in ds.samples:
        patch = ds._crop_patch(pre_image, ann["centroid"])
        assert patch.size == (224, 224)
        pixels = list(patch.getdata())
        black_pixel_count = sum(1 for p in pixels if p == (0, 0, 0))
        # allow a tiny tolerance rather than requiring literally zero, in case
        # of incidental black pixels, but the bug produced 60-70%+ black -
        # anything under 1% here confirms the fix, not luck.
        assert black_pixel_count / len(pixels) < 0.01, (
            f"{black_pixel_count}/{len(pixels)} pixels are black - "
            f"crop box likely ran outside the image bounds again"
        )
