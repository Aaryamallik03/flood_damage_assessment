"""
Unit tests for src/data/ndrrma_sitrep.py.

parse_sitrep_text is tested directly against a saved fixture of real Situation
Report #01 text (tests/fixtures/sitrep01_sample_text.txt) so these tests don't
need network access or a real PDF. discover_sitreps/download_sitrep/fetch_latest
DO need network access to ndrrma.gov.np and are intentionally left untested here
-- exercise those manually once that domain is reachable from your environment.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.ndrrma_sitrep import parse_sitrep_text  # noqa: E402

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "sitrep01_sample_text.txt")


def _load_fixture_text() -> str:
    with open(FIXTURE_PATH) as f:
        return f.read()


def test_parses_all_known_fields_from_sitrep_01():
    text = _load_fixture_text()
    figures = parse_sitrep_text(
        text,
        sitrep_number=1,
        source_url=(
            "https://ndrrma.gov.np/mediafiles/rasuwa/"
            "Rasuwa_Flood_SitRep_Temp_ENG_01_01092026.pdf"
        ),
    )

    assert figures.report_date == "2026-09-01"
    assert figures.deaths == 987
    assert figures.missing == 3916
    assert figures.injured == 279
    assert figures.total_rescued == 11814
    assert figures.security_personnel_deployed == 21011
    assert figures.households_isolated == 3702
    assert figures.bridges_washed_away == 41
    assert figures.bridges_damaged == 4
    assert figures.houses_fully_damaged == 1267
    assert figures.houses_partially_damaged == 1253
    assert figures.satellite_building_exposure_estimate == 4689
    assert figures.holding_centres_total == 30
    assert figures.holding_centres_occupancy == 3930
    assert figures.schools_fully_damaged == 8
    assert figures.schools_partially_damaged == 8
    assert figures.hydropower_capacity_affected_mw == 783.385
    assert figures.districts_affected == 5


def test_parses_correctly_against_real_scrambled_pdfplumber_layout():
    """
    NDRRMA's PDF has a two-column "Human Casualties" infographic (bar chart +
    stat boxes). pdfplumber's real reading order interleaves those columns
    line-by-line, which puts the "Deaths" label next to the *missing* count
    and separates "Injured"/"Discharged" from their numbers by several lines
    of unrelated caption text. This fixture is the actual text pdfplumber
    produced on a real download (captured from a user's local run) - it's
    what caught the original deaths=3916 (should be 987) and injured=None
    (should be 279) bugs.
    """
    fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "sitrep01_real_pdfplumber_order.txt"
    )
    with open(fixture_path) as f:
        text = f.read()

    figures = parse_sitrep_text(
        text, sitrep_number=1, source_url="https://ndrrma.gov.np/mediafiles/rasuwa/test.pdf"
    )

    assert figures.deaths == 987, f"expected 987, got {figures.deaths}"
    assert figures.missing == 3916, f"expected 3916, got {figures.missing}"
    assert figures.injured == 279, f"expected 279, got {figures.injured}"
    assert figures.discharged == 168, f"expected 168, got {figures.discharged}"
    assert figures.total_rescued == 11814
    assert figures.security_personnel_deployed == 21011


def test_missing_fields_return_none_not_an_exception():
    # A near-empty report should parse without raising, with every figure None.
    figures = parse_sitrep_text(
        "Situation Report #02\nDate: 02 September 2026\nNo figures here yet.",
        sitrep_number=2,
        source_url="https://example.invalid/sitrep02.pdf",
    )
    assert figures.report_date == "2026-09-02"
    assert figures.deaths is None
    assert figures.missing is None
    assert figures.satellite_building_exposure_estimate is None


def test_commas_in_numbers_are_parsed_correctly():
    text = "Injured\n12,345\nTotal Rescued\n1,234,567"
    figures = parse_sitrep_text(text, sitrep_number=3, source_url="https://example.invalid")
    assert figures.injured == 12345
    assert figures.total_rescued == 1234567


if __name__ == "__main__":
    test_parses_all_known_fields_from_sitrep_01()
    test_parses_correctly_against_real_scrambled_pdfplumber_layout()
    test_missing_fields_return_none_not_an_exception()
    test_commas_in_numbers_are_parsed_correctly()
    print("All tests passed.")
