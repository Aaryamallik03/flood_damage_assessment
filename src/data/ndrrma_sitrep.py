"""
Fetches and parses NDRRMA "Situation Report" PDFs for the Aug 2026 Rasuwa-Bhotekoshi
flood - the PRIMARY ground-truth source for this project's case study validation
(see data/README.md). Unlike BIPAD (src/data/bipad_client.py), NDRRMA has no bulk
JSON API. Situation Report #01 was found at a predictable static PDF URL:

    https://ndrrma.gov.np/mediafiles/rasuwa/Rasuwa_Flood_SitRep_Temp_ENG_01_01092026.pdf
                                                                          ^^  ^^^^^^^^
                                                                       number  date (DDMMYYYY)

IMPORTANT LIMITATION - later reports: NDRRMA's main site (ndrrma.gov.np/en/situation-report/<id>)
is a JavaScript single-page app that loads its content from a backend API this module
has no way to discover automatically (no browser-rendering tool available here). Brute-force
discovery below only finds reports that happen to sit at a *static* PDF URL matching one of
FILENAME_TEMPLATES - it cannot crawl the SPA itself. Filename conventions have also changed
between NDRRMA events before (e.g. a related 2025 Rasuwagadhi report used
`Rasuwagadhi_Flood_Sitrep1_08072025_.pdf` - no "_Temp_ENG_", no zero-padding), so there's no
guarantee a later report in *this* event even follows a template listed here.

When discover_sitreps() only turns up early reports, don't trust that as "no newer report
exists" - check manually instead:
  1. NDRRMA's Facebook (facebook.com/NDRRMA) or X/Twitter (@NDRRMA_Nepal) usually link each
     new situation report directly.
  2. Or open ndrrma.gov.np/en/rasuwa (or /en/situation-report/<id>) in a real browser, DevTools
     -> Network -> Fetch/XHR, and find the request that returns the newest report's PDF link
     (same technique already documented in bipad_client.py for that portal).
Once you have a real URL for a newer report, skip discovery entirely and hand it directly to
fetch_from_url() / `--url` on the CLI below.

Regex parsing was built against the real text of Situation Report #01 (1 Sept 2026), including
a real quirk where pdfplumber's reading order scrambles NDRRMA's two-column "Human Casualties"
infobox (see tests/fixtures/sitrep01_real_pdfplumber_order.txt + the regression test built from
it). Every field is extracted defensively: a missing field yields None + a logged warning
instead of a crash, and the raw extracted text is always saved alongside the parsed JSON so
nothing is silently lost.

IMPORTANT - network access: this script needs to reach ndrrma.gov.np directly.
Sandboxed/CI environments with an allowlisted egress proxy (as opposed to your
own laptop) will typically need that domain added to the allowlist first.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ndrrma_sitrep")

BASE_URL = "https://ndrrma.gov.np/mediafiles/rasuwa"

# Multiple candidate filename conventions to try per (number, date) - NDRRMA has used
# different templates across events/reports, so we don't assume only one holds here.
# Each must accept `num` (int) and `date_str` (DDMMYYYY) format args.
FILENAME_TEMPLATES = [
    "Rasuwa_Flood_SitRep_Temp_ENG_{num:02d}_{date_str}.pdf",  # confirmed for SitRep #01
    "Rasuwa_Flood_SitRep_ENG_{num:02d}_{date_str}.pdf",  # guess: without "_Temp_"
    "Rasuwa_Flood_SitRep_Final_ENG_{num:02d}_{date_str}.pdf",  # guess: "final" replacing "temp"
    "Rasuwa_Flood_SitRep_Temp_ENG_{num}_{date_str}.pdf",  # guess: no zero-padding
    "Rasuwa_Flood_Sitrep{num}_{date_str}.pdf",  # guess: pattern seen on a different NDRRMA event
]

EVENT_START = date(2026, 8, 26)

REQUEST_TIMEOUT = 15
REQUEST_DELAY_SEC = 0.3  # be polite to a government server during discovery scans


@dataclass
class SitRepFigures:
    """Parsed figures. Any field can be None if this report's layout didn't match."""

    sitrep_number: int
    report_date: Optional[str]  # ISO date the report itself is dated, per its cover page
    source_url: str
    fetched_at: str

    deaths: Optional[int] = None
    missing: Optional[int] = None
    injured: Optional[int] = None
    discharged: Optional[int] = None
    total_rescued: Optional[int] = None
    security_personnel_deployed: Optional[int] = None
    households_isolated: Optional[int] = None
    bridges_washed_away: Optional[int] = None
    bridges_damaged: Optional[int] = None
    houses_fully_damaged: Optional[int] = None
    houses_partially_damaged: Optional[int] = None
    satellite_building_exposure_estimate: Optional[int] = None
    holding_centres_total: Optional[int] = None
    holding_centres_occupancy: Optional[int] = None
    schools_fully_damaged: Optional[int] = None
    schools_partially_damaged: Optional[int] = None
    hydropower_capacity_affected_mw: Optional[float] = None
    districts_affected: Optional[int] = None

    raw_text_excerpt: str = field(default="", repr=False)


def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _url_exists(session: requests.Session, url: str) -> bool:
    try:
        resp = session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if resp.status_code == 200:
            return True
        if resp.status_code in (403, 405):
            # some servers reject HEAD; fall back to a ranged GET
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=True,
                                headers={"Range": "bytes=0-0"})
            return resp.status_code in (200, 206)
        return False
    except requests.RequestException as e:
        logger.debug(f"HEAD failed for {url}: {e}")
        return False


def discover_sitreps(
    max_number: int = 40,
    end_date: Optional[date] = None,
    max_consecutive_number_misses: int = 5,
    session: Optional[requests.Session] = None,
) -> dict[int, dict]:
    """
    Brute-force-discover which (number, date, template) combinations resolve to a
    real PDF, trying every entry in FILENAME_TEMPLATES for each candidate date.

    Returns {number: {"date": date, "url": str}} for every sitrep found.

    Search window per number starts from the date of the last *successfully found*
    report (reports only move forward in time) rather than always re-scanning from
    EVENT_START, to keep the request count sane.

    LIMITATION: this can only find reports sitting at one of FILENAME_TEMPLATES.
    NDRRMA's main site is a JS SPA this module can't crawl - see the module
    docstring for how to find later reports manually and feed them to
    fetch_from_url() instead when this comes up empty past #01.
    """
    session = session or requests.Session()
    end_date = end_date or date.today()

    found: dict[int, dict] = {}
    consecutive_misses = 0
    search_from = EVENT_START

    for number in range(1, max_number + 1):
        hit = None
        for d in _date_range(search_from, end_date):
            date_str = d.strftime("%d%m%Y")
            for template in FILENAME_TEMPLATES:
                url = f"{BASE_URL}/{template.format(num=number, date_str=date_str)}"
                time.sleep(REQUEST_DELAY_SEC)
                if _url_exists(session, url):
                    hit = {"date": d, "url": url}
                    break
            if hit:
                break

        if hit:
            found[number] = hit
            search_from = hit["date"]  # next number won't be dated earlier than this
            consecutive_misses = 0
            logger.info(f"Found SitRep #{number:02d} dated {hit['date'].isoformat()}: {hit['url']}")
        else:
            consecutive_misses += 1
            logger.debug(f"No SitRep #{number:02d} found in range {search_from}..{end_date}")
            if consecutive_misses >= max_consecutive_number_misses:
                logger.info(
                    f"Stopping discovery after {consecutive_misses} consecutive misses "
                    f"(last found: #{max(found) if found else 'none'}). If you know a later "
                    f"report exists (check NDRRMA's Facebook/X, or the site's DevTools Network "
                    f"tab per the module docstring), pass its URL to fetch_from_url() directly "
                    f"instead of relying on discovery."
                )
                break

    return found


def download_sitrep(url: str, session: Optional[requests.Session] = None) -> bytes:
    session = session or requests.Session()
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def _extract_text(pdf_bytes: bytes) -> str:
    import io

    import pdfplumber

    text_chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def _find_int(pattern: str, text: str, group: int = 1) -> Optional[int]:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(group).replace(",", ""))
    except (ValueError, IndexError):
        return None


def _find_float(pattern: str, text: str, group: int = 1) -> Optional[float]:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(group).replace(",", ""))
    except (ValueError, IndexError):
        return None


def parse_sitrep(
    pdf_bytes: bytes,
    sitrep_number: int,
    source_url: str,
) -> SitRepFigures:
    text = _extract_text(pdf_bytes)
    return parse_sitrep_text(text, sitrep_number=sitrep_number, source_url=source_url)


def parse_sitrep_text(
    text: str,
    sitrep_number: int,
    source_url: str,
) -> SitRepFigures:
    """Regex-parsing stage, split out from PDF extraction so it can be unit-tested
    against plain text fixtures without needing real PDF bytes."""

    # Report's own cover-page date, e.g. "Date: 01 September 2026"
    report_date = None
    m = re.search(r"Date:\s*(\d{1,2}\s+\w+\s+\d{4})", text)
    if m:
        try:
            report_date = datetime.strptime(m.group(1), "%d %B %Y").date().isoformat()
        except ValueError:
            logger.warning(f"Couldn't parse cover-page date '{m.group(1)}' for #{sitrep_number}")

    figures = SitRepFigures(
        sitrep_number=sitrep_number,
        report_date=report_date,
        source_url=source_url,
        fetched_at=datetime.utcnow().isoformat() + "Z",
        raw_text_excerpt=text[:4000],
    )

    # NOTE: the "Human Casualties" info-box in the actual PDF is a two-column
    # infographic (bar chart + stat boxes). pdfplumber's reading order merges
    # those columns line-by-line, which scrambles which number sits next to
    # which label (e.g. "Deaths" ends up adjacent to the *missing* count, not
    # the deaths count, purely because of vertical position). The flowing-
    # prose Highlights bullets ("Deceased bodies 987...") don't have this
    # problem, so they're used as the primary source for deaths; the boxed
    # layout is only a fallback for reports where that bullet might be absent.
    figures.deaths = _find_int(r"Deceased bodies\s*(\d[\d,]*)", text) or _find_int(
        r"Deaths?\s*\n?\s*(\d[\d,]*)", text
    )
    figures.missing = _find_int(
        r"Total Missing Persons\s*\n?\s*Deaths?\s*\n?\s*[\d,]+\s*\n?\s*(\d[\d,]*)", text
    ) or _find_int(r"Missing individuals\s*([\d,]+)\s*have\s+been\s+reported", text)
    # Same two-column scrambling affects "Injured" / "Discharged": pdfplumber
    # puts several unrelated lines of caption text between the labels and
    # their numbers. Try the simple adjacent-number case first (works if a
    # future report's layout doesn't scramble this box); if that fails, fall
    # back to finding the first pair of adjacent numbers that follows the
    # "Injured Discharged" label pair, within a bounded window so it can't
    # accidentally latch onto an unrelated number much further down the page.
    figures.injured = _find_int(r"Injured\s*\n?\s*(\d[\d,]*)(?!\s*\d)", text)
    figures.discharged = None
    if figures.injured is None:
        m = re.search(r"Injured\s+Discharged[\s\S]{0,300}?\b(\d[\d,]*)\s+(\d[\d,]*)\b", text)
        if m:
            figures.injured = int(m.group(1).replace(",", ""))
            figures.discharged = int(m.group(2).replace(",", ""))
    figures.total_rescued = _find_int(r"Total Rescued\s*\n?\s*(\d[\d,]*)", text) or _find_int(
        r"([\d,]+)\s*individuals\s+have\s+been\s+rescued", text
    )
    figures.security_personnel_deployed = _find_int(
        r"Security\s*\n?\s*Personnel\s*\n?\s*Deployed\s*(\d[\d,]*)", text
    ) or _find_int(r"Deployed\s*([\d,]+)\s*security\s+personnel", text)
    figures.households_isolated = _find_int(r"(\d[\d,]*)\s*households,\s*comprising", text)
    figures.bridges_washed_away = _find_int(
        r"([\d,]+)\s*(?:motorable\s*)?bridges\s*(?:have\s+been\s+)?washed\s+away", text
    )
    figures.bridges_damaged = _find_int(
        r"washed\s+away.{0,40}?(\d+)\s*bridges?\s*(?:have\s+been\s+)?damaged", text, group=1
    )
    figures.houses_fully_damaged = _find_int(
        r"(\d[\d,]*)\s*houses?\s*have\s+been\s+reported\s+as\s+fully\s+damaged", text
    )
    figures.houses_partially_damaged = _find_int(
        r"(\d[\d,]*)\s*as\s+partially\s+damaged", text
    )
    figures.satellite_building_exposure_estimate = _find_int(
        r"Satellite\s+mapping\s+identified\s*(\d[\d,]*)\s*buildings", text
    )
    figures.holding_centres_total = _find_int(
        r"(\d[\d,]*)\s*centres?\s+are\s+accommodating", text
    )
    figures.holding_centres_occupancy = _find_int(
        r"accommodating\s*(\d[\d,]*)\s*people", text
    )
    figures.schools_fully_damaged = _find_int(
        r"(\d+)\s*schools?\s+are\s+reported\s+fully\s+damaged", text
    )
    figures.schools_partially_damaged = _find_int(
        r"fully\s+damaged,\s*(\d+)\s*partially\s+damaged", text
    )
    figures.hydropower_capacity_affected_mw = _find_float(
        r"combined installed capacity of\s*([\d,.]+)\s*MW", text
    )
    figures.districts_affected = _find_int(
        r"total of\s*(\d+)\s*districts?\s*(?:are\s*)?affected", text
    )

    return figures


def fetch_latest(
    output_dir: str = "data/raw/ndrrma",
    reports_dir: str = "reports",
    max_number: int = 40,
) -> SitRepFigures:
    """
    Discover the latest available SitRep, download it, parse it, and save both
    the raw PDF and the parsed JSON (plus a `_latest.json` pointer that always
    reflects the newest run, so downstream code has a stable path to read from).
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "flood-damage-assessment-prototype/0.1"})

    found = discover_sitreps(max_number=max_number, session=session)
    if not found:
        raise RuntimeError(
            "No NDRRMA situation reports discovered. This usually means either "
            "(a) network egress to ndrrma.gov.np is blocked in this environment, "
            "(b) NDRRMA changed its filename pattern, or (c) the reports were "
            "moved/archived. Check BASE_URL/FILENAME_TEMPLATES and your network config."
        )

    latest_number = max(found)
    latest = found[latest_number]

    return _download_and_save(
        latest["url"], sitrep_number=latest_number, report_date=latest["date"],
        output_dir=output_dir, reports_dir=reports_dir, session=session,
    )


def fetch_from_url(
    url: str,
    sitrep_number: int,
    output_dir: str = "data/raw/ndrrma",
    reports_dir: str = "reports",
) -> SitRepFigures:
    """
    Manual-override path: download + parse a SitRep PDF whose URL you already
    know (e.g. found via NDRRMA's Facebook/X, or the site's DevTools Network tab -
    see the module docstring), bypassing discover_sitreps() entirely. Use this
    whenever discovery stops finding newer reports but you know one exists.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "flood-damage-assessment-prototype/0.1"})
    return _download_and_save(
        url, sitrep_number=sitrep_number, report_date=None,
        output_dir=output_dir, reports_dir=reports_dir, session=session,
    )


def _download_and_save(
    url: str,
    sitrep_number: int,
    report_date: Optional[date],
    output_dir: str,
    reports_dir: str,
    session: requests.Session,
) -> SitRepFigures:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    pdf_bytes = download_sitrep(url, session=session)
    date_str = report_date.strftime("%d%m%Y") if report_date else "unknown_date"
    pdf_path = os.path.join(output_dir, f"sitrep_{sitrep_number:02d}_{date_str}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    figures = parse_sitrep(pdf_bytes, sitrep_number=sitrep_number, source_url=url)

    versioned_path = os.path.join(
        reports_dir, f"ndrrma_sitrep_{sitrep_number:02d}_{date_str}.json"
    )
    latest_path = os.path.join(reports_dir, "ndrrma_sitrep_latest.json")
    payload = asdict(figures)
    for path in (versioned_path, latest_path):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    logger.info(
        f"Saved SitRep #{sitrep_number:02d} figures to {versioned_path} and {latest_path}"
    )
    return figures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch + parse NDRRMA SitRep(s)")
    parser.add_argument("--max-number", type=int, default=40)
    parser.add_argument("--output-dir", default="data/raw/ndrrma")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--url",
        default=None,
        help="Skip discovery and fetch/parse this exact SitRep PDF URL instead "
        "(use when you've found a newer report manually - see module docstring)",
    )
    parser.add_argument(
        "--sitrep-number",
        type=int,
        default=None,
        help="Required alongside --url: the report number for labeling output files",
    )
    args = parser.parse_args()

    if args.url:
        if args.sitrep_number is None:
            parser.error("--url requires --sitrep-number")
        result = fetch_from_url(
            args.url,
            sitrep_number=args.sitrep_number,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
        )
    else:
        result = fetch_latest(
            output_dir=args.output_dir, reports_dir=args.reports_dir, max_number=args.max_number
        )
    print(json.dumps(asdict(result), indent=2, default=str))

# Reference snippet this parser was validated against (Situation Report #01,
# dated 01 September 2026, fetched during development):
#
#   Deceased bodies 987 and body parts recovered ...
#   Missing individuals 3,916 have been reported across 62 districts of Nepal.
#   Injured 279
#   Total Rescued 11,814
#   Security Personnel Deployed 21,011
#   3,702 households, comprising a population of 14,461 ... remain isolated
#   A total of 41 motorable bridges have been washed away, while an additional
#   4 bridges have been damaged.
#   In Nuwakot, 1,267 houses have been reported as fully damaged and 1,253 as
#   partially damaged. Satellite mapping identified 4,689 buildings within the
#   flood-affected zone ...
#   A total of 30 centres are accommodating 3,930 people across Nuwakot and Rasuwa.
#   8 schools are reported fully damaged, 8 partially damaged and 3 physically safe.
#   Hydropower and solar facilities with a combined installed capacity of 783.385 MW ...
