"""
Client for pulling ground-truth incident/damage numbers from Nepal's BIPAD Portal
(bipadportal.gov.np), the actual Disaster Information Management System behind
NDRRMA. Used for CASE STUDY VALIDATION ONLY - this gives you reported casualty/
damage figures to compare your model's output against. It is NOT a source of
satellite imagery or training data for the ML models (see data/README.md for that).

No documented public API was found for BIPAD as of this writing. To find the
real endpoint:
    1. Open bipadportal.gov.np in Chrome, open DevTools (F12) -> Network tab
    2. Filter by Fetch/XHR, navigate to the Incident or Damage & Loss module
    3. Find the request that returns JSON incident/damage data
    4. Fill in BASE_URL and the endpoint path below

Once you have a real endpoint, this becomes a two-line fix - the plumbing
(request, error handling, saving to reports/) is already here.
"""

import json
import os
from typing import Optional

import requests

BASE_URL = "https://bipadportal.gov.np/api"  # PLACEHOLDER - confirm via DevTools


def fetch_incidents(
    district: Optional[str] = None,
    start_date: str = "2026-08-26",
    end_date: str = "2026-09-10",
) -> dict:
    """
    Fetch incident records for a date range, optionally filtered by district.

    NOTE: endpoint path/params below are guesses based on common REST patterns -
    replace with what you actually find in DevTools.
    """
    params = {"start_date": start_date, "end_date": end_date}
    if district:
        params["district"] = district

    response = requests.get(f"{BASE_URL}/incident/", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_damage_loss(district: Optional[str] = None) -> dict:
    """Fetch damage & loss records (households affected, structures damaged, etc.)."""
    params = {}
    if district:
        params["district"] = district

    response = requests.get(f"{BASE_URL}/loss/", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def save_case_study_data(output_path: str = "reports/bipad_ground_truth.json"):
    """Pulls incident + damage data for the Aug 2026 flood districts and saves it locally."""
    districts = ["Rasuwa", "Nuwakot", "Dhading"]
    combined = {"incidents": {}, "damage_loss": {}}

    for district in districts:
        try:
            combined["incidents"][district] = fetch_incidents(district=district)
            combined["damage_loss"][district] = fetch_damage_loss(district=district)
        except requests.RequestException as e:
            print(f"Failed to fetch data for {district}: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved ground-truth data to {output_path}")


if __name__ == "__main__":
    save_case_study_data()
