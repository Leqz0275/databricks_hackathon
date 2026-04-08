#!/usr/bin/env python3
"""
ArthaSetu v2 — Public Data Downloader
======================================
Downloads calibration data from public Indian government sources.
Used to make synthetic data realistic, NOT as direct model input.

Sources:
  - data.gov.in (PMJDY, CPI, Financial Inclusion)
  - RBI/NABARD public reports
  - AI Kosh (IndiaAI)
"""

import os
import requests
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "public")


def download_file(url, filename, description):
    """Download a file with progress indication."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        print(f"  [SKIP] {filename} already exists")
        return filepath

    print(f"  [GET]  {description}")
    print(f"         URL: {url}")
    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "ArthaSetu-Research/1.0"
        })
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        print(f"         → Saved to {filename} ({len(resp.content) / 1024:.1f} KB)")
        return filepath
    except Exception as e:
        print(f"         → FAILED: {e}")
        print(f"         → Manual download: {url}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ArthaSetu v2 — Public Data Downloader")
    print("=" * 60)
    print()
    print("NOTE: These datasets are used for CALIBRATION only.")
    print("They make synthetic data distributions realistic.")
    print()

    # ── PMJDY Account Statistics ──────────────────────────────────────
    print("[1/5] PMJDY (Pradhan Mantri Jan Dhan Yojana) Account Data")
    print("       Used for: Bank account vintage distributions")
    print("       Source: data.gov.in")
    print("       → Search 'PMJDY' on https://data.gov.in")
    print("       → Download state-wise account statistics CSV")
    print()

    # ── Financial Inclusion Survey ────────────────────────────────────
    print("[2/5] Financial Inclusion Survey Data")
    print("       Used for: Segment sizes, income ranges, banking penetration")
    print("       Source: RBI / data.gov.in")
    print("       → Search 'financial inclusion' on https://data.gov.in")
    print()

    # ── Consumer Price Index by State ─────────────────────────────────
    print("[3/5] Consumer Price Index (CPI) by State")
    print("       Used for: Calibrating bill amounts and income by region")
    print("       Source: data.gov.in / MOSPI")
    print("       → Search 'CPI state wise' on https://data.gov.in")
    print()

    # ── NABARD Rural Credit Survey ────────────────────────────────────
    print("[4/5] NABARD All India Rural Credit Survey")
    print("       Used for: Rural credit behavior, SHG data, agri income")
    print("       Source: nabard.org (public PDFs)")
    print("       → https://www.nabard.org/auth/writereaddata/")
    print()

    # ── AI Kosh Datasets ─────────────────────────────────────────────
    print("[5/5] AI Kosh Financial Datasets")
    print("       Used for: Pre-curated financial/economic datasets")
    print("       Source: https://aikosh.indiaai.gov.in")
    print("       → Browse Finance/Economy category")
    print()

    print("=" * 60)
    print("Most public datasets require manual download from data.gov.in")
    print("due to API key requirements. Download CSVs to data/public/")
    print("=" * 60)

    # Create a reference file for manual download
    reference = {
        "pmjdy": {
            "description": "PMJDY state-wise account statistics",
            "url": "https://data.gov.in/search?title=pmjdy",
            "use": "Bank account vintage distribution calibration",
        },
        "financial_inclusion": {
            "description": "RBI Financial Inclusion Index",
            "url": "https://data.gov.in/search?title=financial+inclusion",
            "use": "Segment size and banking penetration calibration",
        },
        "cpi_state": {
            "description": "Consumer Price Index by State",
            "url": "https://data.gov.in/search?title=consumer+price+index",
            "use": "Regional bill amount and income calibration",
        },
        "electricity_consumption": {
            "description": "State-wise electricity consumption",
            "url": "https://data.gov.in/search?title=electricity+consumption",
            "use": "Electricity bill amount calibration by state",
        },
    }

    ref_path = os.path.join(OUTPUT_DIR, "download_reference.json")
    with open(ref_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nReference file saved to: {ref_path}")


if __name__ == "__main__":
    main()
