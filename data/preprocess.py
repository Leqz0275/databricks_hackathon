#!/usr/bin/env python3
"""
ArthaSetu v2 — Data Preprocessor
==================================
Validates, cleans, and prepares generated synthetic data for Databricks upload.
Runs quality checks and produces upload-ready CSVs.
"""

import os
import pandas as pd
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def validate_user_profiles(df):
    """Validate user profiles dataset."""
    print("  Validating user_profiles...")
    assert df["user_id"].is_unique, "Duplicate user_ids found"
    assert df["user_id"].notna().all(), "Null user_ids found"
    assert df["segment"].isin([
        "salaried_urban", "gig_worker", "rural_farmer", "shg_woman", "small_vendor"
    ]).all(), "Invalid segments found"
    assert (df["monthly_income_actual"] > 0).all(), "Non-positive incomes found"
    assert df["age"].between(18, 70).all(), "Ages outside 18-70 range"
    print(f"    ✓ {len(df)} users, {df['segment'].nunique()} segments")
    print(f"    ✓ Default rate: {df['default_flag'].mean():.1%}")
    return df


def validate_upi_transactions(df):
    """Validate UPI transactions dataset."""
    print("  Validating upi_transactions...")
    assert df["txn_id"].is_unique, "Duplicate txn_ids"
    assert df["amount"].gt(0).all(), "Non-positive amounts"
    assert df["txn_type"].isin(["credit", "debit"]).all(), "Invalid txn_types"
    print(f"    ✓ {len(df)} transactions, {df['user_id'].nunique()} users")
    return df


def validate_bill_payments(df):
    """Validate bill payments dataset."""
    print("  Validating bill_payments...")
    assert df["bill_id"].is_unique, "Duplicate bill_ids"
    assert df["bill_amount"].gt(0).all(), "Non-positive bill amounts"
    print(f"    ✓ {len(df)} bills, {df['bill_type'].nunique()} types")
    return df


def validate_land_records(df):
    """Validate land records dataset."""
    print("  Validating land_records...")
    assert df["record_id"].is_unique, "Duplicate record_ids"
    n_owners = (df["property_type"] != "none").sum()
    print(f"    ✓ {len(df)} records, {n_owners} with property ({n_owners/len(df):.0%})")
    return df


def validate_device_logs(df):
    """Validate device logs dataset."""
    print("  Validating device_logs...")
    assert df["log_id"].is_unique, "Duplicate log_ids"
    print(f"    ✓ {len(df)} logs, {df['user_id'].nunique()} users")
    return df


def validate_literacy_engagement(df):
    """Validate literacy engagement dataset."""
    print("  Validating literacy_engagement...")
    assert df["engagement_id"].is_unique, "Duplicate engagement_ids"
    assert df["quiz_score"].between(0, 100).all(), "Quiz scores outside 0-100"
    print(f"    ✓ {len(df)} records, {df['user_id'].nunique()} users")
    return df


def main():
    print("=" * 60)
    print("ArthaSetu v2 — Data Preprocessor & Validator")
    print("=" * 60)
    print()

    files = {
        "user_profiles": validate_user_profiles,
        "upi_transactions": validate_upi_transactions,
        "bill_payments": validate_bill_payments,
        "land_records": validate_land_records,
        "device_logs": validate_device_logs,
        "literacy_engagement": validate_literacy_engagement,
    }

    all_valid = True
    for name, validator in files.items():
        filepath = os.path.join(PROCESSED_DIR, f"{name}.csv")
        if not os.path.exists(filepath):
            print(f"  [MISSING] {name}.csv — run generate_synthetic.py first")
            all_valid = False
            continue

        try:
            df = pd.read_csv(filepath)
            validator(df)
        except Exception as e:
            print(f"  [FAIL] {name}.csv: {e}")
            all_valid = False

    print()
    if all_valid:
        print("✓ All datasets validated successfully!")
        print()
        print("Upload to Databricks Volume:")
        print("  databricks fs cp processed/user_profiles.csv     dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/")
        print("  databricks fs cp processed/upi_transactions.csv  dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/")
        print("  databricks fs cp processed/bill_payments.csv     dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/")
        print("  databricks fs cp processed/land_records.csv      dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/")
        print("  databricks fs cp processed/device_logs.csv       dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/")
        print("  databricks fs cp processed/literacy_engagement.csv dbfs:/Volumes/arthasetu/xscore_bronze/raw_uploads/")
    else:
        print("✗ Some validations failed — fix issues above")


if __name__ == "__main__":
    main()
