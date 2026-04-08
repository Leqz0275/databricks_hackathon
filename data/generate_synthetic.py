#!/usr/bin/env python3
"""
ArthaSetu v2 — Synthetic Data Generator
========================================
Generates 5 correlated synthetic datasets for the XScore credit scoring platform.
Each user's behavioral data is internally consistent and correlates with their
default probability to ensure the ML model can learn meaningful patterns.

Datasets generated:
  1. User Profiles        (1,000 users × 5 segments)
  2. UPI Transactions     (~180K transactions, 6 months)
  3. Bill Payment History (12 months per user)
  4. Land/Property Records (~40% of users)
  5. Device & Location Logs
  6. Financial Literacy Engagement

Usage:
  python generate_synthetic.py [--output-dir processed/] [--seed 42]
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# ─── Configuration ────────────────────────────────────────────────────────────

SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "processed")

SEGMENTS = {
    "salaried_urban": {
        "count": 200,
        "income_range": (20000, 80000),
        "base_default_rate": 0.08,
        "occupations": ["software_engineer", "bank_clerk", "teacher", "accountant", "govt_employee"],
        "education": ["graduate", "postgraduate"],
        "bill_types": ["electricity", "rent", "broadband", "mobile_recharge", "dth"],
        "states": ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "Telangana"],
        "has_smartphone_pct": 0.95,
        "aadhaar_pct": 0.92,
        "pan_pct": 0.85,
        "land_ownership_pct": 0.30,
    },
    "gig_worker": {
        "count": 250,
        "income_range": (10000, 30000),
        "base_default_rate": 0.15,
        "occupations": ["delivery_partner", "cab_driver", "freelancer", "daily_wage"],
        "education": ["secondary", "graduate"],
        "bill_types": ["electricity", "mobile_recharge", "rent"],
        "states": ["Maharashtra", "Karnataka", "Delhi", "Uttar Pradesh", "Rajasthan"],
        "has_smartphone_pct": 0.88,
        "aadhaar_pct": 0.80,
        "pan_pct": 0.45,
        "land_ownership_pct": 0.10,
    },
    "rural_farmer": {
        "count": 200,
        "income_range": (5000, 25000),
        "base_default_rate": 0.18,
        "occupations": ["farmer", "agri_laborer", "dairy_farmer", "fisherman"],
        "education": ["none", "primary", "secondary"],
        "bill_types": ["electricity", "mobile_recharge", "water"],
        "states": ["Madhya Pradesh", "Uttar Pradesh", "Rajasthan", "Bihar", "Odisha"],
        "has_smartphone_pct": 0.55,
        "aadhaar_pct": 0.75,
        "pan_pct": 0.20,
        "land_ownership_pct": 0.60,
    },
    "shg_woman": {
        "count": 200,
        "income_range": (3000, 15000),
        "base_default_rate": 0.10,
        "occupations": ["shg_member", "home_based_worker", "tailor", "handicraft"],
        "education": ["none", "primary", "secondary"],
        "bill_types": ["electricity", "mobile_recharge"],
        "states": ["Andhra Pradesh", "Telangana", "Odisha", "Bihar", "Jharkhand"],
        "has_smartphone_pct": 0.50,
        "aadhaar_pct": 0.85,
        "pan_pct": 0.15,
        "land_ownership_pct": 0.25,
    },
    "small_vendor": {
        "count": 150,
        "income_range": (8000, 40000),
        "base_default_rate": 0.14,
        "occupations": ["shopkeeper", "street_vendor", "chai_stall", "kirana_store"],
        "education": ["primary", "secondary", "graduate"],
        "bill_types": ["electricity", "mobile_recharge", "rent", "gas"],
        "states": ["Uttar Pradesh", "Maharashtra", "Gujarat", "Rajasthan", "Madhya Pradesh"],
        "has_smartphone_pct": 0.75,
        "aadhaar_pct": 0.82,
        "pan_pct": 0.55,
        "land_ownership_pct": 0.35,
    },
}

MERCHANT_CATEGORIES = [
    "grocery", "fuel", "restaurant", "pharmacy", "clothing",
    "electronics", "utilities", "transport", "education", "entertainment",
    "healthcare", "government", "insurance", "recharge", "rent"
]

GIG_PLATFORMS = ["swiggy", "zomato", "uber", "ola", "dunzo", "none"]

DISTRICTS = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore"],
    "Delhi": ["New Delhi", "South Delhi", "North Delhi", "East Delhi"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Karimnagar", "Nizamabad"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro"],
}

FIRST_NAMES_M = [
    "Rahul", "Amit", "Suresh", "Vikram", "Rajesh", "Kiran", "Manoj", "Deepak",
    "Arjun", "Ravi", "Sanjay", "Prakash", "Arun", "Nitin", "Gaurav", "Varun",
    "Rohit", "Sachin", "Dinesh", "Ashok", "Pankaj", "Naveen", "Sunil", "Hemant"
]
FIRST_NAMES_F = [
    "Priya", "Sunita", "Lakshmi", "Anjali", "Rekha", "Meena", "Kavita", "Asha",
    "Pooja", "Neha", "Deepa", "Sita", "Geeta", "Radha", "Anita", "Seema",
    "Rani", "Shanti", "Kamla", "Savitri", "Usha", "Durga", "Parvati", "Renu"
]
LAST_NAMES = [
    "Sharma", "Verma", "Kumar", "Singh", "Patel", "Reddy", "Yadav", "Gupta",
    "Das", "Joshi", "Nair", "Iyer", "Mehta", "Shah", "Patil", "Deshmukh",
    "Choudhary", "Mishra", "Pandey", "Tiwari", "Dubey", "Saxena", "Agarwal",
    "Jain", "Bhat", "Rao", "Naidu", "Pillai"
]

LANGUAGES = {
    "Maharashtra": "mr", "Karnataka": "kn", "Delhi": "hi",
    "Tamil Nadu": "ta", "Telangana": "te", "Uttar Pradesh": "hi",
    "Rajasthan": "hi", "Madhya Pradesh": "hi", "Bihar": "hi",
    "Odisha": "or", "Andhra Pradesh": "te", "Gujarat": "gu",
    "Jharkhand": "hi",
}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


# ─── Dataset 1: User Profiles ────────────────────────────────────────────────

def generate_user_profiles():
    """Generate 1,000 synthetic user profiles across 5 segments."""
    print("  Generating user profiles...")
    users = []
    user_id_counter = 1

    for segment_name, cfg in SEGMENTS.items():
        for _ in range(cfg["count"]):
            uid = f"USR_{user_id_counter:05d}"
            user_id_counter += 1

            state = random.choice(cfg["states"])
            district = random.choice(DISTRICTS.get(state, ["Unknown"]))

            is_female = segment_name == "shg_woman" or random.random() < 0.35
            gender = "F" if is_female else "M"
            first = random.choice(FIRST_NAMES_F if is_female else FIRST_NAMES_M)
            last = random.choice(LAST_NAMES)

            age = np.clip(int(np.random.normal(
                {"salaried_urban": 32, "gig_worker": 27, "rural_farmer": 42,
                 "shg_woman": 35, "small_vendor": 38}[segment_name], 8
            )), 18, 65)

            income = np.clip(
                np.random.lognormal(
                    np.log(np.mean(cfg["income_range"])),
                    0.4
                ),
                cfg["income_range"][0],
                cfg["income_range"][1]
            )

            education = random.choice(cfg["education"])
            occupation = random.choice(cfg["occupations"])
            aadhaar = random.random() < cfg["aadhaar_pct"]
            pan = random.random() < cfg["pan_pct"]
            has_smartphone = random.random() < cfg["has_smartphone_pct"]
            bank_vintage = int(np.clip(np.random.exponential(36), 3, 180))
            address_stability = round(np.clip(np.random.exponential(4), 0.5, 25), 1)

            gig_platform = "none"
            if segment_name == "gig_worker":
                gig_platform = random.choice(["swiggy", "zomato", "uber", "ola", "dunzo"])

            shg_member = segment_name == "shg_woman"

            # Income bucket
            if income < 10000:
                income_range = "5K-10K"
            elif income < 20000:
                income_range = "10K-20K"
            elif income < 50000:
                income_range = "20K-50K"
            else:
                income_range = "50K+"

            # Behavioral attributes (used for default label generation)
            savings_ratio = np.clip(np.random.normal(
                {"salaried_urban": 0.20, "gig_worker": 0.08, "rural_farmer": 0.05,
                 "shg_woman": 0.12, "small_vendor": 0.10}[segment_name], 0.08
            ), -0.1, 0.5)

            bill_timeliness = np.clip(np.random.beta(
                {"salaried_urban": 8, "gig_worker": 4, "rural_farmer": 3,
                 "shg_woman": 6, "small_vendor": 5}[segment_name],
                2
            ), 0.2, 1.0)

            income_cv = np.clip(np.random.exponential(
                {"salaried_urban": 0.10, "gig_worker": 0.30, "rural_farmer": 0.45,
                 "shg_woman": 0.20, "small_vendor": 0.25}[segment_name]
            ), 0.02, 0.8)

            nighttime_txn_ratio = np.clip(np.random.exponential(0.05), 0, 0.3)

            # Generate default probability based on behavioral features
            default_prob = cfg["base_default_rate"]
            if savings_ratio > 0.15:
                default_prob *= 0.7
            if bill_timeliness > 0.85:
                default_prob *= 0.6
            if pan:
                default_prob *= 0.75
            if random.random() < cfg["land_ownership_pct"]:
                default_prob *= 0.65
            if income_cv > 0.5:
                default_prob *= 1.2
            if nighttime_txn_ratio > 0.15:
                default_prob *= 1.3
            if income < 10000:
                default_prob *= 1.15

            default_prob = np.clip(default_prob, 0.01, 0.60)
            default_flag = random.random() < default_prob

            users.append({
                "user_id": uid,
                "name": f"{first} {last}",
                "age": age,
                "gender": gender,
                "state": state,
                "district": district,
                "language_pref": LANGUAGES.get(state, "hi"),
                "occupation": occupation,
                "segment": segment_name,
                "monthly_income_range": income_range,
                "monthly_income_actual": round(income, 2),
                "education": education,
                "aadhaar_linked": aadhaar,
                "pan_linked": pan,
                "bank_account_vintage_months": bank_vintage,
                "address_stability_years": address_stability,
                "shg_member": shg_member,
                "gig_platform": gig_platform,
                "has_smartphone": has_smartphone,
                "primary_device_os": "android" if has_smartphone else "feature_phone",
                "savings_ratio": round(savings_ratio, 4),
                "bill_payment_timeliness": round(bill_timeliness, 4),
                "income_cv": round(income_cv, 4),
                "nighttime_txn_ratio": round(nighttime_txn_ratio, 4),
                "default_flag": default_flag,
                "default_probability": round(default_prob, 4),
            })

    return pd.DataFrame(users)


# ─── Dataset 2: UPI Transactions ─────────────────────────────────────────────

def generate_upi_transactions(profiles_df):
    """Generate 6 months of UPI transactions per user."""
    print("  Generating UPI transactions...")
    txns = []
    txn_id = 1
    start_date = datetime(2025, 7, 1)
    end_date = datetime(2025, 12, 31)

    for _, user in profiles_df.iterrows():
        income = user["monthly_income_actual"]
        segment = user["segment"]

        # Transactions per month based on segment
        txn_per_month = {
            "salaried_urban": random.randint(40, 80),
            "gig_worker": random.randint(25, 55),
            "rural_farmer": random.randint(8, 20),
            "shg_woman": random.randint(10, 25),
            "small_vendor": random.randint(30, 70),
        }[segment]

        monthly_expense = income * (1 - user["savings_ratio"])

        for month_offset in range(6):
            month_start = start_date + timedelta(days=30 * month_offset)
            # Add income variability
            month_income = income * np.clip(np.random.normal(1, user["income_cv"]), 0.5, 2.0)
            month_expense = monthly_expense * np.random.normal(1, 0.15)

            n_txns = int(txn_per_month * np.random.normal(1, 0.2))
            n_txns = max(5, n_txns)

            # Split: ~15-25% credits, rest debits
            n_credits = max(1, int(n_txns * random.uniform(0.10, 0.25)))
            n_debits = n_txns - n_credits

            # Credit transactions (income)
            credit_amounts = np.random.dirichlet(np.ones(n_credits)) * month_income
            for i, amt in enumerate(credit_amounts):
                day = random.randint(1, 28)
                hour = random.randint(6, 22)
                minute = random.randint(0, 59)
                ts = month_start.replace(day=min(day, 28), hour=hour, minute=minute)

                # Salary credits are larger, gig credits are smaller & frequent
                if segment == "salaried_urban" and i == 0:
                    amt = month_income * 0.85  # Main salary
                    merchant = "employer_salary"
                elif segment == "gig_worker":
                    merchant = random.choice(["platform_payout", "customer_tip", "incentive"])
                else:
                    merchant = random.choice(["bank_transfer", "shg_payout", "govt_subsidy", "sale_receipt"])

                txns.append({
                    "txn_id": f"TXN_{txn_id:07d}",
                    "user_id": user["user_id"],
                    "amount": round(max(10, amt), 2),
                    "merchant_name": merchant,
                    "merchant_category": "income",
                    "txn_type": "credit",
                    "timestamp": ts.isoformat(),
                    "is_p2p": merchant in ["bank_transfer", "customer_tip"],
                })
                txn_id += 1

            # Debit transactions (expenses)
            debit_amounts = np.random.dirichlet(np.ones(n_debits)) * month_expense
            for amt in debit_amounts:
                day = random.randint(1, 28)
                # Nighttime ratio control
                if random.random() < user["nighttime_txn_ratio"]:
                    hour = random.randint(0, 5)
                else:
                    hour = random.randint(7, 23)
                minute = random.randint(0, 59)
                ts = month_start.replace(day=min(day, 28), hour=hour, minute=minute)

                category = random.choice(MERCHANT_CATEGORIES)
                is_p2p = random.random() < 0.2

                txns.append({
                    "txn_id": f"TXN_{txn_id:07d}",
                    "user_id": user["user_id"],
                    "amount": round(max(5, amt), 2),
                    "merchant_name": f"{category}_merchant_{random.randint(1,50)}",
                    "merchant_category": category,
                    "txn_type": "debit",
                    "timestamp": ts.isoformat(),
                    "is_p2p": is_p2p,
                })
                txn_id += 1

    return pd.DataFrame(txns)


# ─── Dataset 3: Bill Payment Histories ───────────────────────────────────────

def generate_bill_payments(profiles_df):
    """Generate 12 months of bill payment data per user."""
    print("  Generating bill payment histories...")
    bills = []
    bill_id = 1

    bill_amount_ranges = {
        "electricity":      (200, 3500),
        "water":            (50, 500),
        "mobile_recharge":  (149, 599),
        "rent":             (3000, 25000),
        "broadband":        (499, 1499),
        "dth":              (199, 599),
        "gas":              (300, 1200),
    }

    for _, user in profiles_df.iterrows():
        segment = user["segment"]
        timeliness = user["bill_payment_timeliness"]
        bill_types = SEGMENTS[segment]["bill_types"]

        for bill_type in bill_types:
            amt_range = bill_amount_ranges[bill_type]
            # Scale bill amounts by income
            income_factor = user["monthly_income_actual"] / 20000
            base_amount = random.uniform(amt_range[0], amt_range[1]) * np.clip(income_factor, 0.5, 2.0)

            for month in range(1, 13):
                due_date = datetime(2025, month if month <= 12 else month - 12, random.randint(5, 25))
                bill_amount = round(base_amount * np.random.normal(1, 0.1), 2)
                bill_amount = max(50, bill_amount)

                # Payment behavior correlated with timeliness score
                is_on_time = random.random() < timeliness
                if is_on_time:
                    days_offset = -random.randint(0, 5)  # Early or on time
                else:
                    days_offset = random.randint(1, 30)  # Late

                # Some bills may go unpaid (~3-10% based on segment)
                unpaid_chance = {"salaried_urban": 0.02, "gig_worker": 0.08,
                                 "rural_farmer": 0.10, "shg_woman": 0.05,
                                 "small_vendor": 0.06}[segment]
                is_unpaid = random.random() < unpaid_chance

                payment_date = None if is_unpaid else due_date + timedelta(days=days_offset)
                payment_mode = "none" if is_unpaid else random.choice(
                    ["upi", "upi", "upi", "auto_debit", "cash"]  # UPI-heavy
                )

                bills.append({
                    "bill_id": f"BILL_{bill_id:07d}",
                    "user_id": user["user_id"],
                    "bill_type": bill_type,
                    "bill_amount": bill_amount,
                    "due_date": due_date.strftime("%Y-%m-%d"),
                    "payment_date": payment_date.strftime("%Y-%m-%d") if payment_date else None,
                    "payment_mode": payment_mode,
                    "days_before_after_due": days_offset if not is_unpaid else None,
                    "is_on_time": is_on_time and not is_unpaid,
                    "month": month,
                    "year": 2025,
                })
                bill_id += 1

    return pd.DataFrame(bills)


# ─── Dataset 4: Land/Property Records ────────────────────────────────────────

def generate_land_records(profiles_df):
    """Generate land/property records for ~40% of users."""
    print("  Generating land/property records...")
    records = []
    rec_id = 1

    property_configs = {
        "salaried_urban": {
            "types": ["residential_plot", "house", "flat"],
            "value_range": (500000, 5000000),
        },
        "gig_worker": {
            "types": ["none"],
            "value_range": (0, 0),
        },
        "rural_farmer": {
            "types": ["agricultural_land", "agricultural_land", "house"],
            "value_range": (100000, 2000000),
        },
        "shg_woman": {
            "types": ["house"],
            "value_range": (50000, 800000),
        },
        "small_vendor": {
            "types": ["shop", "residential_plot", "house"],
            "value_range": (200000, 3000000),
        },
    }

    for _, user in profiles_df.iterrows():
        segment = user["segment"]
        has_property = random.random() < SEGMENTS[segment]["land_ownership_pct"]

        if not has_property:
            records.append({
                "record_id": f"LAND_{rec_id:05d}",
                "user_id": user["user_id"],
                "property_type": "none",
                "ownership_status": "none",
                "estimated_value_inr": 0,
                "location_district": user["district"],
                "registration_year": None,
                "area_sqft_or_acres": 0,
                "encumbrance_flag": False,
            })
            rec_id += 1
            continue

        cfg = property_configs[segment]
        prop_type = random.choice(cfg["types"])
        if prop_type == "none":
            prop_type = "residential_plot"

        value = random.uniform(cfg["value_range"][0], cfg["value_range"][1])
        ownership = random.choice(["owned", "joint", "family"] if segment != "shg_woman"
                                   else ["joint", "family", "family"])

        area = 0
        if "land" in prop_type:
            area = round(random.uniform(0.5, 10), 2)  # acres
        elif prop_type in ["residential_plot", "house", "flat", "shop"]:
            area = round(random.uniform(200, 2500), 0)  # sqft

        records.append({
            "record_id": f"LAND_{rec_id:05d}",
            "user_id": user["user_id"],
            "property_type": prop_type,
            "ownership_status": ownership,
            "estimated_value_inr": round(value, 0),
            "location_district": user["district"],
            "registration_year": random.randint(2000, 2024),
            "area_sqft_or_acres": area,
            "encumbrance_flag": random.random() < 0.15,
        })
        rec_id += 1

    return pd.DataFrame(records)


# ─── Dataset 5: Device & Location Logs ───────────────────────────────────────

def generate_device_logs(profiles_df):
    """Generate device and location stability logs."""
    print("  Generating device & location logs...")
    logs = []
    log_id = 1
    start_date = datetime(2025, 7, 1)

    # Base coordinates per state
    state_coords = {
        "Maharashtra": (19.07, 72.87), "Karnataka": (12.97, 77.59),
        "Delhi": (28.61, 77.20), "Tamil Nadu": (13.08, 80.27),
        "Telangana": (17.38, 78.47), "Uttar Pradesh": (26.85, 80.95),
        "Rajasthan": (26.92, 75.79), "Madhya Pradesh": (23.26, 77.41),
        "Bihar": (25.60, 85.14), "Odisha": (20.30, 85.82),
        "Andhra Pradesh": (16.50, 80.64), "Gujarat": (23.02, 72.57),
        "Jharkhand": (23.35, 85.33),
    }

    for _, user in profiles_df.iterrows():
        if not user["has_smartphone"]:
            continue

        base_lat, base_lon = state_coords.get(user["state"], (20.0, 78.0))
        # Add district-level offset
        base_lat += random.uniform(-1, 1)
        base_lon += random.uniform(-1, 1)

        # Device consistency: good users use 1-2 devices, risky use 3+
        if user["default_flag"]:
            n_devices = random.randint(2, 4)
        else:
            n_devices = random.randint(1, 2)

        devices = [f"DEV_{user['user_id']}_{i}" for i in range(n_devices)]
        primary_device = devices[0]

        # Location stability: stable users have small radius
        location_jitter = 0.01 if user["address_stability_years"] > 3 else 0.05

        # Generate weekly logs for 6 months (~26 weeks)
        for week in range(26):
            ts = start_date + timedelta(weeks=week, hours=random.randint(8, 20))

            device = primary_device if random.random() < 0.85 else random.choice(devices)
            lat = base_lat + np.random.normal(0, location_jitter)
            lon = base_lon + np.random.normal(0, location_jitter)

            session_dur = int(np.clip(np.random.exponential(
                {"salaried_urban": 300, "gig_worker": 180,
                 "rural_farmer": 120, "shg_woman": 200,
                 "small_vendor": 240}[user["segment"]]
            ), 30, 3600))

            section = random.choice(["home", "literacy", "credit_check", "profile"])

            logs.append({
                "log_id": f"LOG_{log_id:07d}",
                "user_id": user["user_id"],
                "timestamp": ts.isoformat(),
                "device_id": device,
                "device_change_flag": device != primary_device,
                "location_lat": round(lat, 6),
                "location_lon": round(lon, 6),
                "session_duration_seconds": session_dur,
                "platform_section": section,
            })
            log_id += 1

    return pd.DataFrame(logs)


# ─── Dataset 6: Financial Literacy Engagement ────────────────────────────────

def generate_literacy_engagement(profiles_df):
    """Generate financial literacy module engagement data."""
    print("  Generating literacy engagement data...")
    engagements = []
    eng_id = 1

    modules = [
        ("LIT_001", "credit", 3),
        ("LIT_002", "loans", 4),
        ("LIT_003", "savings", 2),
        ("LIT_004", "upi_safety", 2),
        ("LIT_005", "budgeting", 3),
        ("LIT_006", "interest_rates", 4),
    ]

    for _, user in profiles_df.iterrows():
        # Number of modules completed correlates inversely with default probability
        max_modules = int(np.clip(
            6 - user["default_probability"] * 10 + np.random.normal(0, 1),
            0, 6
        ))

        if max_modules == 0:
            continue

        selected_modules = random.sample(modules, min(max_modules, len(modules)))
        base_date = datetime(2025, 7, 1)

        streak_days = 0
        last_date = None

        for i, (mod_id, topic, difficulty) in enumerate(selected_modules):
            session_date = base_date + timedelta(days=random.randint(i * 5, i * 10 + 15))

            # Quiz score: inversely correlated with default probability
            base_score = int(np.clip(
                np.random.normal(75 - user["default_probability"] * 50, 15),
                20, 100
            ))

            completion = random.random() < (0.9 - user["default_probability"])
            time_spent = int(np.clip(np.random.normal(600, 200), 120, 2400))
            attempts = 1 if base_score > 70 else random.randint(1, 3)

            # Track streak
            if last_date and (session_date - last_date).days <= 2:
                streak_days += 1
            else:
                streak_days = 1
            last_date = session_date

            lang = user["language_pref"]
            if random.random() < 0.3:
                lang = "en"  # Some use English

            engagements.append({
                "engagement_id": f"ENG_{eng_id:06d}",
                "user_id": user["user_id"],
                "module_id": mod_id,
                "topic_category": topic,
                "quiz_score": base_score,
                "completion_flag": completion,
                "time_spent_seconds": time_spent,
                "attempts": attempts,
                "difficulty_level": difficulty,
                "session_timestamp": session_date.isoformat(),
                "language_used": lang,
            })
            eng_id += 1

    return pd.DataFrame(engagements)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for ArthaSetu v2")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for CSVs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("ArthaSetu v2 — Synthetic Data Generator")
    print("=" * 60)

    # Generate all datasets
    print("\n[1/6] User Profiles")
    profiles = generate_user_profiles()
    profiles.to_csv(os.path.join(args.output_dir, "user_profiles.csv"), index=False)
    print(f"       → {len(profiles)} users across {profiles['segment'].nunique()} segments")
    print(f"       → Default rate: {profiles['default_flag'].mean():.1%}")

    print("\n[2/6] UPI Transactions")
    upi = generate_upi_transactions(profiles)
    upi.to_csv(os.path.join(args.output_dir, "upi_transactions.csv"), index=False)
    print(f"       → {len(upi)} transactions")

    print("\n[3/6] Bill Payments")
    bills = generate_bill_payments(profiles)
    bills.to_csv(os.path.join(args.output_dir, "bill_payments.csv"), index=False)
    print(f"       → {len(bills)} bill records")

    print("\n[4/6] Land Records")
    land = generate_land_records(profiles)
    land.to_csv(os.path.join(args.output_dir, "land_records.csv"), index=False)
    print(f"       → {len(land)} records ({(land['property_type'] != 'none').sum()} with property)")

    print("\n[5/6] Device & Location Logs")
    devices = generate_device_logs(profiles)
    devices.to_csv(os.path.join(args.output_dir, "device_logs.csv"), index=False)
    print(f"       → {len(devices)} log entries")

    print("\n[6/6] Literacy Engagement")
    literacy = generate_literacy_engagement(profiles)
    literacy.to_csv(os.path.join(args.output_dir, "literacy_engagement.csv"), index=False)
    print(f"       → {len(literacy)} engagement records")

    print("\n" + "=" * 60)
    print(f"All datasets saved to: {args.output_dir}/")
    print("=" * 60)

    # Summary statistics
    print("\n── Segment Summary ──")
    for seg in SEGMENTS:
        seg_df = profiles[profiles["segment"] == seg]
        print(f"  {seg:20s}: {len(seg_df):4d} users | "
              f"Avg income ₹{seg_df['monthly_income_actual'].mean():,.0f} | "
              f"Default rate {seg_df['default_flag'].mean():.1%}")


if __name__ == "__main__":
    main()
