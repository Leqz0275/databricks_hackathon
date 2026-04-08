# Databricks notebook source
# ArthaSetu v2 — Delta Live Tables Pipeline
# Bronze (Volume) → Silver (Cleaned+Enriched) → Gold (Aggregated)
#
# Catalog: arthasetu
# DLT Pipeline: arthasetu_data_pipeline

import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

VOLUME = "/Volumes/arthasetu/xscore_bronze/raw_uploads"

# ═══════════════════════════════════════════════════════════════════════
# BRONZE — Raw data ingest from Volume (uploaded CSVs)
# ═══════════════════════════════════════════════════════════════════════

@dlt.table(
    name="bronze_user_profiles",
    comment="Raw synthetic user profiles — 1000 users across 5 segments",
    table_properties={"quality": "bronze"}
)
def bronze_user_profiles():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{VOLUME}/user_profiles.csv")
        .withColumn("ingested_at", current_timestamp())
    )

@dlt.table(
    name="bronze_upi_transactions",
    comment="Raw synthetic UPI transactions — 6 months of debit/credit data",
    table_properties={"quality": "bronze"}
)
def bronze_upi_transactions():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{VOLUME}/upi_transactions.csv")
        .withColumn("ingested_at", current_timestamp())
    )

@dlt.table(
    name="bronze_bill_payments",
    comment="Raw synthetic bill payment histories — 12 months per user",
    table_properties={"quality": "bronze"}
)
def bronze_bill_payments():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{VOLUME}/bill_payments.csv")
        .withColumn("ingested_at", current_timestamp())
    )

@dlt.table(
    name="bronze_land_records",
    comment="Raw synthetic land/property ownership records",
    table_properties={"quality": "bronze"}
)
def bronze_land_records():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{VOLUME}/land_records.csv")
        .withColumn("ingested_at", current_timestamp())
    )

@dlt.table(
    name="bronze_device_logs",
    comment="Raw synthetic device and location stability logs",
    table_properties={"quality": "bronze"}
)
def bronze_device_logs():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{VOLUME}/device_logs.csv")
        .withColumn("ingested_at", current_timestamp())
    )

@dlt.table(
    name="bronze_literacy_engagement",
    comment="Raw synthetic financial literacy module engagement data",
    table_properties={"quality": "bronze"}
)
def bronze_literacy_engagement():
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(f"{VOLUME}/literacy_engagement.csv")
        .withColumn("ingested_at", current_timestamp())
    )


# ═══════════════════════════════════════════════════════════════════════
# SILVER — Cleaned, validated, enriched
# ═══════════════════════════════════════════════════════════════════════

@dlt.table(
    name="silver_user_profiles",
    comment="Cleaned user profiles with derived demographic features",
    table_properties={"quality": "silver"}
)
@dlt.expect("valid_user_id", "user_id IS NOT NULL")
@dlt.expect("valid_segment", "segment IN ('salaried_urban','gig_worker','rural_farmer','shg_woman','small_vendor')")
@dlt.expect("valid_age", "age BETWEEN 18 AND 70")
@dlt.expect("valid_income", "monthly_income_actual > 0")
def silver_user_profiles():
    return (
        dlt.read("bronze_user_profiles")
        .filter(col("user_id").isNotNull())
        .filter(col("monthly_income_actual") > 0)
        .withColumn("income_bucket",
            when(col("monthly_income_actual") < 10000, "low")
            .when(col("monthly_income_actual") < 25000, "medium")
            .when(col("monthly_income_actual") < 50000, "high")
            .otherwise("very_high")
        )
        .withColumn("kyc_completeness",
            (col("aadhaar_linked").cast("int") +
             col("pan_linked").cast("int")) / 2.0
        )
        .withColumn("updated_at", current_timestamp())
    )


@dlt.table(
    name="silver_upi_transactions",
    comment="Cleaned UPI transactions with time-based and category features",
    table_properties={"quality": "silver"}
)
@dlt.expect("valid_txn_id", "txn_id IS NOT NULL")
@dlt.expect("valid_amount", "amount > 0")
@dlt.expect("valid_type", "txn_type IN ('credit', 'debit')")
def silver_upi_transactions():
    return (
        dlt.read("bronze_upi_transactions")
        .filter(col("txn_id").isNotNull())
        .filter(col("amount") > 0)
        .withColumn("timestamp", to_timestamp(col("timestamp")))
        .withColumn("txn_date", to_date(col("timestamp")))
        .withColumn("hour_of_day", hour(col("timestamp")))
        .withColumn("day_of_week", dayofweek(col("timestamp")))
        .withColumn("month", month(col("timestamp")))
        .withColumn("is_credit", col("txn_type") == "credit")
        .withColumn("is_nighttime", (col("hour_of_day") >= 0) & (col("hour_of_day") <= 5))
        .withColumn("amount_bucket",
            when(col("amount") < 100, "micro")
            .when(col("amount") < 500, "small")
            .when(col("amount") < 2000, "medium")
            .when(col("amount") < 10000, "large")
            .otherwise("very_large")
        )
        .withColumn("updated_at", current_timestamp())
    )


@dlt.table(
    name="silver_bill_payments",
    comment="Cleaned bill payments with on-time flags and streak calculations",
    table_properties={"quality": "silver"}
)
@dlt.expect("valid_bill_id", "bill_id IS NOT NULL")
@dlt.expect("valid_amount", "bill_amount > 0")
def silver_bill_payments():
    from pyspark.sql.window import Window

    bills = (
        dlt.read("bronze_bill_payments")
        .filter(col("bill_id").isNotNull())
        .filter(col("bill_amount") > 0)
        .withColumn("due_date", to_date(col("due_date")))
        .withColumn("payment_date", to_date(col("payment_date")))
        .withColumn("is_paid", col("payment_date").isNotNull())
        .withColumn("days_late",
            when(col("payment_date").isNotNull(),
                 datediff(col("payment_date"), col("due_date")))
            .otherwise(lit(999))  # Unpaid = very late
        )
        .withColumn("payment_category",
            when(col("payment_date").isNull(), "unpaid")
            .when(col("days_before_after_due") <= -3, "early")
            .when(col("days_before_after_due") <= 0, "on_time")
            .when(col("days_before_after_due") <= 7, "slightly_late")
            .otherwise("very_late")
        )
    )

    # Calculate payment streak per user per bill_type
    w = Window.partitionBy("user_id", "bill_type").orderBy("month")
    bills = (
        bills
        .withColumn("on_time_cumsum",
            sum(col("is_on_time").cast("int")).over(w)
        )
        .withColumn("updated_at", current_timestamp())
    )

    return bills


@dlt.table(
    name="silver_land_records",
    comment="Cleaned land records with value bucketing and ownership scoring",
    table_properties={"quality": "silver"}
)
@dlt.expect("valid_record_id", "record_id IS NOT NULL")
def silver_land_records():
    return (
        dlt.read("bronze_land_records")
        .filter(col("record_id").isNotNull())
        .withColumn("has_property", col("property_type") != "none")
        .withColumn("value_bucket",
            when(col("estimated_value_inr") == 0, "none")
            .when(col("estimated_value_inr") < 500000, "0-5L")
            .when(col("estimated_value_inr") < 1500000, "5-15L")
            .when(col("estimated_value_inr") < 5000000, "15-50L")
            .otherwise("50L+")
        )
        .withColumn("ownership_score",
            when(col("ownership_status") == "owned", 1.0)
            .when(col("ownership_status") == "joint", 0.7)
            .when(col("ownership_status") == "family", 0.4)
            .when(col("ownership_status") == "leased", 0.2)
            .otherwise(0.0)
        )
        .withColumn("property_age_years",
            when(col("registration_year").isNotNull(),
                 lit(2025) - col("registration_year"))
            .otherwise(lit(0))
        )
        .withColumn("updated_at", current_timestamp())
    )


@dlt.table(
    name="silver_device_location",
    comment="Cleaned device logs with stability metrics per user",
    table_properties={"quality": "silver"}
)
@dlt.expect("valid_log_id", "log_id IS NOT NULL")
def silver_device_location():
    return (
        dlt.read("bronze_device_logs")
        .filter(col("log_id").isNotNull())
        .withColumn("timestamp", to_timestamp(col("timestamp")))
        .withColumn("updated_at", current_timestamp())
    )


@dlt.table(
    name="silver_literacy_engagement",
    comment="Cleaned literacy engagement with completion and scoring metrics",
    table_properties={"quality": "silver"}
)
@dlt.expect("valid_engagement_id", "engagement_id IS NOT NULL")
@dlt.expect("valid_quiz_score", "quiz_score BETWEEN 0 AND 100")
def silver_literacy_engagement():
    return (
        dlt.read("bronze_literacy_engagement")
        .filter(col("engagement_id").isNotNull())
        .filter(col("quiz_score").between(0, 100))
        .withColumn("session_timestamp", to_timestamp(col("session_timestamp")))
        .withColumn("session_date", to_date(col("session_timestamp")))
        .withColumn("score_category",
            when(col("quiz_score") >= 80, "excellent")
            .when(col("quiz_score") >= 60, "good")
            .when(col("quiz_score") >= 40, "fair")
            .otherwise("needs_improvement")
        )
        .withColumn("time_spent_minutes", round(col("time_spent_seconds") / 60.0, 1))
        .withColumn("updated_at", current_timestamp())
    )


# ═══════════════════════════════════════════════════════════════════════
# GOLD — Aggregated analytics tables
# ═══════════════════════════════════════════════════════════════════════

@dlt.table(
    name="gold_user_profiles",
    comment="Enriched user profiles with all dimension attributes",
    table_properties={"quality": "gold",
                      "delta.enableChangeDataFeed": "true"}
)
def gold_user_profiles():
    """Union of user profile data with KYC and segment info."""
    return (
        dlt.read("silver_user_profiles")
        .select(
            "user_id", "name", "age", "gender", "state", "district",
            "language_pref", "occupation", "segment", "monthly_income_range",
            "monthly_income_actual", "education", "aadhaar_linked", "pan_linked",
            "bank_account_vintage_months", "address_stability_years",
            "shg_member", "gig_platform", "has_smartphone", "primary_device_os",
            "income_bucket", "kyc_completeness",
            "default_flag", "default_probability",
            "updated_at"
        )
    )


@dlt.table(
    name="gold_user_engagement_metrics",
    comment="Aggregated literacy engagement per user — feeds XScore awareness component",
    table_properties={"quality": "gold",
                      "delta.enableChangeDataFeed": "true"}
)
def gold_user_engagement_metrics():
    """Aggregate literacy engagement metrics per user for XScore."""
    return (
        dlt.read("silver_literacy_engagement")
        .groupBy("user_id")
        .agg(
            count("*").alias("total_engagements"),
            countDistinct("module_id").alias("modules_attempted"),
            sum(col("completion_flag").cast("int")).alias("modules_completed"),
            round(avg("quiz_score"), 2).alias("avg_quiz_score"),
            round(avg(
                when(col("topic_category").isin("credit", "loans", "interest_rates"),
                     col("quiz_score"))
            ), 2).alias("credit_topic_avg_score"),
            round(sum("time_spent_seconds") / 3600.0, 2).alias("total_hours_spent"),
            max("session_date").alias("last_session_date"),
            round(avg(
                when(col("score_category") == "excellent", 1.0)
                .when(col("score_category") == "good", 0.7)
                .when(col("score_category") == "fair", 0.4)
                .otherwise(0.1)
            ), 3).alias("performance_index"),
        )
        .withColumn("updated_at", current_timestamp())
    )


@dlt.table(
    name="gold_bank_portfolio_analytics",
    comment="Segment-level portfolio analytics for bank dashboard",
    table_properties={"quality": "gold"}
)
def gold_bank_portfolio_analytics():
    """Portfolio-level statistics per segment for bank risk officers."""
    profiles = dlt.read("silver_user_profiles")
    return (
        profiles
        .groupBy("segment")
        .agg(
            count("*").alias("user_count"),
            round(avg("monthly_income_actual"), 0).alias("avg_income"),
            round(avg("default_probability"), 4).alias("avg_default_prob"),
            sum(col("default_flag").cast("int")).alias("default_count"),
            round(avg("kyc_completeness"), 3).alias("avg_kyc_completeness"),
            round(avg("bank_account_vintage_months"), 1).alias("avg_bank_vintage"),
            round(avg("address_stability_years"), 1).alias("avg_address_stability"),
        )
        .withColumn("default_rate",
            round(col("default_count") / col("user_count"), 4)
        )
        .withColumn("updated_at", current_timestamp())
    )
