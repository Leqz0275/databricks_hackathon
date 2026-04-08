# Databricks notebook source
# ArthaSetu v2 — XScore Feature Engineering
# Computes 30+ features across 5 XScore components from Silver tables
# Writes feature store to gold.xscore_feature_store

# COMMAND ----------

import json
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *

CATALOG = "arthasetu"
SILVER_SCHEMA = f"{CATALOG}.xscore_silver"
GOLD_SCHEMA = f"{CATALOG}.xscore_gold"

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMPONENT 1: Payment Discipline Score (0-250)
# Strongest predictor of creditworthiness
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Component 1: Payment Discipline Features")
print("═" * 60)

# Load silver tables
bills_df = spark.table(f"{SILVER_SCHEMA}.silver_bill_payments")
upi_df = spark.table(f"{SILVER_SCHEMA}.silver_upi_transactions")

# Feature 1: Bill payment timeliness score
bill_timeliness = (
    bills_df
    .groupBy("user_id")
    .agg(
        F.count("*").alias("total_bills"),
        F.sum(F.col("is_on_time").cast("int")).alias("on_time_count"),
        F.sum(F.col("is_paid").cast("int")).alias("paid_count"),
    )
    .withColumn("bill_payment_timeliness_score",
        F.round(F.col("on_time_count") / F.col("total_bills"), 4)
    )
)

# Feature 2: UPI bill payment regularity (CV of monthly amounts)
monthly_bill_amounts = (
    bills_df
    .filter(F.col("is_paid") == True)
    .groupBy("user_id", "month")
    .agg(F.sum("bill_amount").alias("monthly_bill_total"))
)

bill_regularity = (
    monthly_bill_amounts
    .groupBy("user_id")
    .agg(
        F.avg("monthly_bill_total").alias("avg_monthly_bill"),
        F.stddev("monthly_bill_total").alias("std_monthly_bill"),
    )
    .withColumn("upi_bill_regularity",
        F.round(
            F.lit(1.0) - F.least(
                F.col("std_monthly_bill") / F.greatest(F.col("avg_monthly_bill"), F.lit(1)),
                F.lit(1.0)
            ), 4
        )
    )
)

# Feature 3: Rent payment consistency
rent_consistency = (
    bills_df
    .filter(F.col("bill_type") == "rent")
    .groupBy("user_id")
    .agg(
        F.count("*").alias("rent_months"),
        F.sum(F.col("is_on_time").cast("int")).alias("rent_on_time"),
        F.stddev("bill_amount").alias("rent_amount_std"),
        F.avg("bill_amount").alias("rent_amount_avg"),
    )
    .withColumn("rent_payment_consistency",
        F.when(F.col("rent_months") > 0,
            F.round(F.col("rent_on_time") / F.col("rent_months") *
                    (F.lit(1.0) - F.least(
                        F.col("rent_amount_std") / F.greatest(F.col("rent_amount_avg"), F.lit(1)),
                        F.lit(1.0)
                    )), 4)
        ).otherwise(F.lit(None))
    )
    .select("user_id", "rent_payment_consistency")
)

# Feature 4: Utility payment streak (longest consecutive on-time months)
utility_streak = (
    bills_df
    .filter(F.col("bill_type").isin("electricity", "water", "gas"))
    .groupBy("user_id")
    .agg(
        F.max("on_time_cumsum").alias("utility_payment_streak")
    )
)

# Feature 5: Merchant diversity index (Shannon entropy)
merchant_counts = (
    upi_df
    .filter(F.col("txn_type") == "debit")
    .groupBy("user_id", "merchant_category")
    .agg(F.count("*").alias("cat_count"))
)
total_per_user = (
    merchant_counts
    .groupBy("user_id")
    .agg(F.sum("cat_count").alias("total_txns"))
)
merchant_probs = (
    merchant_counts
    .join(total_per_user, "user_id")
    .withColumn("p", F.col("cat_count") / F.col("total_txns"))
    .withColumn("p_log_p", F.col("p") * F.log(F.col("p")))
)
merchant_diversity = (
    merchant_probs
    .groupBy("user_id")
    .agg(
        F.round(-F.sum("p_log_p"), 4).alias("merchant_diversity_index")
    )
)

# Feature 6: P2P vs merchant ratio
p2p_ratio = (
    upi_df
    .filter(F.col("txn_type") == "debit")
    .groupBy("user_id")
    .agg(
        F.sum(F.col("is_p2p").cast("int")).alias("p2p_count"),
        F.count("*").alias("total_debit"),
    )
    .withColumn("p2p_vs_merchant_ratio",
        F.round(F.col("p2p_count") / F.greatest(F.col("total_debit"), F.lit(1)), 4)
    )
    .select("user_id", "p2p_vs_merchant_ratio")
)

# Combine Payment Discipline features
payment_features = (
    bill_timeliness.select("user_id", "bill_payment_timeliness_score")
    .join(bill_regularity.select("user_id", "upi_bill_regularity"), "user_id", "left")
    .join(rent_consistency, "user_id", "left")
    .join(utility_streak, "user_id", "left")
    .join(merchant_diversity, "user_id", "left")
    .join(p2p_ratio, "user_id", "left")
)

print(f"  Payment Discipline features: {payment_features.count()} users")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMPONENT 2: Financial Stability Score (0-250)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Component 2: Financial Stability Features")
print("═" * 60)

# Monthly cashflow analysis
monthly_cashflow = (
    upi_df
    .groupBy("user_id", "month")
    .agg(
        F.sum(F.when(F.col("is_credit"), F.col("amount")).otherwise(0)).alias("monthly_inflow"),
        F.sum(F.when(~F.col("is_credit"), F.col("amount")).otherwise(0)).alias("monthly_outflow"),
    )
    .withColumn("monthly_net", F.col("monthly_inflow") - F.col("monthly_outflow"))
)

# Feature 1: Income stability (CV of monthly inflows)
income_stability = (
    monthly_cashflow
    .groupBy("user_id")
    .agg(
        F.avg("monthly_inflow").alias("avg_monthly_inflow"),
        F.stddev("monthly_inflow").alias("std_monthly_inflow"),
    )
    .withColumn("income_stability_cv",
        F.round(
            F.lit(1.0) - F.least(
                F.col("std_monthly_inflow") / F.greatest(F.col("avg_monthly_inflow"), F.lit(1)),
                F.lit(1.0)
            ), 4
        )
    )
)

# Feature 2: Savings ratio
savings_ratio = (
    monthly_cashflow
    .groupBy("user_id")
    .agg(
        F.sum("monthly_inflow").alias("total_inflow"),
        F.sum("monthly_outflow").alias("total_outflow"),
    )
    .withColumn("savings_ratio",
        F.round(
            (F.col("total_inflow") - F.col("total_outflow")) /
            F.greatest(F.col("total_inflow"), F.lit(1)),
            4
        )
    )
    .select("user_id", "savings_ratio")
)

# Feature 3: Cashflow trend (slope of monthly net cashflow)
# Using month index as x-axis for linear regression approximation
w_order = Window.partitionBy("user_id").orderBy("month")
cashflow_trend = (
    monthly_cashflow
    .withColumn("month_idx", F.row_number().over(w_order))
    .groupBy("user_id")
    .agg(
        F.count("*").alias("n_months"),
        F.sum("month_idx").alias("sum_x"),
        F.sum("monthly_net").alias("sum_y"),
        F.sum(F.col("month_idx") * F.col("monthly_net")).alias("sum_xy"),
        F.sum(F.col("month_idx") * F.col("month_idx")).alias("sum_xx"),
    )
    .withColumn("cashflow_trend",
        F.round(
            (F.col("n_months") * F.col("sum_xy") - F.col("sum_x") * F.col("sum_y")) /
            F.greatest(
                F.col("n_months") * F.col("sum_xx") - F.col("sum_x") * F.col("sum_x"),
                F.lit(1)
            ),
            2
        )
    )
    .select("user_id", "cashflow_trend")
)

# Feature 4: Expense to income ratio
expense_ratio = (
    monthly_cashflow
    .groupBy("user_id")
    .agg(
        F.avg("monthly_outflow").alias("avg_expense"),
        F.avg("monthly_inflow").alias("avg_income"),
    )
    .withColumn("expense_to_income_ratio",
        F.round(F.col("avg_expense") / F.greatest(F.col("avg_income"), F.lit(1)), 4)
    )
    .select("user_id", "expense_to_income_ratio")
)

# Feature 5: Income diversification (distinct credit sources)
income_diversification = (
    upi_df
    .filter(F.col("is_credit") == True)
    .groupBy("user_id")
    .agg(
        F.countDistinct("merchant_name").alias("income_diversification")
    )
)

# Feature 6: Emergency buffer months
emergency_buffer = (
    monthly_cashflow
    .groupBy("user_id")
    .agg(
        F.avg("monthly_outflow").alias("avg_monthly_expense"),
        F.sum("monthly_net").alias("cumulative_savings"),
    )
    .withColumn("emergency_buffer_months",
        F.round(
            F.greatest(F.col("cumulative_savings"), F.lit(0)) /
            F.greatest(F.col("avg_monthly_expense"), F.lit(1)),
            2
        )
    )
    .select("user_id", "emergency_buffer_months")
)

# Feature 7: Seasonal stability (variance across quarters)
seasonal_stability = (
    monthly_cashflow
    .withColumn("quarter", F.ceil(F.col("month") / 3))
    .groupBy("user_id", "quarter")
    .agg(F.sum("monthly_inflow").alias("quarterly_income"))
    .groupBy("user_id")
    .agg(
        F.stddev("quarterly_income").alias("quarterly_income_std"),
        F.avg("quarterly_income").alias("quarterly_income_avg"),
    )
    .withColumn("seasonal_stability",
        F.round(
            F.lit(1.0) - F.least(
                F.col("quarterly_income_std") / F.greatest(F.col("quarterly_income_avg"), F.lit(1)),
                F.lit(1.0)
            ), 4
        )
    )
    .select("user_id", "seasonal_stability")
)

# Combine Financial Stability features
stability_features = (
    income_stability.select("user_id", "income_stability_cv")
    .join(savings_ratio, "user_id", "left")
    .join(cashflow_trend, "user_id", "left")
    .join(expense_ratio, "user_id", "left")
    .join(income_diversification, "user_id", "left")
    .join(emergency_buffer, "user_id", "left")
    .join(seasonal_stability, "user_id", "left")
)

print(f"  Financial Stability features: {stability_features.count()} users")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMPONENT 3: Asset & Verification Score (0-150)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Component 3: Asset & Verification Features")
print("═" * 60)

land_df = spark.table(f"{SILVER_SCHEMA}.silver_land_records")
profiles_df = spark.table(f"{SILVER_SCHEMA}.silver_user_profiles")

asset_features = (
    land_df
    .select(
        "user_id",
        F.col("has_property").alias("land_ownership_flag"),
        "value_bucket",
        "ownership_score",
        "property_age_years",
        F.col("encumbrance_flag"),
    )
    .join(
        profiles_df.select(
            "user_id",
            F.col("aadhaar_linked").alias("aadhaar_linked_flag"),
            F.col("pan_linked").alias("pan_verified_flag"),
            "bank_account_vintage_months",
            "address_stability_years",
        ),
        "user_id", "left"
    )
    .withColumn("property_value_score",
        F.when(F.col("value_bucket") == "50L+", 1.0)
        .when(F.col("value_bucket") == "15-50L", 0.75)
        .when(F.col("value_bucket") == "5-15L", 0.5)
        .when(F.col("value_bucket") == "0-5L", 0.25)
        .otherwise(0.0)
    )
)

print(f"  Asset & Verification features: {asset_features.count()} users")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMPONENT 4: Digital Trust Score (0-100)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Component 4: Digital Trust Features")
print("═" * 60)

device_df = spark.table(f"{SILVER_SCHEMA}.silver_device_location")

# Device consistency (% of sessions from primary device)
device_consistency = (
    device_df
    .groupBy("user_id")
    .agg(
        F.count("*").alias("total_sessions"),
        F.sum(F.when(~F.col("device_change_flag"), 1).otherwise(0)).alias("primary_device_sessions"),
        F.countDistinct("device_id").alias("unique_devices"),
    )
    .withColumn("device_consistency_score",
        F.round(F.col("primary_device_sessions") / F.greatest(F.col("total_sessions"), F.lit(1)), 4)
    )
)

# Location stability (stddev of lat/lon as proxy for radius)
location_stability = (
    device_df
    .groupBy("user_id")
    .agg(
        F.stddev("location_lat").alias("lat_std"),
        F.stddev("location_lon").alias("lon_std"),
    )
    .withColumn("location_stability_score",
        F.round(
            F.lit(1.0) - F.least(
                F.sqrt(F.col("lat_std") * F.col("lat_std") + F.col("lon_std") * F.col("lon_std")) * 100,
                F.lit(1.0)
            ), 4
        )
    )
    .select("user_id", "location_stability_score")
)

# App engagement regularity (CV of session durations)
app_engagement = (
    device_df
    .groupBy("user_id")
    .agg(
        F.avg("session_duration_seconds").alias("avg_session"),
        F.stddev("session_duration_seconds").alias("std_session"),
    )
    .withColumn("app_engagement_regularity",
        F.round(
            F.lit(1.0) - F.least(
                F.col("std_session") / F.greatest(F.col("avg_session"), F.lit(1)),
                F.lit(1.0)
            ), 4
        )
    )
    .select("user_id", "app_engagement_regularity")
)

# Nighttime transaction ratio
nighttime_ratio = (
    upi_df
    .groupBy("user_id")
    .agg(
        F.sum(F.col("is_nighttime").cast("int")).alias("night_txns"),
        F.count("*").alias("total_txns"),
    )
    .withColumn("nighttime_txn_ratio",
        F.round(F.col("night_txns") / F.greatest(F.col("total_txns"), F.lit(1)), 4)
    )
    .select("user_id", "nighttime_txn_ratio")
)

# Combine Digital Trust features
digital_features = (
    device_consistency.select("user_id", "device_consistency_score", "unique_devices")
    .join(location_stability, "user_id", "left")
    .join(app_engagement, "user_id", "left")
    .join(nighttime_ratio, "user_id", "left")
    .join(
        profiles_df.select("user_id", "bank_account_vintage_months"),
        "user_id", "left"
    )
    .withColumnRenamed("bank_account_vintage_months", "account_age_months")
)

print(f"  Digital Trust features: {digital_features.count()} users")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMPONENT 5: Financial Awareness Score (0-150) — THE NOVEL SIGNAL
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Component 5: Financial Awareness Features (NOVEL)")
print("═" * 60)

literacy_df = spark.table(f"{SILVER_SCHEMA}.silver_literacy_engagement")

awareness_features = (
    literacy_df
    .groupBy("user_id")
    .agg(
        # Feature 1: Overall literacy score
        F.round(F.avg("quiz_score"), 2).alias("overall_literacy_score"),

        # Feature 2: Credit topic mastery
        F.round(F.avg(
            F.when(F.col("topic_category").isin("credit", "loans", "interest_rates"),
                   F.col("quiz_score"))
        ), 2).alias("credit_topic_mastery"),

        # Feature 3: Modules completed count
        F.sum(F.col("completion_flag").cast("int")).alias("modules_completed_count"),

        # Feature 4: Time investment hours
        F.round(F.sum("time_spent_seconds") / 3600.0, 2).alias("time_investment_hours"),

        # Feature 5: Number of attempts (lower is better)
        F.round(F.avg("attempts"), 2).alias("avg_attempts"),

        # Feature 6: Score range for improvement trajectory (max - min as proxy)
        (F.max("quiz_score") - F.min("quiz_score")).alias("score_range"),
    )
    .withColumn("improvement_trajectory",
        F.when(F.col("score_range") > 20, 1.0)
        .when(F.col("score_range") > 10, 0.5)
        .otherwise(0.0)
    )
)

# Handle users with no literacy data (set to 0/null)
print(f"  Financial Awareness features: {awareness_features.count()} users with data")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMBINE ALL FEATURES → gold.xscore_feature_store
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Combining all features into Feature Store")
print("═" * 60)

# Start with all user_ids
all_users = profiles_df.select("user_id", "segment", "default_flag", "default_probability")

# Join all component features
feature_store = (
    all_users
    .join(payment_features, "user_id", "left")
    .join(stability_features, "user_id", "left")
    .join(asset_features, "user_id", "left")
    .join(digital_features, "user_id", "left")
    .join(awareness_features, "user_id", "left")
    .withColumn("feature_timestamp", F.current_timestamp())
    .withColumn("has_upi_data",
        F.col("bill_payment_timeliness_score").isNotNull())
    .withColumn("has_bill_data",
        F.col("upi_bill_regularity").isNotNull())
    .withColumn("has_land_data",
        F.col("land_ownership_flag").isNotNull())
    .withColumn("has_digital_data",
        F.col("device_consistency_score").isNotNull())
    .withColumn("has_literacy_data",
        F.col("overall_literacy_score").isNotNull())
    .withColumn("data_completeness",
        (F.col("has_upi_data").cast("int") +
         F.col("has_bill_data").cast("int") +
         F.col("has_land_data").cast("int") +
         F.col("has_digital_data").cast("int") +
         F.col("has_literacy_data").cast("int")) / 5.0
    )
)

# Fill nulls with 0 for numeric features
numeric_cols = [
    "bill_payment_timeliness_score", "upi_bill_regularity",
    "rent_payment_consistency", "utility_payment_streak",
    "merchant_diversity_index", "p2p_vs_merchant_ratio",
    "income_stability_cv", "savings_ratio", "cashflow_trend",
    "expense_to_income_ratio", "income_diversification",
    "emergency_buffer_months", "seasonal_stability",
    "ownership_score", "property_value_score",
    "device_consistency_score", "location_stability_score",
    "app_engagement_regularity", "nighttime_txn_ratio",
    "overall_literacy_score", "credit_topic_mastery",
    "modules_completed_count", "time_investment_hours",
    "improvement_trajectory",
]
for c in numeric_cols:
    feature_store = feature_store.withColumn(c,
        F.coalesce(F.col(c), F.lit(0.0))
    )

# Write to Gold
(feature_store
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.xscore_feature_store")
)

n_features = len(numeric_cols)
n_users = feature_store.count()
print(f"\n✓ Feature store written: {n_users} users × {n_features} features")
print(f"  Data completeness distribution:")
feature_store.groupBy(
    F.when(F.col("data_completeness") >= 0.8, "High")
    .when(F.col("data_completeness") >= 0.6, "Medium")
    .otherwise("Low").alias("confidence")
).count().show()
