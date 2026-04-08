# Databricks notebook source
# ArthaSetu v2 — CDF (Change Data Feed) Triggers
# Automatic recalculations when literacy engagement updates XScore

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime

CATALOG = "arthasetu"
GOLD_SCHEMA = f"{CATALOG}.xscore_gold"

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# CDF TRIGGER 1: Literacy Engagement → XScore Recalculation
# When a user completes a new literacy module, recalculate their
# Financial Awareness component and update their XScore
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("CDF Trigger 1: Literacy → XScore Recalculation")
print("═" * 60)

# Read CDF changes from engagement metrics
try:
    engagement_changes = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", 0)  # In production, track last processed version
        .table(f"{GOLD_SCHEMA}.user_engagement_metrics")
        .filter(F.col("_change_type").isin("insert", "update_postimage"))
    )

    changed_users = engagement_changes.select("user_id").distinct()
    n_changed = changed_users.count()
    print(f"  Users with engagement changes: {n_changed}")

    if n_changed > 0:
        # Recalculate Financial Awareness component for changed users
        print("  Recalculating Financial Awareness scores...")

        # Get updated engagement metrics
        current_engagement = spark.table(f"{GOLD_SCHEMA}.user_engagement_metrics")

        # Compute new awareness features
        updated_awareness = (
            current_engagement
            .filter(F.col("user_id").isin(
                [row.user_id for row in changed_users.collect()]
            ))
            .select(
                "user_id",
                F.col("avg_quiz_score").alias("overall_literacy_score"),
                F.col("credit_topic_avg_score").alias("credit_topic_mastery"),
                F.col("modules_completed").alias("modules_completed_count"),
                F.col("total_hours_spent").alias("time_investment_hours"),
                F.col("performance_index").alias("improvement_trajectory"),
            )
        )

        # Compute new awareness sub-score (0-150)
        updated_awareness = updated_awareness.withColumn(
            "financial_awareness_score_new",
            F.round(
                (
                    F.coalesce(F.col("overall_literacy_score"), F.lit(0)) / 100.0 * 0.25 +
                    F.coalesce(F.col("credit_topic_mastery"), F.lit(0)) / 100.0 * 0.25 +
                    F.least(F.col("modules_completed_count") / 6.0, F.lit(1.0)) * 0.20 +
                    F.least(F.col("time_investment_hours") / 10.0, F.lit(1.0)) * 0.15 +
                    F.coalesce(F.col("improvement_trajectory"), F.lit(0)) * 0.15
                ) * 150
            ).cast("int")
        )

        # MERGE updated scores back into xscore_components
        updated_awareness.createOrReplaceTempView("updated_awareness_view")

        spark.sql(f"""
            MERGE INTO {GOLD_SCHEMA}.xscore_components AS target
            USING updated_awareness_view AS source
            ON target.user_id = source.user_id
            WHEN MATCHED THEN
                UPDATE SET
                    financial_awareness_score = source.financial_awareness_score_new,
                    component_timestamp = current_timestamp()
        """)

        print(f"  ✓ Updated {n_changed} users' Financial Awareness scores")

        # Recalculate XScore for affected users
        # Load current XScores and update with new awareness
        spark.sql(f"""
            MERGE INTO {GOLD_SCHEMA}.xscores AS target
            USING (
                SELECT
                    c.user_id,
                    c.payment_discipline_score,
                    c.financial_stability_score,
                    c.asset_verification_score,
                    c.digital_trust_score,
                    c.financial_awareness_score,
                    -- Simple weighted recalculation
                    CAST(ROUND(
                        (c.payment_discipline_score / 250.0 * 0.27 +
                         c.financial_stability_score / 250.0 * 0.26 +
                         c.asset_verification_score / 150.0 * 0.17 +
                         c.digital_trust_score / 100.0 * 0.08 +
                         c.financial_awareness_score / 150.0 * 0.22) * 900
                    ) AS INT) AS new_xscore
                FROM {GOLD_SCHEMA}.xscore_components c
                WHERE c.user_id IN (SELECT user_id FROM updated_awareness_view)
            ) AS source
            ON target.user_id = source.user_id
            WHEN MATCHED THEN
                UPDATE SET
                    xscore = source.new_xscore,
                    financial_awareness_score = source.financial_awareness_score,
                    score_band = CASE
                        WHEN source.new_xscore >= 750 THEN 'Excellent'
                        WHEN source.new_xscore >= 650 THEN 'Good'
                        WHEN source.new_xscore >= 500 THEN 'Fair'
                        WHEN source.new_xscore >= 300 THEN 'Needs Improvement'
                        ELSE 'Insufficient Data'
                    END,
                    score_timestamp = current_timestamp()
        """)

        print(f"  ✓ XScores recalculated for {n_changed} users")

except Exception as e:
    print(f"  [INFO] CDF not available yet: {e}")
    print("  This is expected on first run — CDF starts tracking after first write.")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# CDF TRIGGER 2: XScore Change → Literacy Recommendations
# When a user's XScore changes and awareness is low, recommend modules
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("CDF Trigger 2: XScore → Literacy Recommendations")
print("═" * 60)

try:
    xscore_changes = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", 0)
        .table(f"{GOLD_SCHEMA}.xscores")
        .filter(F.col("_change_type").isin("insert", "update_postimage"))
    )

    # Find users with low awareness scores
    low_awareness_users = (
        xscore_changes
        .filter(F.col("financial_awareness_score") < 60)  # Below 40% of max (150)
        .select("user_id", "xscore", "financial_awareness_score", "segment")
    )

    n_low = low_awareness_users.count()
    print(f"  Users with low Financial Awareness: {n_low}")

    if n_low > 0:
        # Generate module recommendations
        # Map segments to priority modules
        MODULE_RECOMMENDATIONS = {
            "salaried_urban": ["LIT_001", "LIT_006", "LIT_002"],
            "gig_worker": ["LIT_002", "LIT_003", "LIT_005"],
            "rural_farmer": ["LIT_003", "LIT_001", "LIT_004"],
            "shg_woman": ["LIT_001", "LIT_003", "LIT_002"],
            "small_vendor": ["LIT_002", "LIT_005", "LIT_006"],
        }

        recommendations = []
        for row in low_awareness_users.collect():
            segment = row.segment or "salaried_urban"
            modules = MODULE_RECOMMENDATIONS.get(segment, ["LIT_001", "LIT_002", "LIT_003"])
            recommendations.append({
                "user_id": row.user_id,
                "recommended_modules": ",".join(modules),
                "current_awareness_score": row.financial_awareness_score,
                "estimated_improvement": min(50, 150 - row.financial_awareness_score),
                "priority": "high" if row.financial_awareness_score < 30 else "medium",
                "generated_at": datetime.now().isoformat(),
            })

        rec_df = spark.createDataFrame(recommendations)
        (rec_df
            .write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{GOLD_SCHEMA}.literacy_recommendations")
        )

        print(f"  ✓ Generated {len(recommendations)} literacy recommendations")
        rec_df.show(10, truncate=False)

except Exception as e:
    print(f"  [INFO] CDF not available: {e}")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# CDF TRIGGER 3: XScore → Bank Portfolio Refresh
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("CDF Trigger 3: XScore → Portfolio Analytics Refresh")
print("═" * 60)

try:
    # Recompute portfolio analytics from current XScores
    xscores = spark.table(f"{GOLD_SCHEMA}.xscores")

    portfolio = (
        xscores
        .groupBy("segment")
        .agg(
            F.count("*").alias("user_count"),
            F.round(F.avg("xscore"), 1).alias("avg_xscore"),
            F.round(F.min("xscore"), 0).alias("min_xscore"),
            F.round(F.max("xscore"), 0).alias("max_xscore"),
            F.sum(F.when(F.col("score_band") == "Excellent", 1).otherwise(0)).alias("excellent_count"),
            F.sum(F.when(F.col("score_band") == "Good", 1).otherwise(0)).alias("good_count"),
            F.sum(F.when(F.col("score_band") == "Fair", 1).otherwise(0)).alias("fair_count"),
            F.sum(F.when(F.col("score_band") == "Needs Improvement", 1).otherwise(0)).alias("needs_improvement_count"),
            F.sum(F.when(F.col("score_band") == "Insufficient Data", 1).otherwise(0)).alias("insufficient_count"),
            F.sum(F.when(F.col("confidence") == "High", 1).otherwise(0)).alias("high_confidence_count"),
        )
        .withColumn("approval_rate_at_600",
            F.round(
                (F.col("excellent_count") + F.col("good_count") + F.col("fair_count")) /
                F.greatest(F.col("user_count"), F.lit(1)), 3
            )
        )
        .withColumn("batch_timestamp", F.current_timestamp())
    )

    (portfolio
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD_SCHEMA}.bank_portfolio_analytics")
    )

    print("  ✓ Portfolio analytics refreshed")
    portfolio.show(truncate=False)

except Exception as e:
    print(f"  [ERROR] Portfolio refresh failed: {e}")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# DEMO: Time Travel — Score Progression
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Time Travel Demo Queries")
print("═" * 60)

print("""
-- Demo SQL: Rahul's XScore Journey Over Time

-- Version 1: Initial score (only UPI data, no literacy)
SELECT user_id, xscore, score_band, confidence,
       payment_discipline_score, financial_awareness_score
FROM gold.xscores VERSION AS OF 1
WHERE user_id = 'USR_00042';

-- Latest version: After completing literacy modules + 3 months bills
SELECT user_id, xscore, score_band, confidence,
       payment_discipline_score, financial_awareness_score
FROM gold.xscores
WHERE user_id = 'USR_00042';

-- Full version history
DESCRIBE HISTORY arthasetu.xscore_gold.xscores;
""")

# Show version history if available
try:
    history = spark.sql(f"DESCRIBE HISTORY {GOLD_SCHEMA}.xscores")
    print("  XScore table version history:")
    history.select("version", "timestamp", "operation", "operationMetrics").show(truncate=False)
except Exception as e:
    print(f"  [INFO] Version history not available: {e}")

print("\n✓ CDF Triggers Pipeline Complete!")
