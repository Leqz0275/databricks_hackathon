# Databricks notebook source
# ArthaSetu v2 — XScore Model Training
# Two-stage scoring: Component Models + GBT Meta-Model
# K-Means segmentation + MLflow tracking + SHAP explainability

# COMMAND ----------

import mlflow
import mlflow.spark
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import numpy as np
import pandas as pd
import json

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

CATALOG = "arthasetu"
GOLD_SCHEMA = f"{CATALOG}.xscore_gold"

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# LOAD FEATURE STORE
# ═══════════════════════════════════════════════════════════════════════

print("Loading XScore Feature Store...")
feature_df = spark.table(f"{GOLD_SCHEMA}.xscore_feature_store")
print(f"  {feature_df.count()} users loaded")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# STAGE 0: K-Means User Segmentation
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Stage 0: K-Means User Segmentation")
print("═" * 60)

seg_features = [
    "bill_payment_timeliness_score", "income_stability_cv", "savings_ratio",
    "expense_to_income_ratio", "merchant_diversity_index",
    "overall_literacy_score", "modules_completed_count",
]

seg_assembler = VectorAssembler(inputCols=seg_features, outputCol="seg_features_vec")
seg_scaler = StandardScaler(inputCol="seg_features_vec", outputCol="seg_scaled", withMean=True, withStd=True)
kmeans = KMeans(featuresCol="seg_scaled", predictionCol="cluster_id", k=5, seed=42)

seg_pipeline = Pipeline(stages=[seg_assembler, seg_scaler, kmeans])

mlflow.set_experiment(f"/Users/arthasetu_team/xscore_segmentation")

with mlflow.start_run(run_name="kmeans_segmentation"):
    mlflow.set_tags({"project": "arthasetu", "component": "segmentation"})

    seg_model = seg_pipeline.fit(feature_df)
    clustered_df = seg_model.transform(feature_df)

    # Map cluster IDs to segment names
    cluster_summary = (
        clustered_df
        .groupBy("cluster_id", "segment")
        .count()
        .orderBy("cluster_id", F.desc("count"))
    )

    # Log cluster sizes
    cluster_sizes = clustered_df.groupBy("cluster_id").count().toPandas()
    for _, row in cluster_sizes.iterrows():
        mlflow.log_metric(f"cluster_{int(row['cluster_id'])}_size", int(row["count"]))

    mlflow.log_params({
        "n_clusters": 5,
        "features": json.dumps(seg_features),
    })

    mlflow.spark.log_model(seg_model, "kmeans_model",
                           registered_model_name="arthasetu_segmentation")

    print("  Cluster distribution:")
    cluster_summary.show(25, truncate=False)

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: Component Score Models (5 × LogisticRegression)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Stage 1: Component Score Models")
print("═" * 60)

# Define features per component
COMPONENT_FEATURES = {
    "payment_discipline": {
        "features": [
            "bill_payment_timeliness_score", "upi_bill_regularity",
            "rent_payment_consistency", "utility_payment_streak",
            "merchant_diversity_index", "p2p_vs_merchant_ratio",
        ],
        "max_score": 250,
    },
    "financial_stability": {
        "features": [
            "income_stability_cv", "savings_ratio", "cashflow_trend",
            "expense_to_income_ratio", "income_diversification",
            "emergency_buffer_months", "seasonal_stability",
        ],
        "max_score": 250,
    },
    "asset_verification": {
        "features": [
            "land_ownership_flag", "property_value_score", "ownership_score",
            "aadhaar_linked_flag", "pan_verified_flag",
            "bank_account_vintage_months", "address_stability_years",
        ],
        "max_score": 150,
    },
    "digital_trust": {
        "features": [
            "device_consistency_score", "location_stability_score",
            "app_engagement_regularity", "nighttime_txn_ratio",
            "account_age_months",
        ],
        "max_score": 100,
    },
    "financial_awareness": {
        "features": [
            "overall_literacy_score", "credit_topic_mastery",
            "modules_completed_count", "time_investment_hours",
            "improvement_trajectory",
        ],
        "max_score": 150,
    },
}

# Cast boolean columns to numeric
bool_cols = ["land_ownership_flag", "aadhaar_linked_flag", "pan_verified_flag"]
for c in bool_cols:
    feature_df = feature_df.withColumn(c, F.col(c).cast("double"))

# Prepare label: default_flag (we invert: 1 = good, 0 = default)
feature_df = feature_df.withColumn("label", (~F.col("default_flag")).cast("double"))

# Train/test split
train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)

component_scores = {}  # Store column names for later

mlflow.set_experiment(f"/Users/arthasetu_team/xscore_components")

for comp_name, comp_cfg in COMPONENT_FEATURES.items():
    print(f"\n  Training: {comp_name} (max {comp_cfg['max_score']})")

    features = comp_cfg["features"]
    max_score = comp_cfg["max_score"]

    assembler = VectorAssembler(inputCols=features, outputCol=f"{comp_name}_vec",
                                handleInvalid="skip")
    scaler = StandardScaler(inputCol=f"{comp_name}_vec",
                            outputCol=f"{comp_name}_scaled")
    lr = LogisticRegression(featuresCol=f"{comp_name}_scaled", labelCol="label",
                            maxIter=100, regParam=0.01)

    pipeline = Pipeline(stages=[assembler, scaler, lr])

    with mlflow.start_run(run_name=f"component_{comp_name}"):
        mlflow.set_tags({
            "project": "arthasetu",
            "component": comp_name,
            "max_score": max_score,
        })

        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        # Evaluate
        evaluator_auc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
        )
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )

        auc = evaluator_auc.evaluate(predictions)
        acc = evaluator_acc.evaluate(predictions)

        mlflow.log_params({
            "features": json.dumps(features),
            "max_score": max_score,
            "n_features": len(features),
        })
        mlflow.log_metrics({
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
        })

        mlflow.spark.log_model(model, f"{comp_name}_model",
                               registered_model_name=f"arthasetu_{comp_name}")

        print(f"    AUC: {auc:.4f} | Accuracy: {acc:.4f}")

    # Generate component scores for all data
    scored_df = model.transform(feature_df)

    # Extract probability of positive class and scale to component range
    score_col = f"{comp_name}_score"
    from pyspark.ml.functions import vector_to_array
    feature_df = (
        scored_df
        .withColumn(f"{comp_name}_prob",
            vector_to_array(F.col("probability"))[1]
        )
        .withColumn(score_col,
            F.round(F.col(f"{comp_name}_prob") * max_score).cast("int")
        )
        .select(feature_df.columns + [score_col])
    )
    component_scores[comp_name] = score_col

print("\n  ✓ All 5 component models trained")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: GBT Meta-Model (Component Scores → Refined XScore)
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Stage 2: GBT Meta-Model")
print("═" * 60)

# ─── VERSION 1: Without literacy features ─────────────────────────────

meta_features_v1 = [
    "payment_discipline_score", "financial_stability_score",
    "asset_verification_score", "digital_trust_score",
]

meta_features_v2 = meta_features_v1 + ["financial_awareness_score"]

# String index the segment for GBT
seg_indexer = StringIndexer(inputCol="segment", outputCol="segment_idx",
                            handleInvalid="keep")
feature_df = seg_indexer.fit(feature_df).transform(feature_df)

meta_features_full = meta_features_v2 + ["segment_idx", "data_completeness"]

mlflow.set_experiment(f"/Users/arthasetu_team/xscore_meta_gbt")

# V1: Without literacy
print("\n  Training V1 (without literacy)...")
assembler_v1 = VectorAssembler(inputCols=meta_features_v1 + ["segment_idx"],
                                outputCol="meta_features_v1",
                                handleInvalid="skip")
gbt_v1 = GBTClassifier(featuresCol="meta_features_v1", labelCol="label",
                         maxIter=100, maxDepth=5, seed=42)
pipeline_v1 = Pipeline(stages=[assembler_v1, gbt_v1])

train_df_meta, test_df_meta = feature_df.randomSplit([0.8, 0.2], seed=42)

with mlflow.start_run(run_name="gbt_meta_v1_no_literacy"):
    mlflow.set_tags({"project": "arthasetu", "model": "meta_gbt", "version": "v1"})

    model_v1 = pipeline_v1.fit(train_df_meta)
    preds_v1 = model_v1.transform(test_df_meta)

    auc_v1 = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction"
    ).evaluate(preds_v1)

    mlflow.log_params({"features": json.dumps(meta_features_v1), "version": "v1"})
    mlflow.log_metrics({"auc": round(auc_v1, 4)})
    mlflow.spark.log_model(model_v1, "gbt_meta_v1",
                           registered_model_name="arthasetu_xscore_meta")

    print(f"    V1 AUC (no literacy): {auc_v1:.4f}")

# V2: With literacy
print("\n  Training V2 (with literacy)...")
assembler_v2 = VectorAssembler(inputCols=meta_features_full,
                                outputCol="meta_features_v2",
                                handleInvalid="skip")
gbt_v2 = GBTClassifier(featuresCol="meta_features_v2", labelCol="label",
                         maxIter=100, maxDepth=5, seed=42)
pipeline_v2 = Pipeline(stages=[assembler_v2, gbt_v2])

with mlflow.start_run(run_name="gbt_meta_v2_with_literacy"):
    mlflow.set_tags({"project": "arthasetu", "model": "meta_gbt", "version": "v2"})

    model_v2 = pipeline_v2.fit(train_df_meta)
    preds_v2 = model_v2.transform(test_df_meta)

    auc_v2 = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction"
    ).evaluate(preds_v2)

    mlflow.log_params({"features": json.dumps(meta_features_full), "version": "v2"})
    mlflow.log_metrics({"auc": round(auc_v2, 4)})
    mlflow.spark.log_model(model_v2, "gbt_meta_v2",
                           registered_model_name="arthasetu_xscore_meta")

    auc_improvement = auc_v2 - auc_v1
    mlflow.log_metric("auc_improvement_from_literacy", round(auc_improvement, 4))

    print(f"    V2 AUC (with literacy): {auc_v2:.4f}")
    print(f"    ✓ Literacy feature improvement: +{auc_improvement:.4f} AUC")

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# COMPUTE FINAL XSCORES
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Computing Final XScores")
print("═" * 60)

# Segment-specific weights (from architecture doc)
SEGMENT_WEIGHTS = {
    "salaried_urban":  {"payment": 0.30, "stability": 0.30, "asset": 0.15, "digital": 0.10, "awareness": 0.15},
    "gig_worker":      {"payment": 0.25, "stability": 0.35, "asset": 0.10, "digital": 0.10, "awareness": 0.20},
    "rural_farmer":    {"payment": 0.20, "stability": 0.20, "asset": 0.30, "digital": 0.05, "awareness": 0.25},
    "shg_woman":       {"payment": 0.25, "stability": 0.20, "asset": 0.15, "digital": 0.05, "awareness": 0.35},
    "small_vendor":    {"payment": 0.30, "stability": 0.25, "asset": 0.15, "digital": 0.10, "awareness": 0.20},
}

# Weighted aggregation for interpretable score
xscore_df = feature_df

# Build weighted score expression per segment
for seg, weights in SEGMENT_WEIGHTS.items():
    xscore_df = xscore_df.withColumn(
        f"weighted_{seg}",
        (
            F.col("payment_discipline_score") / 250.0 * weights["payment"] +
            F.col("financial_stability_score") / 250.0 * weights["stability"] +
            F.col("asset_verification_score") / 150.0 * weights["asset"] +
            F.col("digital_trust_score") / 100.0 * weights["digital"] +
            F.col("financial_awareness_score") / 150.0 * weights["awareness"]
        ) * 900
    )

# Select the correct weighted score based on segment
xscore_df = xscore_df.withColumn("xscore_weighted",
    F.when(F.col("segment") == "salaried_urban", F.col("weighted_salaried_urban"))
    .when(F.col("segment") == "gig_worker", F.col("weighted_gig_worker"))
    .when(F.col("segment") == "rural_farmer", F.col("weighted_rural_farmer"))
    .when(F.col("segment") == "shg_woman", F.col("weighted_shg_woman"))
    .when(F.col("segment") == "small_vendor", F.col("weighted_small_vendor"))
    .otherwise(F.col("weighted_salaried_urban"))
)

# Apply GBT meta-model refinement
scored_v2 = model_v2.transform(xscore_df)
xscore_df = (
    scored_v2
    .withColumn("gbt_prob", vector_to_array(F.col("probability"))[1])
    .withColumn("xscore_refined", F.round(F.col("gbt_prob") * 900).cast("int"))
    # Final score: 60% weighted + 40% GBT refined
    .withColumn("xscore",
        F.round(F.col("xscore_weighted") * 0.6 + F.col("xscore_refined") * 0.4).cast("int")
    )
    .withColumn("xscore", F.least(F.col("xscore"), F.lit(900)))
    .withColumn("xscore", F.greatest(F.col("xscore"), F.lit(0)))
)

# Score bands
xscore_df = xscore_df.withColumn("score_band",
    F.when(F.col("xscore") >= 750, "Excellent")
    .when(F.col("xscore") >= 650, "Good")
    .when(F.col("xscore") >= 500, "Fair")
    .when(F.col("xscore") >= 300, "Needs Improvement")
    .otherwise("Insufficient Data")
)

# Confidence level
xscore_df = xscore_df.withColumn("confidence",
    F.when(F.col("data_completeness") >= 0.8, "High")
    .when(F.col("data_completeness") >= 0.6, "Medium")
    .otherwise("Low")
)

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# WRITE XSCORES TO GOLD
# ═══════════════════════════════════════════════════════════════════════

print("Writing XScores to Gold tables...")

# gold.xscores — main score table
xscores_final = xscore_df.select(
    "user_id", "xscore", "score_band", "confidence",
    "payment_discipline_score", "financial_stability_score",
    "asset_verification_score", "digital_trust_score",
    "financial_awareness_score",
    "segment", "data_completeness",
    F.current_timestamp().alias("score_timestamp"),
)

(xscores_final
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.xscores")
)

# gold.xscore_components — detailed component breakdown
components_final = xscore_df.select(
    "user_id",
    "payment_discipline_score", "financial_stability_score",
    "asset_verification_score", "digital_trust_score",
    "financial_awareness_score",
    F.current_timestamp().alias("component_timestamp"),
)

(components_final
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_SCHEMA}.xscore_components")
)

print("  ✓ gold.xscores written")
print("  ✓ gold.xscore_components written")

# Summary
print("\n── XScore Distribution ──")
xscores_final.groupBy("score_band").count().orderBy("score_band").show()
print("\n── Confidence Distribution ──")
xscores_final.groupBy("confidence").count().orderBy("confidence").show()
print("\n── Score Statistics ──")
xscores_final.select(
    F.avg("xscore").alias("avg"),
    F.min("xscore").alias("min"),
    F.max("xscore").alias("max"),
    F.stddev("xscore").alias("stddev"),
).show()

# COMMAND ----------
# ═══════════════════════════════════════════════════════════════════════
# SHAP EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════

print("═" * 60)
print("Generating SHAP Explanations")
print("═" * 60)

# For SHAP we use the sklearn approach on a pandas sample
try:
    import shap

    sample_pd = feature_df.select(
        meta_features_full + ["label"]
    ).limit(500).toPandas()

    X_sample = sample_pd[meta_features_full].values
    y_sample = sample_pd["label"].values

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    gbt_sklearn = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    gbt_sklearn.fit(X_scaled, y_sample)

    explainer = shap.TreeExplainer(gbt_sklearn)
    shap_values = explainer.shap_values(X_scaled)

    # Log average SHAP values per feature
    avg_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = dict(zip(meta_features_full, avg_shap.tolist()))

    with mlflow.start_run(run_name="shap_analysis"):
        mlflow.set_tags({"project": "arthasetu", "component": "explainability"})
        for feat, importance in shap_importance.items():
            mlflow.log_metric(f"shap_{feat}", round(importance, 6))

    print("  SHAP feature importance (avg |SHAP value|):")
    for feat, imp in sorted(shap_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:35s}: {imp:.6f}")

    print("\n  ✓ SHAP explanations computed and logged to MLflow")

except ImportError:
    print("  [SKIP] SHAP not installed — pip install shap")
except Exception as e:
    print(f"  [WARN] SHAP computation failed: {e}")

# COMMAND ----------

print("\n" + "═" * 60)
print("XScore Model Training Complete!")
print("═" * 60)
print(f"\n  V1 AUC (no literacy):   {auc_v1:.4f}")
print(f"  V2 AUC (with literacy): {auc_v2:.4f}")
print(f"  Literacy improvement:   +{auc_improvement:.4f}")
print(f"\n  Models registered in MLflow:")
print(f"    - arthasetu_segmentation (K-Means)")
print(f"    - arthasetu_payment_discipline (LogReg)")
print(f"    - arthasetu_financial_stability (LogReg)")
print(f"    - arthasetu_asset_verification (LogReg)")
print(f"    - arthasetu_digital_trust (LogReg)")
print(f"    - arthasetu_financial_awareness (LogReg)")
print(f"    - arthasetu_xscore_meta (GBT v1 + v2)")
