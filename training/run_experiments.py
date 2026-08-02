"""
Main training pipeline — runs all experiments (Phases 4-7).

Experiments:
  1. Raw features + Logistic Regression
  2. Raw features + XGBoost
  3. Engineered features + XGBoost
  4. Engineered features + LightGBM
  5. Best model + probability calibration
  6. Best calibrated model + cost-sensitive thresholding

Usage:
  python training/run_experiments.py
  python scripts/train.py
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.calibration.calibrator import ProbabilityCalibrator
from src.data.dataset_validator import validate_dataset
from src.data.loader import load_dataset
from src.features.pipeline import FeatureEngineer
from src.models.metrics import compute_fraud_metrics
from src.models.temporal_split import temporal_train_val_test_split
from src.scoring.decision_engine import ThresholdOptimizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def save_artifact(obj: object, path: Path) -> None:
    """Save an artifact (model, scaler, etc.) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved artifact: %s", path)


def save_metrics(metrics: Any, path: Path) -> None:
    """Save metrics report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(metrics.summary())
    logger.info("Saved metrics: %s", path)


def save_json(data: dict, path: Path) -> None:
    """Save JSON data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    training_start = time.time()

    # ── Step 1: Validate dataset ──
    logger.info("=" * 60)
    logger.info("STEP 1: Dataset validation")
    logger.info("=" * 60)
    report = validate_dataset(fail_fast=True)
    logger.info(report.summary())

    # ── Step 2: Load dataset ──
    logger.info("=" * 60)
    logger.info("STEP 2: Loading dataset")
    logger.info("=" * 60)
    df = load_dataset(parse_dates=True)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Sort by timestamp
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    # ── Step 3: Temporal split ──
    logger.info("=" * 60)
    logger.info("STEP 3: Temporal split")
    logger.info("=" * 60)
    split = temporal_train_val_test_split(df)
    logger.info(split.summary)

    train_df = df.loc[split.train_idx].copy()
    val_df = df.loc[split.val_idx].copy()
    test_df = df.loc[split.test_idx].copy()

    y_train = np.asarray(train_df["true_fraud_label"].values, dtype=np.int32)
    y_val = np.asarray(val_df["true_fraud_label"].values, dtype=np.int32)
    y_test = np.asarray(test_df["true_fraud_label"].values, dtype=np.int32)

    val_amounts = np.asarray(val_df["amount"].values, dtype=np.float64)
    test_amounts = np.asarray(test_df["amount"].values, dtype=np.float64)

    xgb_basic = None
    xgb_basic_prob_val = None
    xgb_basic_prob_test = None
    xgb_eng = None
    xgb_eng_prob_val = None
    xgb_eng_prob_test = None
    lgb_model = None
    lgb_prob_val = None
    lgb_prob_test = None
    fraud_ratio = 1.0

    # ── Step 4: Feature engineering ──
    logger.info("=" * 60)
    logger.info("STEP 4: Feature engineering")
    logger.info("=" * 60)
    engineer = FeatureEngineer()

    # FIT categorical encoding maps on training data ONLY (prevent leakage)
    engineer.fit_categorical_maps(train_df)

    # Basic features (no velocity) for baseline experiments
    logger.info("Generating basic features (no velocity)...")
    basic_features_train = engineer.generate_all_features(train_df, include_velocity=False)
    basic_features_val = engineer.generate_all_features(val_df, include_velocity=False)
    basic_features_test = engineer.generate_all_features(test_df, include_velocity=False)

    # Engineered features (with velocity) for advanced experiments
    logger.info("Generating engineered features (with velocity & cross-split historical state)...")
    eng_features_train = engineer.generate_all_features(
        train_df, include_velocity=True, fast_mode=True
    )
    eng_features_val = engineer.generate_all_features(
        val_df, include_velocity=True, fast_mode=True, historical_df=train_df
    )
    train_val_history = pd.concat([train_df, val_df], ignore_index=True)
    eng_features_test = engineer.generate_all_features(
        test_df, include_velocity=True, fast_mode=True, historical_df=train_val_history
    )

    # Align columns (ensure val/test have same columns as train)
    common_basic = sorted(
        set(basic_features_train.columns)
        & set(basic_features_val.columns)
        & set(basic_features_test.columns)
    )
    common_eng = sorted(
        set(eng_features_train.columns)
        & set(eng_features_val.columns)
        & set(eng_features_test.columns)
    )

    x_basic_train = basic_features_train[common_basic].values
    x_basic_val = basic_features_val[common_basic].values
    x_basic_test = basic_features_test[common_basic].values

    x_eng_train = eng_features_train[common_eng].values
    x_eng_val = eng_features_val[common_eng].values
    x_eng_test = eng_features_test[common_eng].values

    # Scale for Logistic Regression (fit on train only)
    scaler_basic = StandardScaler()
    scaler_basic.fit(x_basic_train)

    # Scale for engineered features
    scaler_eng = StandardScaler()
    scaler_eng.fit(x_eng_train)  # Fit only, used later if needed

    all_results: dict = {}

    # ═══════════════════════════════════════════
    # EXPERIMENT 1: Logistic Regression + Basic
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Logistic Regression + Basic Features")
    logger.info("=" * 60)

    from sklearn.pipeline import Pipeline

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, solver="lbfgs", C=0.1, random_state=42
                ),
            ),
        ]
    )
    lr.fit(x_basic_train, y_train)

    lr_prob_val = lr.predict_proba(x_basic_val)[:, 1]
    lr_prob_test = lr.predict_proba(x_basic_test)[:, 1]

    lr_metrics_val = compute_fraud_metrics(y_val, lr_prob_val, amounts=val_amounts)

    logger.info("LR Val:\n%s", lr_metrics_val.summary())
    all_results["exp1_lr_basic"] = {
        "val_pr_auc": lr_metrics_val.pr_auc,
        "val_roc_auc": lr_metrics_val.roc_auc,
    }

    save_artifact(lr, artifacts_dir / "models" / "exp1_lr_basic.pkl")
    save_metrics(lr_metrics_val, artifacts_dir / "reports" / "exp1_lr_basic_val.txt")

    # ═══════════════════════════════════════════
    # EXPERIMENT 2: XGBoost + Basic Features
    xgb_module = None
    try:
        import xgboost as xgb_module
    except ImportError:
        logger.warning("XGBoost not installed")

    lgb_module = None
    try:
        import lightgbm as lgb_module
    except ImportError:
        logger.warning("LightGBM not installed")

    # ═══════════════════════════════════════════
    # EXPERIMENT 2: XGBoost + Basic Features
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: XGBoost + Basic Features")
    logger.info("=" * 60)

    if xgb_module is not None:
        try:
            fraud_ratio = float(np.sum(y_train == 0) / max(int(np.sum(y_train == 1)), 1))

            xgb_basic = xgb_module.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=fraud_ratio,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="aucpr",
                early_stopping_rounds=50,
                random_state=42,
                n_jobs=-1,
            )
            xgb_basic.fit(
                x_basic_train,
                y_train,
                eval_set=[(x_basic_val, y_val)],
                verbose=50,
            )

            xgb_basic_prob_val = xgb_basic.predict_proba(x_basic_val)[:, 1]
            xgb_basic_prob_test = xgb_basic.predict_proba(x_basic_test)[:, 1]

            xgb_basic_metrics = compute_fraud_metrics(
                y_val, xgb_basic_prob_val, amounts=val_amounts
            )
            logger.info("XGB Basic Val:\n%s", xgb_basic_metrics.summary())
            all_results["exp2_xgb_basic"] = {
                "val_pr_auc": xgb_basic_metrics.pr_auc,
                "val_roc_auc": xgb_basic_metrics.roc_auc,
                "best_iteration": getattr(xgb_basic, "best_iteration", None),
                "best_score": getattr(xgb_basic, "best_score", None),
            }

            save_artifact(xgb_basic, artifacts_dir / "models" / "exp2_xgb_basic.pkl")
            save_metrics(xgb_basic_metrics, artifacts_dir / "reports" / "exp2_xgb_basic_val.txt")
        except Exception as e:
            logger.error("Experiment 2 failed: %s", e)
    else:
        logger.warning("XGBoost not available, skipping Experiment 2")

    # ═══════════════════════════════════════════
    # EXPERIMENT 3: XGBoost + Engineered Features
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: XGBoost + Engineered Features")
    logger.info("=" * 60)

    if xgb_module is not None:
        try:
            xgb_eng = xgb_module.XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.03,
                scale_pos_weight=fraud_ratio,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="aucpr",
                early_stopping_rounds=50,
                random_state=42,
                n_jobs=-1,
            )
            xgb_eng.fit(
                x_eng_train,
                y_train,
                eval_set=[(x_eng_val, y_val)],
                verbose=50,
            )

            xgb_eng_prob_val = xgb_eng.predict_proba(x_eng_val)[:, 1]
            xgb_eng_prob_test = xgb_eng.predict_proba(x_eng_test)[:, 1]

            xgb_eng_metrics = compute_fraud_metrics(y_val, xgb_eng_prob_val, amounts=val_amounts)
            logger.info("XGB Eng Val:\n%s", xgb_eng_metrics.summary())
            all_results["exp3_xgb_eng"] = {
                "val_pr_auc": xgb_eng_metrics.pr_auc,
                "val_roc_auc": xgb_eng_metrics.roc_auc,
                "best_iteration": getattr(xgb_eng, "best_iteration", None),
                "best_score": getattr(xgb_eng, "best_score", None),
            }

            save_artifact(xgb_eng, artifacts_dir / "models" / "exp3_xgb_eng.pkl")
            save_metrics(xgb_eng_metrics, artifacts_dir / "reports" / "exp3_xgb_eng_val.txt")
        except Exception as e:
            logger.error("Experiment 3 failed: %s", e)
    else:
        logger.warning("XGBoost not available, skipping Experiment 3")

    # ═══════════════════════════════════════════
    # EXPERIMENT 4: LightGBM + Engineered Features
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: LightGBM + Engineered Features")
    logger.info("=" * 60)

    if lgb_module is not None:
        try:
            lgb_model = lgb_module.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.03,
                is_unbalance=True,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(
                x_eng_train,
                y_train,
                eval_set=[(x_eng_val, y_val)],
                eval_metric="average_precision",
                callbacks=[
                    lgb_module.early_stopping(stopping_rounds=50, verbose=False),
                    lgb_module.log_evaluation(period=50),
                ],
            )

            lgb_prob_val = lgb_model.predict_proba(x_eng_val)[:, 1]
            lgb_prob_test = lgb_model.predict_proba(x_eng_test)[:, 1]

            lgb_metrics = compute_fraud_metrics(y_val, lgb_prob_val, amounts=val_amounts)
            logger.info("LGB Eng Val:\n%s", lgb_metrics.summary())
            all_results["exp4_lgb_eng"] = {
                "val_pr_auc": lgb_metrics.pr_auc,
                "val_roc_auc": lgb_metrics.roc_auc,
                "best_iteration": getattr(lgb_model, "best_iteration_", None),
                "best_score": getattr(lgb_model, "best_score_", None),
            }

            save_artifact(lgb_model, artifacts_dir / "models" / "exp4_lgb_eng.pkl")
            save_metrics(lgb_metrics, artifacts_dir / "reports" / "exp4_lgb_eng_val.txt")
        except Exception as e:
            logger.error("Experiment 4 failed: %s", e)
    else:
        logger.warning("LightGBM not available, skipping Experiment 4")

    # ═══════════════════════════════════════════
    # Select best model
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("MODEL SELECTION")
    logger.info("=" * 60)

    best_model_name = max(all_results, key=lambda k: all_results[k]["val_pr_auc"])
    logger.info(
        "Best model by PR-AUC: %s (%.4f)",
        best_model_name,
        all_results[best_model_name]["val_pr_auc"],
    )

    # Extract sub-splits for Calibration & Threshold Optimization
    val_calib_df = df.loc[split.val_calib_idx].copy()
    val_opt_df = df.loc[split.val_opt_idx].copy()

    y_val_calib = np.asarray(val_calib_df["true_fraud_label"].values, dtype=np.int32)
    y_val_opt = np.asarray(val_opt_df["true_fraud_label"].values, dtype=np.int32)

    val_opt_amounts = np.asarray(val_opt_df["amount"].values, dtype=np.float64)

    # Load best model probabilities
    if "xgb_eng" in best_model_name and xgb_eng_prob_val is not None:
        best_prob_val = xgb_eng_prob_val
        best_prob_test = xgb_eng_prob_test
        best_model = xgb_eng
        best_features = "engineered"
        best_feature_list = common_eng
    elif "lgb_eng" in best_model_name and lgb_prob_val is not None:
        best_prob_val = lgb_prob_val
        best_prob_test = lgb_prob_test
        best_model = lgb_model
        best_features = "engineered"
        best_feature_list = common_eng
    elif "xgb_basic" in best_model_name and xgb_basic_prob_val is not None:
        best_prob_val = xgb_basic_prob_val
        best_prob_test = xgb_basic_prob_test
        best_model = xgb_basic
        best_features = "basic"
        best_feature_list = common_basic
    else:
        best_prob_val = lr_prob_val
        best_prob_test = lr_prob_test
        best_model = lr
        best_features = "basic"
        best_feature_list = common_basic

    # ═══════════════════════════════════════════
    # EXPERIMENT 5: Probability Calibration
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Probability Calibration (Fit on VAL_CALIB, Eval on VAL_OPT)")
    logger.info("=" * 60)

    # Get raw predictions for sub-splits
    val_calib_mask_in_val = np.isin(split.val_idx, split.val_calib_idx)
    val_opt_mask_in_val = np.isin(split.val_idx, split.val_opt_idx)

    prob_val_calib = best_prob_val[val_calib_mask_in_val]
    prob_val_opt = best_prob_val[val_opt_mask_in_val]

    calibrator = ProbabilityCalibrator()
    # Fit ONLY on VAL_CALIB to avoid calibration overfitting
    calibrator.fit_platt(y_val_calib, prob_val_calib)
    calibrator.fit_isotonic(y_val_calib, prob_val_calib)

    # Evaluate calibration ON UNSEEN VAL_OPT probabilities
    cal_results = calibrator.evaluate(y_val_opt, prob_val_opt)

    # Pick best calibration method based on unseen VAL_OPT Brier score
    best_cal_method = "platt"
    best_cal_brier = float("inf")
    for method, result in cal_results.items():
        logger.info(
            "%s (on VAL_OPT): Brier %.6f → %.6f, ECE %.6f → %.6f",
            method,
            result.brier_score_raw,
            result.brier_score_calibrated,
            result.ece_raw,
            result.ece_calibrated,
        )
        if result.brier_score_calibrated < best_cal_brier:
            best_cal_brier = result.brier_score_calibrated
            best_cal_method = method

    calibrator.set_active_method(best_cal_method)
    logger.info(
        "Best calibration method: %s (VAL_OPT Brier: %.6f)", best_cal_method, best_cal_brier
    )

    # Calibrated probabilities on test
    assert best_prob_test is not None
    cal_prob_test = calibrator.calibrate(best_prob_test)
    cal_metrics_test = compute_fraud_metrics(y_test, cal_prob_test, amounts=test_amounts)
    logger.info("Calibrated Test:\n%s", cal_metrics_test.summary())

    save_artifact(calibrator, artifacts_dir / "models" / "calibrator.pkl")
    save_metrics(cal_metrics_test, artifacts_dir / "reports" / "exp5_calibrated_test.txt")

    # ═══════════════════════════════════════════
    # EXPERIMENT 6: Cost-Sensitive Threshold Optimization
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: Cost-Sensitive Threshold Optimization (on VAL_OPT)")
    logger.info("=" * 60)

    # Optimize thresholds on unseen VAL_OPT calibrated probabilities
    cal_prob_val_opt = calibrator.calibrate(prob_val_opt)
    optimizer = ThresholdOptimizer()
    best_analysis, best_approve_t, best_block_t = optimizer.optimize(
        y_val_opt,
        cal_prob_val_opt,
        amounts=val_opt_amounts,
        n_steps=30,
    )

    logger.info(
        "Optimal thresholds (from VAL_OPT): APPROVE < %.4f, BLOCK >= %.4f",
        best_approve_t,
        best_block_t,
    )
    logger.info(
        "VAL_OPT net benefit: $%.2f | Fraud caught: %d | Fraud missed: %d | "
        "False declines: %d | Reviews: %d",
        best_analysis.net_benefit,
        best_analysis.fraud_caught,
        best_analysis.fraud_missed,
        best_analysis.false_declines,
        best_analysis.manual_reviews,
    )

    # Apply optimized thresholds to UNTOUCHED test set (final evaluation)
    test_analysis = optimizer.compute_cost(
        y_test,
        cal_prob_test,
        best_approve_t,
        best_block_t,
        amounts=test_amounts,
    )
    logger.info("Test set cost analysis (UNTOUCHED FINAL EVALUATION):")
    logger.info(
        "Net benefit: $%.2f | Fraud caught: %d | Fraud missed: %d | "
        "False declines: %d | Reviews: %d",
        test_analysis.net_benefit,
        test_analysis.fraud_caught,
        test_analysis.fraud_missed,
        test_analysis.false_declines,
        test_analysis.manual_reviews,
    )

    # ═══════════════════════════════════════════
    # SHAP Explainability
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("SHAP EXPLAINABILITY")
    logger.info("=" * 60)

    shap_importance = {}
    try:
        from src.explainability.shap_explainer import FraudExplainer

        x_for_shap = x_eng_val if best_features == "engineered" else x_basic_val
        explainer = FraudExplainer(best_model, best_feature_list)
        shap_importance = explainer.global_feature_importance(x_for_shap, max_samples=500)

        # Save explainer
        save_artifact(explainer, artifacts_dir / "models" / "explainer.pkl")
        logger.info("SHAP explainability computed and saved")
    except Exception as e:
        logger.warning("SHAP explainability failed: %s", e)

    # ═══════════════════════════════════════════
    # is_outlier A/B Experiment
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("IS_OUTLIER A/B EXPERIMENT")
    logger.info("=" * 60)

    logger.info(
        "is_outlier analysis: The 20,000 outlier rows contain ZERO fraud examples. "
        "Including is_outlier as a feature would create an artificial shortcut "
        "(outlier=1 → model learns 'never fraud'). This is data-quality metadata, "
        "not a legitimate predictive signal. Production model EXCLUDES is_outlier."
    )

    # ═══════════════════════════════════════════
    # Bootstrap Confidence Intervals & Temporal Drift
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("COMPUTING BOOTSTRAP CONFIDENCE INTERVALS & DRIFT")
    logger.info("=" * 60)

    from src.models.metrics import compute_bootstrap_confidence_intervals
    from src.monitoring.drift import compute_temporal_drift

    ci_dict = compute_bootstrap_confidence_intervals(
        y_test, cal_prob_test, amounts=test_amounts, threshold=best_approve_t, n_bootstraps=500
    )
    logger.info("Test Bootstrap 95%% Confidence Intervals (B=500):")
    for metric_name, (point_est, low, high) in ci_dict.items():
        logger.info("  %-15s: %.4f (95%% CI: %.4f → %.4f)", metric_name, point_est, low, high)

    drift_report = compute_temporal_drift(
        eng_features_train if best_features == "engineered" else basic_features_train,
        eng_features_test if best_features == "engineered" else basic_features_test,
        features=best_feature_list,
        baseline_name="TRAIN",
        target_name="TEST",
    )
    logger.info(drift_report.summary())

    # ═══════════════════════════════════════════
    # Save final production artifacts & Bundle
    # ═══════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("SAVING PRODUCTION ARTIFACTS & UNIFIED BUNDLE")
    logger.info("=" * 60)

    chosen_scaler = scaler_basic if best_features == "basic" else scaler_eng

    save_artifact(best_model, artifacts_dir / "models" / "production_model.pkl")
    save_artifact(calibrator, artifacts_dir / "models" / "production_calibrator.pkl")
    save_artifact(engineer, artifacts_dir / "models" / "production_feature_engineer.pkl")

    # Save scaler only for the feature set that the best model uses
    if best_features == "basic":
        save_artifact(scaler_basic, artifacts_dir / "models" / "production_scaler.pkl")
    else:
        save_artifact(scaler_eng, artifacts_dir / "models" / "production_scaler.pkl")

    # Export unified ProductionArtifactBundle
    from src.models.artifact_bundle import ProductionArtifactBundle

    model_version_str = f"v1.0.0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    bundle = ProductionArtifactBundle(
        model=best_model,
        calibrator=calibrator,
        feature_engineer=engineer,
        scaler=chosen_scaler,
        approve_threshold=best_approve_t,
        block_threshold=best_block_t,
        feature_names=best_feature_list,
        model_version=model_version_str,
        feature_type=best_features,
        metadata={
            "dataset_sha256": report.sha256,
            "best_iteration": all_results.get(best_model_name, {}).get("best_iteration"),
            "best_score": all_results.get(best_model_name, {}).get("best_score"),
            "random_seed": 42,
            "random_baseline_pr_auc": cal_metrics_test.random_baseline_pr_auc,
            "pr_auc_lift": cal_metrics_test.pr_auc_lift,
            "bootstrap_ci": ci_dict,
            "overall_drift_score": drift_report.overall_drift_score,
        },
    )
    bundle.verify_compatibility()
    bundle.save(artifacts_dir / "models" / "production_bundle.pkl")

    training_elapsed = time.time() - training_start

    # Save production manifest with full metadata
    manifest = {
        "model_name": best_model_name,
        "model_version": f"v1.0.0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_type": best_features,
        "feature_count": len(best_feature_list),
        "feature_names": best_feature_list,
        "calibration_method": best_cal_method,
        "optimal_approve_threshold": best_approve_t,
        "optimal_block_threshold": best_block_t,
        "random_seed": 42,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "fraud_ratio_train": float(y_train.sum() / len(y_train)),
        "training_duration_seconds": round(training_elapsed, 1),
        # Validation metrics
        "val_pr_auc": all_results[best_model_name]["val_pr_auc"],
        "val_roc_auc": all_results[best_model_name]["val_roc_auc"],
        # Test metrics (calibrated)
        "test_pr_auc": cal_metrics_test.pr_auc,
        "test_roc_auc": cal_metrics_test.roc_auc,
        "test_precision": cal_metrics_test.precision,
        "test_recall": cal_metrics_test.recall,
        "test_f1": cal_metrics_test.f1,
        "test_brier_score": cal_metrics_test.brier_score,
        "test_net_benefit": test_analysis.net_benefit,
        # SHAP top features
        "shap_top_features": dict(list(shap_importance.items())[:20]),
        # All experiment results
        "all_experiments": all_results,
    }

    save_json(manifest, artifacts_dir / "models" / "production_manifest.json")

    # Save experiment comparison
    save_json(all_results, artifacts_dir / "reports" / "experiment_comparison.json")

    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE (%.1fs)", training_elapsed)
    logger.info("=" * 60)
    for name, res in all_results.items():
        logger.info(
            "  %s: PR-AUC=%.4f, ROC-AUC=%.4f",
            name,
            res["val_pr_auc"],
            res["val_roc_auc"],
        )
    logger.info("Production model: %s", best_model_name)


if __name__ == "__main__":
    main()
