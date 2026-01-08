"""Prefect flows for continuous monitoring.

Provides:
- Data quality monitoring
- Drift monitoring
- Model health checks
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact

from .tasks import (
    load_data,
    validate_data,
    check_drift,
    compare_distributions,
    send_alert,
    check_thresholds,
    load_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)


# Default monitoring configuration
MONITORING_CONFIG = {
    "data_dir": "data",
    "model_dir": "models",
    "reference_dir": "data/reference",
    "data_quality_thresholds": {
        "max_missing_pct": 0.1,
        "max_duplicate_pct": 0.05,
    },
    "drift_thresholds": {
        "psi_threshold": 0.2,
        "drift_share_threshold": 0.3,
    },
    "performance_thresholds": {
        "fraud": {
            "accuracy": {"min": 0.85},
            "f1": {"min": 0.80},
        },
        "price": {
            "r2": {"min": 0.70},
        },
        "churn": {
            "accuracy": {"min": 0.80},
        },
    },
}


@flow(
    name="Data Quality Monitoring",
    description="Monitor data quality for incoming data",
)
def data_quality_flow(
    data_path: str,
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Monitor data quality for a dataset.

    Args:
        data_path: Path to data to check
        model_type: Type of model (for schema validation)
        config: Configuration overrides

    Returns:
        Data quality report
    """
    run_logger = get_run_logger()
    run_logger.info(f"Starting data quality check for {model_type}")

    cfg = {**MONITORING_CONFIG, **(config or {})}

    results = {
        "model_type": model_type,
        "data_path": data_path,
        "checked_at": datetime.now().isoformat(),
        "issues": [],
    }

    try:
        # Load data
        run_logger.info("Loading data...")
        df = load_data(data_path)

        # Validate data
        run_logger.info("Validating data quality...")
        validation = validate_data(
            df,
            max_missing_pct=cfg["data_quality_thresholds"]["max_missing_pct"],
        )

        results["row_count"] = validation["row_count"]
        results["column_count"] = validation["column_count"]
        results["is_valid"] = validation["is_valid"]
        results["issues"] = validation.get("issues", [])

        # Check for duplicates
        dup_pct = validation.get("duplicate_count", 0) / len(df) if len(df) > 0 else 0
        if dup_pct > cfg["data_quality_thresholds"]["max_duplicate_pct"]:
            results["issues"].append(f"High duplicate rate: {dup_pct:.2%}")

        # Calculate quality score
        quality_score = 1.0
        if results["issues"]:
            quality_score -= len(results["issues"]) * 0.1
        results["quality_score"] = max(0, quality_score)

        # Send alert if quality is low
        if results["quality_score"] < 0.8:
            send_alert(
                alert_type="data_quality_issue",
                message=f"Data quality issues detected for {model_type}",
                severity="warning" if results["quality_score"] >= 0.6 else "critical",
                metadata={
                    "quality_score": results["quality_score"],
                    "issues": results["issues"],
                },
            )

        # Create artifact
        create_markdown_artifact(
            key=f"{model_type}-data-quality",
            markdown=f"""
# Data Quality Report: {model_type}

**Quality Score:** {results['quality_score']:.0%} {'✅' if results['quality_score'] >= 0.8 else '⚠️'}

## Data Summary
- **Rows:** {results['row_count']:,}
- **Columns:** {results['column_count']}

## Issues
{chr(10).join([f"- {i}" for i in results['issues']]) if results['issues'] else "None detected"}

**Checked:** {results['checked_at']}
""",
            description=f"Data quality report for {model_type}",
        )

        results["status"] = "success"

    except Exception as e:
        run_logger.error(f"Data quality check failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results


@flow(
    name="Drift Monitoring",
    description="Monitor for data and concept drift",
)
def drift_monitoring_flow(
    model_type: str,
    current_data_path: Optional[str] = None,
    reference_data_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Monitor for drift in model data.

    Args:
        model_type: Type of model to monitor
        current_data_path: Path to current data
        reference_data_path: Path to reference data
        config: Configuration overrides

    Returns:
        Drift monitoring report
    """
    run_logger = get_run_logger()
    run_logger.info(f"Starting drift monitoring for {model_type}")

    cfg = {**MONITORING_CONFIG, **(config or {})}

    if current_data_path is None:
        current_data_path = f"{cfg['data_dir']}/processed/{model_type}_data.parquet"
    if reference_data_path is None:
        reference_data_path = f"{cfg['reference_dir']}/{model_type}_reference.parquet"

    results = {
        "model_type": model_type,
        "checked_at": datetime.now().isoformat(),
        "drift_detected": False,
        "features_with_drift": [],
    }

    try:
        # Load data
        run_logger.info("Loading current and reference data...")
        current_data = load_data(current_data_path)
        reference_data = load_data(reference_data_path)

        # Check drift
        run_logger.info("Checking for drift...")
        drift_result = check_drift(
            current_data,
            reference_data,
            model_name=model_type,
            psi_threshold=cfg["drift_thresholds"]["psi_threshold"],
        )

        results["drift_detected"] = drift_result.get("drift_detected", False)
        results["drift_share"] = drift_result.get("drift_share", 0)
        results["features_with_drift"] = drift_result.get("drifted_features", [])
        results["feature_scores"] = drift_result.get("feature_scores", {})

        # Compare distributions for additional insights
        run_logger.info("Comparing distributions...")
        distribution_comparison = compare_distributions(current_data, reference_data)
        results["distribution_comparison"] = distribution_comparison

        # Determine severity
        if results["drift_detected"]:
            severity = "warning"
            if results["drift_share"] > cfg["drift_thresholds"]["drift_share_threshold"]:
                severity = "critical"

            send_alert(
                alert_type="drift_detected",
                message=f"Data drift detected for {model_type}: {results['drift_share']:.1%} of features",
                severity=severity,
                metadata={
                    "drift_share": results["drift_share"],
                    "features": results["features_with_drift"],
                },
            )

        # Create artifact
        drift_status = "🚨 DRIFT DETECTED" if results["drift_detected"] else "✅ No Drift"
        create_markdown_artifact(
            key=f"{model_type}-drift-monitoring",
            markdown=f"""
# Drift Monitoring Report: {model_type}

**Status:** {drift_status}

## Summary
- **Drift Share:** {results.get('drift_share', 0):.1%}
- **Features with Drift:** {len(results['features_with_drift'])}

## Affected Features
{chr(10).join([f"- {f}" for f in results['features_with_drift'][:10]]) if results['features_with_drift'] else "None"}

**Checked:** {results['checked_at']}
""",
            description=f"Drift monitoring report for {model_type}",
        )

        results["status"] = "success"

    except Exception as e:
        run_logger.error(f"Drift monitoring failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results


@flow(
    name="Model Health Check",
    description="Check model health and performance",
)
def model_health_flow(
    model_type: str,
    test_data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check model health and performance.

    Args:
        model_type: Type of model to check
        test_data_path: Path to test data
        model_path: Path to model (uses default if None)
        config: Configuration overrides

    Returns:
        Model health report
    """
    run_logger = get_run_logger()
    run_logger.info(f"Starting model health check for {model_type}")

    cfg = {**MONITORING_CONFIG, **(config or {})}

    if test_data_path is None:
        test_data_path = f"{cfg['data_dir']}/processed/{model_type}_test.parquet"
    if model_path is None:
        model_path = f"{cfg['model_dir']}/{model_type}_detector"

    results = {
        "model_type": model_type,
        "model_path": model_path,
        "checked_at": datetime.now().isoformat(),
        "is_healthy": True,
        "issues": [],
    }

    try:
        # Load model
        run_logger.info("Loading model...")
        model = load_model(model_path, model_type)

        # Load test data
        run_logger.info("Loading test data...")
        test_data = load_data(test_data_path)

        # Prepare features
        target_col = _get_target_column(model_type)
        X_test = test_data.drop(columns=[target_col], errors="ignore")
        y_test = test_data.get(target_col)

        if y_test is not None:
            # Evaluate model
            run_logger.info("Evaluating model...")
            metrics = evaluate_model(model, X_test, y_test, model_name=model_type)
            results["metrics"] = metrics

            # Check thresholds
            thresholds = cfg["performance_thresholds"].get(model_type, {})
            violations = check_thresholds(metrics, thresholds, model_type)

            if violations:
                results["is_healthy"] = False
                results["issues"].extend([
                    f"{v['metric']}: {v['value']:.4f} (threshold: {v['threshold']})"
                    for v in violations
                ])

                send_alert(
                    alert_type="performance_degradation",
                    message=f"Model {model_type} performance below threshold",
                    severity="warning",
                    metadata={
                        "violations": violations,
                        "metrics": metrics,
                    },
                )

        # Check model can make predictions
        run_logger.info("Testing prediction capability...")
        try:
            sample = X_test.iloc[:5]
            from src.models.preprocessing import FeaturePreprocessor
            preprocessor = FeaturePreprocessor()
            sample_processed = preprocessor.fit_transform(sample)
            predictions = model.predict(sample_processed)
            results["prediction_test"] = "passed"
        except Exception as e:
            results["is_healthy"] = False
            results["issues"].append(f"Prediction test failed: {e}")
            results["prediction_test"] = "failed"

        # Create artifact
        health_status = "✅ Healthy" if results["is_healthy"] else "⚠️ Issues Detected"
        create_markdown_artifact(
            key=f"{model_type}-model-health",
            markdown=f"""
# Model Health Report: {model_type}

**Status:** {health_status}

## Metrics
| Metric | Value |
|--------|-------|
""" + "\n".join([f"| {k} | {v:.4f} |" for k, v in results.get('metrics', {}).items()]) + f"""

## Issues
{chr(10).join([f"- {i}" for i in results['issues']]) if results['issues'] else "None"}

## Prediction Test
- **Status:** {results.get('prediction_test', 'not run')}

**Checked:** {results['checked_at']}
""",
            description=f"Model health report for {model_type}",
        )

        results["status"] = "success"

    except Exception as e:
        run_logger.error(f"Model health check failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)
        results["is_healthy"] = False

    return results


@flow(
    name="Full Monitoring Suite",
    description="Run all monitoring flows for all models",
)
def full_monitoring_flow(
    models: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run complete monitoring suite for all models.

    Args:
        models: List of models to monitor (default: all)
        config: Configuration overrides

    Returns:
        Complete monitoring report
    """
    run_logger = get_run_logger()
    run_logger.info("Starting full monitoring suite")

    if models is None:
        models = ["fraud", "price", "churn"]

    results = {
        "started_at": datetime.now().isoformat(),
        "models": {},
    }

    for model_type in models:
        run_logger.info(f"Monitoring {model_type}...")

        model_results = {
            "data_quality": None,
            "drift": None,
            "health": None,
        }

        try:
            # Data quality
            model_results["data_quality"] = data_quality_flow(
                data_path=f"data/processed/{model_type}_data.parquet",
                model_type=model_type,
                config=config,
            )

            # Drift monitoring
            model_results["drift"] = drift_monitoring_flow(
                model_type=model_type,
                config=config,
            )

            # Model health
            model_results["health"] = model_health_flow(
                model_type=model_type,
                config=config,
            )

        except Exception as e:
            run_logger.error(f"Error monitoring {model_type}: {e}")
            model_results["error"] = str(e)

        results["models"][model_type] = model_results

    # Generate summary
    results["completed_at"] = datetime.now().isoformat()

    # Count issues
    total_issues = 0
    drift_detected = []
    unhealthy = []

    for model_type, mr in results["models"].items():
        if mr.get("drift", {}).get("drift_detected"):
            drift_detected.append(model_type)
        if mr.get("health", {}).get("is_healthy") == False:
            unhealthy.append(model_type)
        if mr.get("data_quality", {}).get("issues"):
            total_issues += len(mr["data_quality"]["issues"])

    results["summary"] = {
        "total_models": len(models),
        "models_with_drift": drift_detected,
        "unhealthy_models": unhealthy,
        "total_issues": total_issues,
    }

    # Overall status
    if unhealthy:
        overall_status = "critical"
    elif drift_detected:
        overall_status = "warning"
    else:
        overall_status = "healthy"

    results["overall_status"] = overall_status

    # Create summary artifact
    status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}[overall_status]
    create_markdown_artifact(
        key="monitoring-suite-summary",
        markdown=f"""
# Monitoring Suite Summary

**Overall Status:** {status_emoji} {overall_status.upper()}

## Model Status
| Model | Data Quality | Drift | Health |
|-------|--------------|-------|--------|
""" + "\n".join([
            f"| {m} | {'✅' if mr.get('data_quality', {}).get('quality_score', 0) >= 0.8 else '⚠️'} | {'🚨' if mr.get('drift', {}).get('drift_detected') else '✅'} | {'✅' if mr.get('health', {}).get('is_healthy') else '⚠️'} |"
            for m, mr in results["models"].items()
        ]) + f"""

## Summary
- **Models with Drift:** {', '.join(drift_detected) if drift_detected else 'None'}
- **Unhealthy Models:** {', '.join(unhealthy) if unhealthy else 'None'}
- **Total Issues:** {total_issues}

**Completed:** {results['completed_at']}
""",
        description="Full monitoring suite summary",
    )

    run_logger.info(f"Monitoring complete. Overall status: {overall_status}")
    return results


def _get_target_column(model_type: str) -> str:
    """Get the target column name for a model type."""
    targets = {
        "fraud": "is_fraud",
        "price": "price",
        "churn": "churned",
    }
    return targets.get(model_type, "target")
