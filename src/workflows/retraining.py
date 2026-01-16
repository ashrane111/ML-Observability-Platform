"""Prefect flows for model retraining.

Provides:
- Manual retraining flow
- Drift-triggered retraining
- Scheduled retraining
"""

import logging
from datetime import datetime

# from pathlib import Path
from typing import Any, Dict, List, Optional

from prefect import flow, get_run_logger
from prefect.artifacts import create_markdown_artifact
from prefect.deployments import run_deployment  # noqa:F401

from .tasks import (
    check_drift,
    check_thresholds,
    evaluate_model,
    load_data,
    load_model,
    prepare_features,
    save_model,
    send_alert,
    train_model,
    validate_data,
)

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_CONFIG = {
    "data_dir": "data",
    "model_dir": "models",
    "reference_dir": "data/reference",
    "metrics_thresholds": {
        "fraud": {
            "test_accuracy": {"min": 0.85},
            "test_f1": {"min": 0.80},
            "test_auc": {"min": 0.85},
        },
        "price": {
            "test_r2": {"min": 0.70},
            "test_rmse": {"max": 50000},
        },
        "churn": {
            "test_accuracy": {"min": 0.80},
            "test_f1": {"min": 0.75},
        },
    },
    "drift_thresholds": {
        "psi_threshold": 0.2,
        "drift_share_threshold": 0.3,
    },
}


@flow(
    name="Model Retraining",
    description="Retrain a model with new data",
    retries=1,
    retry_delay_seconds=60,
)
def retraining_flow(
    model_type: str,
    data_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Main retraining flow for a single model.

    Args:
        model_type: Type of model (fraud, price, churn)
        data_path: Path to training data (uses default if None)
        config: Configuration overrides
        force: Force retraining even if metrics are good

    Returns:
        Dictionary with retraining results
    """
    run_logger = get_run_logger()
    run_logger.info(f"Starting retraining flow for {model_type}")

    # Merge config
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Determine data path
    if data_path is None:
        data_path = f"{cfg['data_dir']}/processed/{model_type}_data.parquet"

    results = {
        "model_type": model_type,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }

    try:
        # Step 1: Load and validate data
        run_logger.info("Loading training data...")
        df = load_data(data_path)
        validation = validate_data(df)

        if not validation["is_valid"]:
            run_logger.error(f"Data validation failed: {validation['issues']}")
            results["status"] = "failed"
            results["error"] = "Data validation failed"
            return results

        results["data_rows"] = validation["row_count"]

        # Step 2: Prepare features
        run_logger.info("Preparing features...")
        target_col = _get_target_column(model_type)
        X, y = prepare_features(df, target_column=target_col)

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Train model
        run_logger.info("Training model...")
        model = train_model(X_train, y_train, model_type=model_type)

        # Step 4: Evaluate model
        run_logger.info("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test, model_name=model_type)
        results["metrics"] = metrics

        # Step 5: Check thresholds
        thresholds = cfg["metrics_thresholds"].get(model_type, {})
        violations = check_thresholds(metrics, thresholds, model_type)

        if violations and not force:
            run_logger.warning(f"Model failed threshold checks: {violations}")
            results["status"] = "rejected"
            results["violations"] = violations

            send_alert(
                alert_type="model_rejected",
                message=f"Retrained {model_type} model rejected due to metric thresholds",
                severity="warning",
                metadata={"violations": violations, "metrics": metrics},
            )
            return results

        # Step 6: Save model
        run_logger.info("Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"

        save_path = save_model(
            model,
            cfg["model_dir"],
            model_name,
            metadata={
                "trained_at": datetime.now().isoformat(),
                "metrics": metrics,
                "data_rows": len(df),
                "data_path": data_path,
            },
        )

        results["model_path"] = str(save_path)
        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()

        # Create summary artifact
        create_markdown_artifact(
            key=f"{model_type}-retraining-summary",
            markdown=f"""
# Model Retraining Summary: {model_type}

**Status:** ✅ Success

## Training Data
- Rows: {len(df):,}
- Features: {X.shape[1]}

## Model Metrics
| Metric | Value |
|--------|-------|
"""
            + "\n".join([f"| {k} | {v:.4f} |" for k, v in metrics.items()])
            + f"""

## Model Path
`{save_path}`

**Timestamp:** {datetime.now().isoformat()}
""",
            description=f"Retraining summary for {model_type}",
        )

        run_logger.info(f"Retraining complete: {model_name}")
        return results

    except Exception as e:
        run_logger.error(f"Retraining failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

        send_alert(
            alert_type="retraining_failed",
            message=f"Model retraining failed for {model_type}: {e}",
            severity="critical",
            metadata={"model_type": model_type, "error": str(e)},
        )

        return results


@flow(
    name="Drift-Triggered Retraining",
    description="Check for drift and retrain if needed",
)
def drift_triggered_retraining_flow(
    model_type: str,
    current_data_path: Optional[str] = None,
    reference_data_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flow that checks for drift and triggers retraining if needed.

    Args:
        model_type: Type of model
        current_data_path: Path to current data
        reference_data_path: Path to reference data
        config: Configuration overrides

    Returns:
        Dictionary with flow results
    """
    run_logger = get_run_logger()
    run_logger.info(f"Starting drift-triggered retraining for {model_type}")

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Determine paths
    if current_data_path is None:
        current_data_path = f"{cfg['data_dir']}/processed/{model_type}_data.parquet"
    if reference_data_path is None:
        reference_data_path = f"{cfg['reference_dir']}/{model_type}_reference.parquet"

    results = {
        "model_type": model_type,
        "started_at": datetime.now().isoformat(),
        "drift_detected": False,
        "retrained": False,
    }

    try:
        # Load current and reference data
        run_logger.info("Loading data for drift check...")
        current_data = load_data(current_data_path)
        reference_data = load_data(reference_data_path)

        # Check for drift
        run_logger.info("Checking for drift...")
        drift_result = check_drift(
            current_data,
            reference_data,
            model_name=model_type,
            psi_threshold=cfg["drift_thresholds"]["psi_threshold"],
        )

        results["drift_result"] = drift_result
        results["drift_detected"] = drift_result.get("drift_detected", False)
        results["drift_share"] = drift_result.get("drift_share", 0)

        if results["drift_detected"]:
            run_logger.warning(f"Drift detected! Drift share: {results['drift_share']:.2%}")

            # Send drift alert
            send_alert(
                alert_type="drift_detected",
                message=f"Data drift detected for {model_type}. Triggering retraining.",
                severity="warning",
                metadata={
                    "drift_share": results["drift_share"],
                    "drifted_features": drift_result.get("drifted_features", []),
                },
            )

            # Trigger retraining
            run_logger.info("Triggering model retraining...")
            retraining_result = retraining_flow(
                model_type=model_type,
                data_path=current_data_path,
                config=config,
            )

            results["retrained"] = retraining_result.get("status") == "success"
            results["retraining_result"] = retraining_result

        else:
            run_logger.info("No drift detected. Skipping retraining.")

        results["status"] = "success"
        results["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        run_logger.error(f"Drift-triggered retraining failed: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    return results


@flow(
    name="Scheduled Retraining",
    description="Scheduled retraining for all models",
)
def scheduled_retraining_flow(
    models: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Flow for scheduled retraining of all models.

    Args:
        models: List of models to retrain (default: all)
        config: Configuration overrides

    Returns:
        Dictionary with results for all models
    """
    run_logger = get_run_logger()
    run_logger.info("Starting scheduled retraining flow")

    if models is None:
        models = ["fraud", "price", "churn"]

    results = {
        "started_at": datetime.now().isoformat(),
        "models": {},
    }

    for model_type in models:
        run_logger.info(f"Processing {model_type} model...")
        try:
            model_result = drift_triggered_retraining_flow(
                model_type=model_type,
                config=config,
            )
            results["models"][model_type] = model_result
        except Exception as e:
            run_logger.error(f"Failed to process {model_type}: {e}")
            results["models"][model_type] = {
                "status": "failed",
                "error": str(e),
            }

    # Generate summary
    success_count = sum(1 for r in results["models"].values() if r.get("status") == "success")
    retrained_count = sum(1 for r in results["models"].values() if r.get("retrained", False))

    results["summary"] = {
        "total_models": len(models),
        "successful": success_count,
        "retrained": retrained_count,
    }
    results["completed_at"] = datetime.now().isoformat()

    # Create summary artifact
    create_markdown_artifact(
        key="scheduled-retraining-summary",
        markdown=f"""
# Scheduled Retraining Summary

**Time:** {results['completed_at']}

## Results
| Model | Status | Drift Detected | Retrained |
|-------|--------|----------------|-----------|
"""
        + "\n".join(
            [
                f"| {m} | {r.get('status', 'unknown')} | {'Yes' if r.get('drift_detected') else 'No'} | {'Yes' if r.get('retrained') else 'No'} |"  # noqa:E501
                for m, r in results["models"].items()
            ]
        )
        + f"""

## Summary
- **Total Models:** {len(models)}
- **Successful:** {success_count}
- **Retrained:** {retrained_count}
""",
        description="Scheduled retraining summary",
    )

    run_logger.info(f"Scheduled retraining complete: {success_count}/{len(models)} successful")
    return results


def _get_target_column(model_type: str) -> str:
    """Get the target column name for a model type."""
    targets = {
        "fraud": "is_fraud",
        "price": "price",
        "churn": "churned",
    }
    return targets.get(model_type, "target")


# ============================================================================
# Deployment Configuration
# ============================================================================


def create_deployments():
    """Create Prefect deployments for the flows.

    Run this function to register deployments with Prefect server.
    """
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    # Scheduled retraining - daily at 2 AM
    scheduled_deployment = Deployment.build_from_flow(
        flow=scheduled_retraining_flow,
        name="daily-retraining",
        schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
        parameters={"models": ["fraud", "price", "churn"]},
        tags=["production", "scheduled"],
    )

    # On-demand retraining per model
    for model_type in ["fraud", "price", "churn"]:
        deployment = Deployment.build_from_flow(
            flow=retraining_flow,
            name=f"{model_type}-retraining",
            parameters={"model_type": model_type},
            tags=["production", "on-demand", model_type],
        )
        deployment.apply()

    # Drift monitoring - every 6 hours
    for model_type in ["fraud", "price", "churn"]:
        drift_deployment = Deployment.build_from_flow(
            flow=drift_triggered_retraining_flow,
            name=f"{model_type}-drift-check",
            schedule=CronSchedule(cron="0 */6 * * *"),  # Every 6 hours
            parameters={"model_type": model_type},
            tags=["production", "monitoring", model_type],
        )
        drift_deployment.apply()

    scheduled_deployment.apply()

    print("Deployments created successfully!")


if __name__ == "__main__":
    # Quick test
    result = retraining_flow(model_type="fraud")
    print(f"Result: {result}")
