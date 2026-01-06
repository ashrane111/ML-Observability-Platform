"""
Drift Detection Module

Uses Evidently AI to detect data and model drift.
Supports:
- Data drift (feature distribution changes)
- Target drift (label distribution changes)
- Prediction drift (model output changes)
- Data quality monitoring
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

# Handle Evidently 0.7.x API
try:
    from evidently import ColumnMapping, Report
    from evidently.presets import DataDriftPreset

    EVIDENTLY_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older API
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        try:
            from evidently import ColumnMapping
        except ImportError:
            from evidently.pipeline.column_mapping import ColumnMapping

        EVIDENTLY_AVAILABLE = True
    except ImportError:
        EVIDENTLY_AVAILABLE = False
        ColumnMapping = None  # type: ignore[misc, assignment]
        Report = None  # type: ignore[misc, assignment]
        DataDriftPreset = None  # type: ignore[misc, assignment]
        logger.warning("Evidently not available, using fallback drift detection")


class DriftStatus(str, Enum):
    """Status of drift detection."""

    NO_DRIFT = "no_drift"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection analysis."""

    timestamp: datetime
    dataset_name: str
    drift_status: DriftStatus

    # Overall metrics
    dataset_drift_detected: bool
    drift_share: float  # Fraction of features with drift
    number_of_drifted_features: int
    total_features: int

    # Per-feature results
    feature_drift_scores: dict[str, float]
    drifted_features: list[str]

    # Thresholds used
    psi_threshold: float
    drift_threshold: float

    # Raw report data
    report_json: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "drift_status": self.drift_status.value,
            "dataset_drift_detected": self.dataset_drift_detected,
            "drift_share": self.drift_share,
            "number_of_drifted_features": self.number_of_drifted_features,
            "total_features": self.total_features,
            "feature_drift_scores": self.feature_drift_scores,
            "drifted_features": self.drifted_features,
            "psi_threshold": self.psi_threshold,
            "drift_threshold": self.drift_threshold,
        }


@dataclass
class DataQualityResult:
    """Result of data quality analysis."""

    timestamp: datetime
    dataset_name: str

    # Missing values
    missing_values_share: float
    columns_with_missing: list[str]

    # Other quality metrics
    duplicate_rows: int
    total_rows: int

    # Per-column stats
    column_stats: dict[str, dict]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "missing_values_share": self.missing_values_share,
            "columns_with_missing": self.columns_with_missing,
            "duplicate_rows": self.duplicate_rows,
            "total_rows": self.total_rows,
            "column_stats": self.column_stats,
        }


class DriftDetector:
    """
    Detects data drift using Evidently AI.

    Compares a reference dataset (training data) against current data
    to identify distribution shifts that may affect model performance.
    """

    def __init__(
        self,
        psi_threshold_warning: float = 0.1,
        psi_threshold_critical: float = 0.2,
        drift_share_threshold: float = 0.3,
    ):
        """
        Initialize the drift detector.

        Args:
            psi_threshold_warning: PSI threshold for warning (default 0.1)
            psi_threshold_critical: PSI threshold for critical alert (default 0.2)
            drift_share_threshold: Fraction of features drifted to trigger alert
        """
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_critical = psi_threshold_critical
        self.drift_share_threshold = drift_share_threshold

        self._reference_data: Optional[pd.DataFrame] = None
        self._numerical_features: list[str] = []
        self._categorical_features: list[str] = []
        self._target_column: Optional[str] = None
        self._prediction_column: Optional[str] = None

        logger.info(
            f"DriftDetector initialized: "
            f"PSI warning={psi_threshold_warning}, critical={psi_threshold_critical}"
        )

    def set_reference_data(
        self,
        reference_data: pd.DataFrame,
        target_column: Optional[str] = None,
        prediction_column: Optional[str] = None,
        numerical_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
    ) -> None:
        """
        Set the reference dataset for drift comparison.

        Args:
            reference_data: Reference DataFrame (usually training data)
            target_column: Name of target column
            prediction_column: Name of prediction column
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self._reference_data = reference_data.copy()
        self._target_column = target_column
        self._prediction_column = prediction_column

        # Auto-detect feature types if not provided
        if numerical_features is None:
            numerical_features = reference_data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target and prediction columns
            for col in [target_column, prediction_column]:
                if col is not None and col in numerical_features:
                    numerical_features.remove(col)

        if categorical_features is None:
            categorical_features = reference_data.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

        self._numerical_features = numerical_features
        self._categorical_features = categorical_features

        logger.info(
            f"Reference data set: {len(reference_data)} samples, "
            f"{len(numerical_features)} numerical, {len(categorical_features)} categorical features"
        )

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> DriftResult:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current DataFrame to check for drift
            dataset_name: Name for logging/reporting

        Returns:
            DriftResult with drift analysis
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        logger.info(f"Detecting drift for {dataset_name}: {len(current_data)} samples")

        # Use statistical comparison for drift detection
        drift_results = self._compute_drift_statistics(current_data)

        # Determine drift status
        drift_status = self._determine_drift_status(
            drift_results["drift_share"],
            drift_results["feature_scores"],
        )

        result = DriftResult(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            drift_status=drift_status,
            dataset_drift_detected=drift_results["drift_detected"],
            drift_share=drift_results["drift_share"],
            number_of_drifted_features=drift_results["n_drifted"],
            total_features=drift_results["n_features"],
            feature_drift_scores=drift_results["feature_scores"],
            drifted_features=drift_results["drifted_features"],
            psi_threshold=self.psi_threshold_warning,
            drift_threshold=self.drift_share_threshold,
            report_json=None,
        )

        logger.info(
            f"Drift detection complete: status={drift_status.value}, "
            f"drift_share={drift_results['drift_share']:.2%}"
        )

        return result

    def _compute_drift_statistics(self, current_data: pd.DataFrame) -> dict:
        """Compute drift statistics using statistical tests."""
        feature_scores = {}
        drifted_features = []

        all_features = self._numerical_features + self._categorical_features

        for feature in all_features:
            if feature not in current_data.columns:
                continue
            if self._reference_data is None:
                continue
            if feature not in self._reference_data.columns:
                continue

            ref_col = self._reference_data[feature].dropna()
            cur_col = current_data[feature].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                continue

            # Compute PSI for numerical features
            if feature in self._numerical_features:
                psi = self._calculate_psi(ref_col, cur_col)
            else:
                # For categorical, use chi-squared based metric
                psi = self._calculate_categorical_drift(ref_col, cur_col)

            feature_scores[feature] = psi

            if psi > self.psi_threshold_warning:
                drifted_features.append(feature)

        n_features = len(feature_scores)
        n_drifted = len(drifted_features)
        drift_share = n_drifted / n_features if n_features > 0 else 0.0

        return {
            "drift_detected": drift_share > self.drift_share_threshold,
            "drift_share": drift_share,
            "n_drifted": n_drifted,
            "n_features": n_features,
            "feature_scores": feature_scores,
            "drifted_features": drifted_features,
        }

    def _calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the shift in distribution between reference and current data.
        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change
        """
        # Create bins based on reference data
        try:
            bins = pd.qcut(reference, q=n_bins, duplicates="drop").cat.categories
            n_bins_actual = len(bins)

            if n_bins_actual < 2:
                return 0.0

            # Bin the data
            ref_binned = pd.cut(reference, bins=bins, include_lowest=True)
            cur_binned = pd.cut(current, bins=bins, include_lowest=True)

            # Calculate proportions
            ref_props = ref_binned.value_counts(normalize=True, sort=False)
            cur_props = cur_binned.value_counts(normalize=True, sort=False)

            # Align indices
            all_bins = ref_props.index.union(cur_props.index)
            ref_props = ref_props.reindex(all_bins, fill_value=0.0001)
            cur_props = cur_props.reindex(all_bins, fill_value=0.0001)

            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))

            return max(0, float(psi))  # PSI should be non-negative

        except Exception as e:
            logger.debug(f"PSI calculation error: {e}")
            return 0.0

    def _calculate_categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series,
    ) -> float:
        """Calculate drift score for categorical features."""
        try:
            # Get value counts
            ref_counts = reference.value_counts(normalize=True)
            cur_counts = current.value_counts(normalize=True)

            # Align categories
            all_categories = ref_counts.index.union(cur_counts.index)
            ref_counts = ref_counts.reindex(all_categories, fill_value=0.0001)
            cur_counts = cur_counts.reindex(all_categories, fill_value=0.0001)

            # Calculate Jensen-Shannon divergence (symmetric KL divergence)
            m = 0.5 * (ref_counts + cur_counts)
            js_div = 0.5 * np.sum(ref_counts * np.log(ref_counts / m)) + 0.5 * np.sum(
                cur_counts * np.log(cur_counts / m)
            )

            return max(0, float(js_div))

        except Exception as e:
            logger.debug(f"Categorical drift calculation error: {e}")
            return 0.0

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        feature_name: str,
    ) -> dict[str, Any]:
        """
        Detect drift for a single feature.

        Args:
            current_data: Current DataFrame
            feature_name: Name of feature to check

        Returns:
            Dictionary with drift metrics for the feature
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        if feature_name not in current_data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in current data")

        ref_col = self._reference_data[feature_name].dropna()
        cur_col = current_data[feature_name].dropna()

        # Determine if numerical or categorical
        if feature_name in self._numerical_features:
            drift_score = self._calculate_psi(ref_col, cur_col)
            stattest_name = "psi"
        else:
            drift_score = self._calculate_categorical_drift(ref_col, cur_col)
            stattest_name = "js_divergence"

        return {
            "feature": feature_name,
            "drift_detected": drift_score > self.psi_threshold_warning,
            "drift_score": drift_score,
            "stattest_name": stattest_name,
            "stattest_threshold": self.psi_threshold_warning,
        }

    def check_data_quality(
        self,
        data: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> DataQualityResult:
        """
        Check data quality metrics.

        Args:
            data: DataFrame to check
            dataset_name: Name for logging/reporting

        Returns:
            DataQualityResult with quality metrics
        """
        logger.info(f"Checking data quality for {dataset_name}: {len(data)} samples")

        # Calculate missing values
        missing_by_col = data.isnull().sum()
        total_values = len(data) * len(data.columns)
        total_missing = missing_by_col.sum()
        missing_share = total_missing / total_values if total_values > 0 else 0.0

        columns_with_missing = missing_by_col[missing_by_col > 0].index.tolist()

        # Get duplicate count
        duplicate_rows = data.duplicated().sum()

        # Column statistics
        column_stats = {}
        for col in data.columns:
            stats: dict[str, Any] = {
                "missing_count": int(data[col].isnull().sum()),
                "missing_share": float(data[col].isnull().mean()),
            }
            if np.issubdtype(data[col].dtype, np.number):
                if not data[col].isnull().all():
                    stats.update(
                        {
                            "mean": float(data[col].mean()),
                            "std": float(data[col].std()),
                            "min": float(data[col].min()),
                            "max": float(data[col].max()),
                        }
                    )
                else:
                    stats.update({"mean": None, "std": None, "min": None, "max": None})
            column_stats[col] = stats

        result = DataQualityResult(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            missing_values_share=missing_share,
            columns_with_missing=columns_with_missing,
            duplicate_rows=int(duplicate_rows),
            total_rows=len(data),
            column_stats=column_stats,
        )

        logger.info(
            f"Data quality check complete: "
            f"missing={missing_share:.2%}, duplicates={duplicate_rows}"
        )

        return result

    def generate_report_html(
        self,
        current_data: pd.DataFrame,
        output_path: str,
    ) -> str:
        """
        Generate an HTML drift report.

        Args:
            current_data: Current DataFrame
            output_path: Path to save HTML report

        Returns:
            Path to generated report
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        if EVIDENTLY_AVAILABLE:
            try:
                # Create column mapping
                column_mapping = ColumnMapping(
                    target=self._target_column,
                    prediction=self._prediction_column,
                    numerical_features=self._numerical_features,
                    categorical_features=self._categorical_features,
                )

                # Create report with DataDriftPreset
                report = Report([DataDriftPreset()])

                report.run(
                    reference_data=self._reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping,
                )

                report.save_html(output_path)
                logger.info(f"Drift report saved to {output_path}")

            except Exception as e:
                logger.warning(f"Could not generate Evidently report: {e}")
                self._generate_simple_html_report(current_data, output_path)
        else:
            self._generate_simple_html_report(current_data, output_path)

        return output_path

    def _generate_simple_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: str,
    ) -> None:
        """Generate a simple HTML report without Evidently."""
        drift_result = self.detect_drift(current_data)

        html_content = f"""
        <html>
        <head><title>Drift Report</title></head>
        <body>
        <h1>Drift Detection Report</h1>
        <p>Generated: {drift_result.timestamp}</p>
        <h2>Summary</h2>
        <ul>
            <li>Status: {drift_result.drift_status.value}</li>
            <li>Drift Share: {drift_result.drift_share:.2%}</li>
            <li>Drifted Features: {drift_result.number_of_drifted_features}/ \
                {drift_result.total_features}</li>
        </ul>
        <h2>Feature Drift Scores</h2>
        <table border="1">
            <tr><th>Feature</th><th>Score</th><th>Drifted</th></tr>
            {"".join(
                f"<tr><td>{f}</td><td>{s:.4f}</td>"
                f"<td>{'Yes' if f in drift_result.drifted_features else 'No'}</td></tr>"
                for f, s in drift_result.feature_drift_scores.items()
            )}
        </table>
        </body>
        </html>
        """

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"Simple drift report saved to {output_path}")

    def _determine_drift_status(
        self,
        drift_share: float,
        feature_scores: dict[str, float],
    ) -> DriftStatus:
        """Determine overall drift status."""
        # Check if any feature has critical drift
        critical_features = [
            f for f, score in feature_scores.items() if score > self.psi_threshold_critical
        ]

        if critical_features or drift_share > self.drift_share_threshold:
            return DriftStatus.CRITICAL

        # Check for warning level
        warning_features = [
            f for f, score in feature_scores.items() if score > self.psi_threshold_warning
        ]

        if warning_features or drift_share > self.drift_share_threshold * 0.5:
            return DriftStatus.WARNING

        return DriftStatus.NO_DRIFT


def create_drift_detector(
    reference_data: pd.DataFrame,
    target_column: Optional[str] = None,
    **kwargs: Any,
) -> DriftDetector:
    """
    Convenience function to create a configured drift detector.

    Args:
        reference_data: Reference DataFrame
        target_column: Name of target column
        **kwargs: Additional arguments for DriftDetector

    Returns:
        Configured DriftDetector instance
    """
    detector = DriftDetector(**kwargs)
    detector.set_reference_data(reference_data, target_column=target_column)
    return detector
