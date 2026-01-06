"""
Preprocessing Module

Provides feature preprocessing pipelines for all models.
Handles categorical encoding, scaling, and feature engineering.
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class FeaturePreprocessor:
    """
    Preprocesses features for ML models.

    Handles:
    - Categorical encoding (one-hot or label)
    - Numerical scaling
    - Missing value imputation
    - Feature type detection
    """

    def __init__(
        self,
        numerical_strategy: str = "standard",  # 'standard', 'minmax', or 'none'
        categorical_strategy: str = "onehot",  # 'onehot' or 'label'
        handle_missing: bool = True,
    ):
        """
        Initialize the preprocessor.

        Args:
            numerical_strategy: Scaling strategy for numerical features
            categorical_strategy: Encoding strategy for categorical features
            handle_missing: Whether to impute missing values
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.handle_missing = handle_missing

        self.numerical_features: list[str] = []
        self.categorical_features: list[str] = []
        self.boolean_features: list[str] = []

        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.is_fitted: bool = False

        logger.debug(
            f"FeaturePreprocessor initialized: "
            f"numerical={numerical_strategy}, categorical={categorical_strategy}"
        )

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> "FeaturePreprocessor":
        """
        Fit the preprocessor on the data.

        Args:
            df: Input DataFrame
            target_col: Name of target column to exclude from preprocessing

        Returns:
            self for method chaining
        """
        logger.info(f"Fitting preprocessor on {len(df)} samples, {len(df.columns)} columns")

        # Identify feature types
        self._identify_feature_types(df, target_col)

        # Build preprocessing pipeline
        self._build_preprocessor()

        # Fit the preprocessor
        features_df = self._get_features_df(df, target_col)
        if self.preprocessor is not None:
            self.preprocessor.fit(features_df)

        self.is_fitted = True
        logger.info(
            f"Preprocessor fitted: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical, "
            f"{len(self.boolean_features)} boolean features"
        )
        return self

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.

        Args:
            df: Input DataFrame
            target_col: Name of target column to exclude

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        features_df = self._get_features_df(df, target_col)

        if self.preprocessor is not None:
            transformed = self.preprocessor.transform(features_df)

            # Get feature names after transformation
            feature_names = self._get_transformed_feature_names()

            # Convert to DataFrame
            result_df = pd.DataFrame(transformed, columns=feature_names, index=df.index)
        else:
            result_df = features_df.copy()

        logger.debug(f"Transformed data shape: {result_df.shape}")
        return result_df

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame
            target_col: Name of target column to exclude

        Returns:
            Transformed DataFrame
        """
        self.fit(df, target_col)
        return self.transform(df, target_col)

    def _identify_feature_types(self, df: pd.DataFrame, target_col: Optional[str]) -> None:
        """Identify numerical, categorical, and boolean features."""
        self.numerical_features = []
        self.categorical_features = []
        self.boolean_features = []

        for col in df.columns:
            if col == target_col:
                continue

            dtype = df[col].dtype

            if dtype == bool or (dtype == object and df[col].nunique() == 2):
                # Check if it's actually boolean-like
                unique_vals = set(df[col].dropna().unique())
                bool_like_values = {"True", "False", "true", "false"}
                int_like_values = {0, 1}
                if unique_vals <= bool_like_values or unique_vals <= int_like_values:
                    self.boolean_features.append(col)
                else:
                    self.categorical_features.append(col)
            elif dtype in ["object", "category"] or str(dtype) == "string":
                self.categorical_features.append(col)
            elif np.issubdtype(dtype, np.number):
                self.numerical_features.append(col)
            else:
                # Default to categorical for unknown types
                self.categorical_features.append(col)

        logger.debug(f"Numerical features: {self.numerical_features}")
        logger.debug(f"Categorical features: {self.categorical_features}")
        logger.debug(f"Boolean features: {self.boolean_features}")

    def _build_preprocessor(self) -> None:
        """Build the sklearn preprocessing pipeline."""
        transformers = []

        # Numerical features pipeline
        if self.numerical_features:
            numerical_steps = []

            if self.handle_missing:
                numerical_steps.append(("imputer", SimpleImputer(strategy="median")))

            if self.numerical_strategy == "standard":
                numerical_steps.append(("scaler", StandardScaler()))
            elif self.numerical_strategy == "minmax":
                from sklearn.preprocessing import MinMaxScaler

                numerical_steps.append(("scaler", MinMaxScaler()))

            if numerical_steps:
                numerical_pipeline = Pipeline(steps=numerical_steps)
                transformers.append(("numerical", numerical_pipeline, self.numerical_features))

        # Categorical features pipeline
        if self.categorical_features:
            categorical_steps = []

            if self.handle_missing:
                categorical_steps.append(
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing"))
                )

            if self.categorical_strategy == "onehot":
                categorical_steps.append(
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                )
            elif self.categorical_strategy == "label":
                # For label encoding, we'll handle it separately
                pass

            if categorical_steps:
                categorical_pipeline = Pipeline(steps=categorical_steps)
                transformers.append(
                    ("categorical", categorical_pipeline, self.categorical_features)
                )

        # Boolean features - convert to int
        # Boolean features - convert to int first, then impute
        if self.boolean_features:
            from sklearn.preprocessing import FunctionTransformer

            boolean_pipeline = Pipeline(
                steps=[
                    ("to_int", FunctionTransformer(lambda x: x.astype(int))),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            )
            transformers.append(("boolean", boolean_pipeline, self.boolean_features))

        if transformers:
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",  # Drop any remaining columns
                verbose_feature_names_out=False,
            )
        else:
            self.preprocessor = None

    def _get_features_df(self, df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
        """Get DataFrame with only feature columns."""
        all_features = self.numerical_features + self.categorical_features + self.boolean_features
        available_features = [f for f in all_features if f in df.columns]
        return df[available_features].copy()

    def _get_transformed_feature_names(self) -> list[str]:
        """Get feature names after transformation."""
        if self.preprocessor is None:
            return self.numerical_features + self.categorical_features + self.boolean_features

        try:
            return list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback for older sklearn versions - build names manually
            feature_names = []

            for name, transformer, columns in self.preprocessor.transformers_:
                if name == "remainder":
                    continue
                if hasattr(transformer, "get_feature_names_out"):
                    try:
                        names = list(transformer.get_feature_names_out(columns))
                        feature_names.extend(names)
                    except Exception:
                        feature_names.extend(columns)
                else:
                    feature_names.extend(columns)

            return feature_names

    def get_feature_info(self) -> dict:
        """Get information about identified features."""
        return {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "boolean_features": self.boolean_features,
            "total_features": (
                len(self.numerical_features)
                + len(self.categorical_features)
                + len(self.boolean_features)
            ),
            "is_fitted": self.is_fitted,
        }


def prepare_fraud_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for fraud detection model.

    Args:
        df: Raw fraud data DataFrame

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    feature_cols = [
        "amount",
        "transaction_type",
        "merchant_category",
        "latitude",
        "longitude",
        "distance_from_home",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "avg_transaction_amount",
        "transaction_count_24h",
        "transaction_count_7d",
        "is_online",
        "is_foreign",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    y = df["is_fraud"].copy()

    return X, y


def prepare_price_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for price prediction model.

    Args:
        df: Raw price data DataFrame

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    feature_cols = [
        "property_type",
        "square_feet",
        "bedrooms",
        "bathrooms",
        "year_built",
        "latitude",
        "longitude",
        "neighborhood_score",
        "school_rating",
        "crime_rate",
        "has_garage",
        "has_pool",
        "has_garden",
        "renovated",
        "days_on_market",
        "num_price_changes",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    y = df["price"].copy()

    return X, y


def prepare_churn_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for churn prediction model.

    Args:
        df: Raw churn data DataFrame

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    feature_cols = [
        "age",
        "gender",
        "location",
        "subscription_plan",
        "monthly_charges",
        "total_charges",
        "payment_method",
        "tenure_months",
        "login_frequency",
        "feature_usage_score",
        "last_activity_days",
        "support_tickets",
        "complaints",
        "email_opt_in",
        "referrals",
        "nps_score",
        "contract_type",
        "auto_renewal",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    y = df["churned"].copy()

    return X, y
