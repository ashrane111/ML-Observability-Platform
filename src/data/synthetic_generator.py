"""
Synthetic Data Generator Module

Generates realistic synthetic datasets for:
- Fraud Detection (transactions)
- Price Prediction (properties)
- Churn Prediction (customers)

Each generator creates baseline data that can later have drift injected.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.schemas import (
    MerchantCategory,
    PaymentMethod,
    PropertyType,
    SubscriptionPlan,
    TransactionType,
)


class SyntheticDataGenerator:
    """
    Generates synthetic datasets for ML model training and monitoring.

    Each dataset is designed to have realistic patterns and correlations
    that can be used to train models and demonstrate drift detection.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the generator with a random seed for reproducibility.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.info(f"SyntheticDataGenerator initialized with seed={seed}")

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random seed."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    # =========================================================================
    # Fraud Detection Dataset
    # =========================================================================

    def generate_fraud_data(
        self,
        n_samples: int = 10000,
        fraud_rate: float = 0.02,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic transaction data for fraud detection.

        Creates realistic transaction patterns with:
        - Normal transactions with typical patterns
        - Fraudulent transactions with anomalous patterns
        - Temporal patterns (time of day, day of week)
        - Geographic patterns (distance from home)

        Args:
            n_samples: Number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions (default 2%)
            start_date: Start date for transactions
            end_date: End date for transactions

        Returns:
            DataFrame with transaction data
        """
        logger.info(f"Generating fraud dataset: n_samples={n_samples}, fraud_rate={fraud_rate}")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=90)
        if end_date is None:
            end_date = datetime.now()

        n_fraud = int(n_samples * fraud_rate)
        n_normal = n_samples - n_fraud

        # Generate normal transactions
        normal_data = self._generate_normal_transactions(n_normal, start_date, end_date)

        # Generate fraudulent transactions
        fraud_data = self._generate_fraud_transactions(n_fraud, start_date, end_date)

        # Combine and shuffle
        df = pd.concat([normal_data, fraud_data], ignore_index=True)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Add transaction IDs
        df["transaction_id"] = [f"txn_{uuid.uuid4().hex[:12]}" for _ in range(len(df))]

        logger.info(f"Generated {len(df)} transactions ({n_fraud} fraudulent)")
        return df

    def _generate_normal_transactions(
        self, n: int, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate normal (non-fraudulent) transactions."""

        # Time distribution - more transactions during business hours
        hours = self.rng.choice(24, size=n, p=self._get_hour_distribution(is_fraud=False))
        days = self.rng.integers(0, 7, size=n)

        # Generate timestamps
        date_range = (end_date - start_date).days
        random_days = self.rng.integers(0, max(1, date_range), size=n)
        timestamps = [
            start_date + timedelta(days=int(d), hours=int(h), minutes=int(self.rng.integers(0, 60)))
            for d, h in zip(random_days, hours)
        ]

        # Transaction amounts - log-normal distribution for normal spending
        amounts = self.rng.lognormal(mean=3.5, sigma=1.0, size=n)
        amounts = np.clip(amounts, 1, 5000)  # Realistic range

        # Average transaction amount per user (correlated with current amount)
        avg_amounts = amounts * self.rng.uniform(0.7, 1.3, size=n)

        # Transaction counts
        tx_count_24h = self.rng.poisson(2, size=n)
        tx_count_7d = tx_count_24h * 7 * self.rng.uniform(0.8, 1.2, size=n)

        # Distance from home - mostly close
        distance = self.rng.exponential(scale=5, size=n)
        distance = np.clip(distance, 0, 100)

        # Location (centered around a "home" location with some variance)
        base_lat, base_lon = 40.7128, -74.0060  # NYC as base
        lat = base_lat + self.rng.normal(0, 0.1, size=n)
        lon = base_lon + self.rng.normal(0, 0.1, size=n)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "amount": amounts,
                "transaction_type": self.rng.choice(
                    [t.value for t in TransactionType], size=n, p=[0.5, 0.15, 0.1, 0.15, 0.1]
                ),
                "merchant_category": self.rng.choice(
                    [m.value for m in MerchantCategory],
                    size=n,
                    p=[0.2, 0.15, 0.1, 0.15, 0.1, 0.05, 0.08, 0.05, 0.1, 0.02],
                ),
                "latitude": lat,
                "longitude": lon,
                "distance_from_home": distance,
                "hour_of_day": hours,
                "day_of_week": days,
                "is_weekend": np.isin(days, [5, 6]),
                "avg_transaction_amount": avg_amounts,
                "transaction_count_24h": tx_count_24h,
                "transaction_count_7d": tx_count_7d.astype(int),
                "is_online": self.rng.random(n) < 0.3,
                "is_foreign": self.rng.random(n) < 0.02,
                "is_fraud": False,
            }
        )

    def _generate_fraud_transactions(
        self, n: int, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Generate fraudulent transactions with anomalous patterns."""

        if n == 0:
            return pd.DataFrame()

        # Fraud happens more at odd hours
        hours = self.rng.choice(24, size=n, p=self._get_hour_distribution(is_fraud=True))
        days = self.rng.integers(0, 7, size=n)

        # Generate timestamps
        date_range = (end_date - start_date).days
        random_days = self.rng.integers(0, max(1, date_range), size=n)
        timestamps = [
            start_date + timedelta(days=int(d), hours=int(h), minutes=int(self.rng.integers(0, 60)))
            for d, h in zip(random_days, hours)
        ]

        # Fraud amounts - often higher than normal
        amounts = self.rng.lognormal(mean=5.0, sigma=1.5, size=n)
        amounts = np.clip(amounts, 50, 10000)

        # Avg amount is much lower (unusual spending pattern)
        avg_amounts = amounts * self.rng.uniform(0.1, 0.3, size=n)

        # Many transactions in short period (fraud pattern)
        tx_count_24h = self.rng.poisson(8, size=n)
        tx_count_7d = tx_count_24h * self.rng.uniform(1.5, 3, size=n)

        # Far from home
        distance = self.rng.exponential(scale=50, size=n)
        distance = np.clip(distance, 10, 500)

        # Different location
        base_lat, base_lon = 40.7128, -74.0060
        lat = base_lat + self.rng.normal(0, 0.5, size=n)
        lon = base_lon + self.rng.normal(0, 0.5, size=n)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "amount": amounts,
                "transaction_type": self.rng.choice(
                    [t.value for t in TransactionType],
                    size=n,
                    p=[0.3, 0.4, 0.2, 0.05, 0.05],  # More transfers/withdrawals
                ),
                "merchant_category": self.rng.choice(
                    [m.value for m in MerchantCategory],
                    size=n,
                    p=[0.05, 0.05, 0.05, 0.4, 0.1, 0.15, 0.02, 0.02, 0.1, 0.06],
                ),
                "latitude": lat,
                "longitude": lon,
                "distance_from_home": distance,
                "hour_of_day": hours,
                "day_of_week": days,
                "is_weekend": np.isin(days, [5, 6]),
                "avg_transaction_amount": avg_amounts,
                "transaction_count_24h": tx_count_24h,
                "transaction_count_7d": tx_count_7d.astype(int),
                "is_online": self.rng.random(n) < 0.7,  # More online fraud
                "is_foreign": self.rng.random(n) < 0.3,  # More foreign
                "is_fraud": True,
            }
        )

    def _get_hour_distribution(self, is_fraud: bool) -> np.ndarray:
        """Get probability distribution for transaction hours."""
        if is_fraud:
            # Fraud more likely at night
            probs = np.array(
                [
                    0.08,
                    0.08,
                    0.07,
                    0.06,
                    0.05,
                    0.04,  # 0-5am
                    0.03,
                    0.03,
                    0.03,
                    0.03,
                    0.03,
                    0.03,  # 6-11am
                    0.03,
                    0.03,
                    0.03,
                    0.03,
                    0.03,
                    0.03,  # 12-5pm
                    0.03,
                    0.04,
                    0.05,
                    0.06,
                    0.07,
                    0.08,  # 6-11pm
                ]
            )
        else:
            # Normal transactions during business hours
            probs = np.array(
                [
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.01,
                    0.02,  # 0-5am
                    0.03,
                    0.05,
                    0.07,
                    0.08,
                    0.08,
                    0.08,  # 6-11am
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                    0.06,
                    0.06,  # 12-5pm
                    0.05,
                    0.04,
                    0.03,
                    0.02,
                    0.01,
                    0.01,  # 6-11pm
                ]
            )
        return probs / probs.sum()

    # =========================================================================
    # Price Prediction Dataset
    # =========================================================================

    def generate_price_data(
        self,
        n_samples: int = 5000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic property data for price prediction.

        Creates realistic property listings with:
        - Correlated features (size, bedrooms, price)
        - Location effects on price
        - Market conditions

        Args:
            n_samples: Number of properties to generate
            start_date: Start date for listings
            end_date: End date for listings

        Returns:
            DataFrame with property data
        """
        logger.info(f"Generating price dataset: n_samples={n_samples}")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        # Property types with different base prices
        property_types = self.rng.choice(
            [p.value for p in PropertyType], size=n_samples, p=[0.4, 0.3, 0.2, 0.1]
        )

        # Size based on property type
        base_sqft = {"house": 2200, "apartment": 1000, "condo": 1200, "townhouse": 1800}
        square_feet = np.array(
            [self.rng.normal(base_sqft[pt], base_sqft[pt] * 0.3) for pt in property_types]
        )
        square_feet = np.clip(square_feet, 400, 8000)

        # Bedrooms correlated with size
        bedrooms = np.round(square_feet / 500 + self.rng.normal(0, 0.5, n_samples))
        bedrooms = np.clip(bedrooms, 1, 10).astype(int)

        # Bathrooms correlated with bedrooms
        bathrooms = bedrooms * 0.6 + self.rng.normal(0, 0.3, n_samples)
        bathrooms = np.clip(np.round(bathrooms * 2) / 2, 1, 8)  # Round to 0.5

        # Year built
        year_built = self.rng.normal(1990, 20, n_samples).astype(int)
        year_built = np.clip(year_built, 1920, 2024)

        # Location features
        neighborhood_score = self.rng.beta(5, 2, n_samples) * 10
        school_rating = self.rng.beta(4, 2, n_samples) * 10
        crime_rate = self.rng.exponential(3, n_samples)
        crime_rate = np.clip(crime_rate, 0.5, 20)

        # Location coordinates
        base_lat, base_lon = 40.7128, -74.0060
        lat = base_lat + self.rng.normal(0, 0.2, n_samples)
        lon = base_lon + self.rng.normal(0, 0.2, n_samples)

        # Amenities
        has_garage = self.rng.random(n_samples) < 0.6
        has_pool = self.rng.random(n_samples) < 0.15
        has_garden = self.rng.random(n_samples) < 0.5
        renovated = self.rng.random(n_samples) < 0.25

        # Market features
        days_on_market = self.rng.exponential(30, n_samples).astype(int)
        days_on_market = np.clip(days_on_market, 1, 365)
        num_price_changes = self.rng.poisson(0.5, n_samples)

        # Generate timestamps
        date_range = (end_date - start_date).days
        random_days = self.rng.integers(0, max(1, date_range), size=n_samples)
        listing_dates = [start_date + timedelta(days=int(d)) for d in random_days]

        # Calculate price based on features
        base_price = 150  # $/sqft base

        price = (
            square_feet * base_price
            + bedrooms * 15000
            + bathrooms * 10000
            + (2024 - year_built) * (-500)  # Older = cheaper
            + neighborhood_score * 20000
            + school_rating * 15000
            - crime_rate * 5000
            + has_garage * 25000
            + has_pool * 40000
            + has_garden * 15000
            + renovated * 35000
            + self.rng.normal(0, 50000, n_samples)  # Random noise
        )
        price = np.clip(price, 50000, 3000000)

        df = pd.DataFrame(
            {
                "property_id": [f"prop_{uuid.uuid4().hex[:12]}" for _ in range(n_samples)],
                "listing_date": listing_dates,
                "property_type": property_types,
                "square_feet": square_feet,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "year_built": year_built,
                "latitude": lat,
                "longitude": lon,
                "neighborhood_score": neighborhood_score,
                "school_rating": school_rating,
                "crime_rate": crime_rate,
                "has_garage": has_garage,
                "has_pool": has_pool,
                "has_garden": has_garden,
                "renovated": renovated,
                "days_on_market": days_on_market,
                "num_price_changes": num_price_changes,
                "price": price,
            }
        )

        logger.info(
            f"Generated {len(df)} properties, "
            f"price range: ${price.min():,.0f} - ${price.max():,.0f}"
        )
        return df

    # =========================================================================
    # Churn Prediction Dataset
    # =========================================================================

    def generate_churn_data(
        self,
        n_samples: int = 8000,
        churn_rate: float = 0.15,
        start_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic customer data for churn prediction.

        Creates realistic customer profiles with:
        - Subscription patterns
        - Usage behavior
        - Support interactions
        - Churn indicators

        Args:
            n_samples: Number of customers to generate
            churn_rate: Proportion of churned customers
            start_date: Earliest signup date

        Returns:
            DataFrame with customer data
        """
        logger.info(f"Generating churn dataset: n_samples={n_samples}, churn_rate={churn_rate}")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=730)  # 2 years

        n_churned = int(n_samples * churn_rate)
        n_active = n_samples - n_churned

        # Generate active customers
        active_data = self._generate_active_customers(n_active, start_date)

        # Generate churned customers
        churned_data = self._generate_churned_customers(n_churned, start_date)

        # Combine and shuffle
        df = pd.concat([active_data, churned_data], ignore_index=True)
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Add customer IDs
        df["customer_id"] = [f"cust_{uuid.uuid4().hex[:12]}" for _ in range(len(df))]

        logger.info(f"Generated {len(df)} customers ({n_churned} churned)")
        return df

    def _generate_active_customers(self, n: int, start_date: datetime) -> pd.DataFrame:
        """Generate active (non-churned) customers."""

        # Longer tenure
        tenure = self.rng.exponential(18, n)
        tenure = np.clip(tenure, 1, 60).astype(int)

        # Signup dates
        signup_dates = [datetime.now() - timedelta(days=int(t * 30)) for t in tenure]

        # Higher engagement
        login_freq = self.rng.gamma(5, 1, n)
        feature_usage = self.rng.beta(6, 3, n) * 100
        last_activity = self.rng.exponential(3, n).astype(int)
        last_activity = np.clip(last_activity, 0, 30)

        # Lower support issues
        tickets = self.rng.poisson(1, n)
        complaints = self.rng.poisson(0.2, n)

        # Higher NPS
        nps = self.rng.normal(7.5, 1.5, n)
        nps = np.clip(nps, 0, 10)

        # Subscription distribution
        plans = self.rng.choice(
            [p.value for p in SubscriptionPlan],
            size=n,
            p=[0.1, 0.3, 0.45, 0.15],  # More premium for active
        )

        # Monthly charges based on plan
        base_charges = {"free": 0, "basic": 9.99, "premium": 29.99, "enterprise": 99.99}
        monthly = np.array([base_charges[p] * self.rng.uniform(0.9, 1.1) for p in plans])

        # More likely to have annual contracts
        contract_types = self.rng.choice(["month-to-month", "annual"], size=n, p=[0.3, 0.7])

        return pd.DataFrame(
            {
                "signup_date": signup_dates,
                "age": np.clip(self.rng.normal(38, 12, n), 18, 80).astype(int),
                "gender": self.rng.choice(
                    ["male", "female", "other"], size=n, p=[0.48, 0.48, 0.04]
                ),
                "location": self.rng.choice(
                    ["northeast", "southeast", "midwest", "southwest", "west"],
                    size=n,
                    p=[0.2, 0.2, 0.2, 0.15, 0.25],
                ),
                "subscription_plan": plans,
                "monthly_charges": monthly,
                "total_charges": monthly * tenure,
                "payment_method": self.rng.choice(
                    [p.value for p in PaymentMethod], size=n, p=[0.4, 0.3, 0.2, 0.1]
                ),
                "tenure_months": tenure,
                "login_frequency": login_freq,
                "feature_usage_score": feature_usage,
                "last_activity_days": last_activity,
                "support_tickets": tickets,
                "complaints": complaints,
                "email_opt_in": self.rng.random(n) < 0.7,
                "referrals": self.rng.poisson(1.5, n),
                "nps_score": nps,
                "contract_type": contract_types,
                "auto_renewal": self.rng.random(n) < 0.8,
                "churned": False,
            }
        )

    def _generate_churned_customers(self, n: int, start_date: datetime) -> pd.DataFrame:
        """Generate churned customers with at-risk patterns."""

        if n == 0:
            return pd.DataFrame()

        # Shorter tenure
        tenure = self.rng.exponential(8, n)
        tenure = np.clip(tenure, 1, 36).astype(int)

        # Signup dates
        signup_dates = [datetime.now() - timedelta(days=int(t * 30)) for t in tenure]

        # Lower engagement
        login_freq = self.rng.gamma(2, 0.5, n)
        feature_usage = self.rng.beta(2, 5, n) * 100
        last_activity = self.rng.exponential(15, n).astype(int)
        last_activity = np.clip(last_activity, 5, 60)

        # More support issues
        tickets = self.rng.poisson(4, n)
        complaints = self.rng.poisson(1.5, n)

        # Lower NPS
        nps = self.rng.normal(4.5, 2, n)
        nps = np.clip(nps, 0, 10)

        # More basic plans
        plans = self.rng.choice(
            [p.value for p in SubscriptionPlan], size=n, p=[0.25, 0.45, 0.25, 0.05]
        )

        base_charges = {"free": 0, "basic": 9.99, "premium": 29.99, "enterprise": 99.99}
        monthly = np.array([base_charges[p] * self.rng.uniform(0.9, 1.1) for p in plans])

        # More month-to-month
        contract_types = self.rng.choice(["month-to-month", "annual"], size=n, p=[0.7, 0.3])

        return pd.DataFrame(
            {
                "signup_date": signup_dates,
                "age": np.clip(self.rng.normal(35, 15, n), 18, 80).astype(int),
                "gender": self.rng.choice(
                    ["male", "female", "other"], size=n, p=[0.48, 0.48, 0.04]
                ),
                "location": self.rng.choice(
                    ["northeast", "southeast", "midwest", "southwest", "west"],
                    size=n,
                    p=[0.2, 0.2, 0.2, 0.15, 0.25],
                ),
                "subscription_plan": plans,
                "monthly_charges": monthly,
                "total_charges": monthly * tenure,
                "payment_method": self.rng.choice(
                    [p.value for p in PaymentMethod], size=n, p=[0.3, 0.3, 0.25, 0.15]
                ),
                "tenure_months": tenure,
                "login_frequency": login_freq,
                "feature_usage_score": feature_usage,
                "last_activity_days": last_activity,
                "support_tickets": tickets,
                "complaints": complaints,
                "email_opt_in": self.rng.random(n) < 0.3,
                "referrals": self.rng.poisson(0.3, n),
                "nps_score": nps,
                "contract_type": contract_types,
                "auto_renewal": self.rng.random(n) < 0.2,
                "churned": True,
            }
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def generate_all_datasets(
        self,
        fraud_samples: int = 10000,
        price_samples: int = 5000,
        churn_samples: int = 8000,
        save_path: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate all three datasets at once.

        Args:
            fraud_samples: Number of fraud detection samples
            price_samples: Number of price prediction samples
            churn_samples: Number of churn prediction samples
            save_path: Optional path to save datasets

        Returns:
            Dictionary with all three DataFrames
        """
        logger.info("Generating all datasets...")

        datasets = {
            "fraud": self.generate_fraud_data(n_samples=fraud_samples),
            "price": self.generate_price_data(n_samples=price_samples),
            "churn": self.generate_churn_data(n_samples=churn_samples),
        }

        if save_path:
            import os

            os.makedirs(save_path, exist_ok=True)
            for name, df in datasets.items():
                filepath = os.path.join(save_path, f"{name}_data.parquet")
                df.to_parquet(filepath, index=False)
                logger.info(f"Saved {name} dataset to {filepath}")

        return datasets

    def get_feature_columns(self, dataset_type: str) -> list[str]:
        """Get feature column names for a dataset type."""
        columns = {
            "fraud": [
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
            ],
            "price": [
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
            ],
            "churn": [
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
            ],
        }
        return columns.get(dataset_type, [])

    def get_target_column(self, dataset_type: str) -> str:
        """Get target column name for a dataset type."""
        targets = {"fraud": "is_fraud", "price": "price", "churn": "churned"}
        return targets.get(dataset_type, "")


# Convenience function
def generate_datasets(seed: int = 42, save_path: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """
    Generate all datasets with default parameters.

    Args:
        seed: Random seed
        save_path: Optional path to save datasets

    Returns:
        Dictionary with fraud, price, and churn DataFrames
    """
    generator = SyntheticDataGenerator(seed=seed)
    return generator.generate_all_datasets(save_path=save_path)
