#!/usr/bin/env python3
"""
Data Generation Script

Generates synthetic datasets for all three ML models:
- Fraud Detection
- Price Prediction
- Churn Prediction

Also generates reference datasets (without drift) and current datasets (with drift)
for drift detection demonstration.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --with-drift
    python scripts/generate_data.py --samples 20000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ruff: noqa: E402
from loguru import logger  # noqa: E402

from src.data.drift_injector import DriftConfig, DriftInjector, DriftType  # noqa: E402
from src.data.synthetic_generator import SyntheticDataGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for ML Observability Platform"
    )
    parser.add_argument(
        "--fraud-samples",
        type=int,
        default=10000,
        help="Number of fraud detection samples (default: 10000)",
    )
    parser.add_argument(
        "--price-samples",
        type=int,
        default=5000,
        help="Number of price prediction samples (default: 5000)",
    )
    parser.add_argument(
        "--churn-samples",
        type=int,
        default=8000,
        help="Number of churn prediction samples (default: 8000)",
    )
    parser.add_argument(
        "--with-drift",
        action="store_true",
        help="Also generate drifted datasets for testing",
    )
    parser.add_argument(
        "--drift-scenario",
        type=str,
        default="gradual_degradation",
        choices=[
            "gradual_degradation",
            "sudden_shift",
            "feature_corruption",
            "concept_change",
            "seasonal_variation",
            "catastrophic_failure",
        ],
        help="Drift scenario to apply (default: gradual_degradation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for generated data (default: data)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output file format (default: parquet)",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to generate all datasets."""
    args = parse_args()

    # Setup paths
    raw_dir = Path(args.output_dir) / "raw"
    reference_dir = Path(args.output_dir) / "reference"
    processed_dir = Path(args.output_dir) / "processed"

    for dir_path in [raw_dir, reference_dir, processed_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ML Observability Platform - Data Generation")
    logger.info("=" * 60)

    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)

    # Generate datasets
    datasets = {}

    # 1. Fraud Detection Dataset
    logger.info("\n📊 Generating Fraud Detection Dataset...")
    fraud_df = generator.generate_fraud_data(n_samples=args.fraud_samples)
    datasets["fraud"] = fraud_df
    logger.info(f"   Samples: {len(fraud_df)}")
    logger.info(f"   Fraud rate: {fraud_df['is_fraud'].mean()*100:.2f}%")
    logger.info(f"   Features: {len(generator.get_feature_columns('fraud'))}")

    # 2. Price Prediction Dataset
    logger.info("\n🏠 Generating Price Prediction Dataset...")
    price_df = generator.generate_price_data(n_samples=args.price_samples)
    datasets["price"] = price_df
    logger.info(f"   Samples: {len(price_df)}")
    logger.info(
        f"   Price range: ${price_df['price'].min():,.0f} - ${price_df['price'].max():,.0f}"
    )
    logger.info(f"   Mean price: ${price_df['price'].mean():,.0f}")

    # 3. Churn Prediction Dataset
    logger.info("\n👥 Generating Churn Prediction Dataset...")
    churn_df = generator.generate_churn_data(n_samples=args.churn_samples)
    datasets["churn"] = churn_df
    logger.info(f"   Samples: {len(churn_df)}")
    logger.info(f"   Churn rate: {churn_df['churned'].mean()*100:.2f}%")

    # Save raw datasets
    logger.info("\n💾 Saving raw datasets...")
    for name, df in datasets.items():
        if args.format == "parquet":
            filepath = raw_dir / f"{name}_data.parquet"
            df.to_parquet(filepath, index=False)
        else:
            filepath = raw_dir / f"{name}_data.csv"
            df.to_csv(filepath, index=False)
        logger.info(f"   Saved: {filepath}")

    # Save reference datasets (first 70% of data, no drift)
    logger.info("\n📁 Saving reference datasets (for drift detection baseline)...")
    for name, df in datasets.items():
        split_idx = int(len(df) * 0.7)
        reference_df = df.iloc[:split_idx].copy()

        if args.format == "parquet":
            filepath = reference_dir / f"{name}_reference.parquet"
            reference_df.to_parquet(filepath, index=False)
        else:
            filepath = reference_dir / f"{name}_reference.csv"
            reference_df.to_csv(filepath, index=False)
        logger.info(f"   Saved: {filepath} ({len(reference_df)} samples)")

    # Generate drifted datasets if requested
    if args.with_drift:
        logger.info(f"\n🌊 Generating drifted datasets (scenario: {args.drift_scenario})...")
        drift_injector = DriftInjector(seed=args.seed)

        for name, df in datasets.items():
            # Apply drift scenario
            drifted_df = drift_injector.create_drift_scenario(
                df.copy(), scenario=args.drift_scenario
            )

            # Save drifted dataset
            if args.format == "parquet":
                filepath = processed_dir / f"{name}_drifted_{args.drift_scenario}.parquet"
                drifted_df.to_parquet(filepath, index=False)
            else:
                filepath = processed_dir / f"{name}_drifted_{args.drift_scenario}.csv"
                drifted_df.to_csv(filepath, index=False)

            logger.info(f"   Saved: {filepath}")

        # Also create current datasets (last 30% with drift)
        logger.info("\n📊 Creating 'current' datasets for drift comparison...")
        for name, df in datasets.items():
            split_idx = int(len(df) * 0.7)
            current_df = df.iloc[split_idx:].copy()

            # Apply light drift to simulate production data
            config = DriftConfig(
                drift_type=DriftType.GRADUAL,
                magnitude=0.25,
            )
            current_df = drift_injector.inject_drift(current_df, config)

            if args.format == "parquet":
                filepath = processed_dir / f"{name}_current.parquet"
                current_df.to_parquet(filepath, index=False)
            else:
                filepath = processed_dir / f"{name}_current.csv"
                current_df.to_csv(filepath, index=False)

            logger.info(f"   Saved: {filepath} ({len(current_df)} samples)")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Data Generation Complete!")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {args.output_dir}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Seed: {args.seed}")

    logger.info("\n📁 Generated files:")
    for dir_path in [raw_dir, reference_dir, processed_dir]:
        files = list(dir_path.glob(f"*.{args.format}"))
        if files:
            logger.info(f"\n  {dir_path}/")
            for f in sorted(files):
                size_kb = f.stat().st_size / 1024
                logger.info(f"    - {f.name} ({size_kb:.1f} KB)")

    logger.info("\n🚀 Next steps:")
    logger.info("   1. Train models: make train")
    logger.info("   2. Start services: make up")
    logger.info("   3. Run drift demo: make demo")


if __name__ == "__main__":
    main()
