#!/usr/bin/env python
"""
Traffic Simulation Script

Simulates realistic API traffic for demonstrating monitoring capabilities.

Usage:
    python scripts/simulate_traffic.py [--api-url URL] [--duration SECONDS]
"""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import requests  # noqa: E402
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

from loguru import logger  # noqa: E402


def generate_fraud_request(drift: bool = False) -> dict[str, Any]:
    """Generate a sample fraud detection request."""
    base_amount = random.uniform(10, 500)
    if drift:
        base_amount = random.uniform(500, 5000)

    return {
        "amount": base_amount,
        "transaction_type": random.choice(["purchase", "withdrawal", "transfer"]),
        "merchant_category": random.choice(["retail", "food", "travel"]),
        "latitude": random.uniform(25, 48),
        "longitude": random.uniform(-125, -70),
        "distance_from_home": random.uniform(0, 100) if not drift else random.uniform(100, 500),
        "hour_of_day": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "is_weekend": random.choice([True, False]),
        "avg_transaction_amount": random.uniform(50, 300),
        "transaction_count_24h": random.randint(0, 20),
        "transaction_count_7d": random.randint(0, 100),
        "is_online": random.choice([True, False]),
        "is_foreign": random.choice([True, False]) if not drift else True,
    }


def generate_price_request(drift: bool = False) -> dict[str, Any]:
    """Generate a sample price prediction request."""
    base_sqft = random.uniform(800, 3000)
    if drift:
        base_sqft = random.uniform(3000, 8000)

    return {
        "property_type": random.choice(["house", "condo", "townhouse"]),
        "square_feet": base_sqft,
        "bedrooms": random.randint(1, 6),
        "bathrooms": random.randint(1, 4),
        "year_built": random.randint(1950, 2023),
        "latitude": random.uniform(25, 48),
        "longitude": random.uniform(-125, -70),
        "neighborhood_score": random.uniform(3, 10),
        "school_rating": random.uniform(3, 10),
        "crime_rate": random.uniform(0, 10),
        "has_garage": random.choice([True, False]),
        "has_pool": random.choice([True, False]) if not drift else True,
        "has_garden": random.choice([True, False]),
        "renovated": random.choice([True, False]),
        "days_on_market": random.randint(0, 180),
        "num_price_changes": random.randint(0, 5),
    }


def generate_churn_request(drift: bool = False) -> dict[str, Any]:
    """Generate a sample churn prediction request."""
    base_tenure = random.randint(1, 72)
    if drift:
        base_tenure = random.randint(1, 6)

    return {
        "age": random.randint(18, 80),
        "gender": random.choice(["male", "female", "other"]),
        "location": random.choice(["urban", "suburban", "rural"]),
        "subscription_plan": random.choice(["basic", "standard", "premium"]),
        "monthly_charges": random.uniform(10, 200),
        "total_charges": random.uniform(100, 5000),
        "payment_method": random.choice(["credit_card", "debit_card", "bank_transfer"]),
        "tenure_months": base_tenure,
        "login_frequency": random.uniform(0, 30),
        "feature_usage_score": random.uniform(10, 100) if not drift else random.uniform(0, 30),
        "last_activity_days": random.randint(0, 90) if not drift else random.randint(30, 180),
        "support_tickets": random.randint(0, 10),
        "complaints": random.randint(0, 5) if not drift else random.randint(3, 10),
        "email_opt_in": random.choice([True, False]),
        "referrals": random.randint(0, 10),
        "nps_score": random.randint(0, 10) if not drift else random.randint(0, 4),
        "contract_type": random.choice(["monthly", "annual", "two_year"]),
        "auto_renewal": random.choice([True, False]),
    }


class APIClient:
    """Client for interacting with the ML Observability API."""

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def predict_fraud(self, data: dict) -> dict[str, Any] | None:
        """Make a fraud prediction."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/fraud", json=data, timeout=self.timeout
            )
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception as e:
            logger.debug(f"Fraud prediction failed: {e}")
            return None

    def predict_price(self, data: dict) -> dict[str, Any] | None:
        """Make a price prediction."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/price", json=data, timeout=self.timeout
            )
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception as e:
            logger.debug(f"Price prediction failed: {e}")
            return None

    def predict_churn(self, data: dict) -> dict[str, Any] | None:
        """Make a churn prediction."""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/churn", json=data, timeout=self.timeout
            )
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception as e:
            logger.debug(f"Churn prediction failed: {e}")
            return None

    def check_drift(self, model_type: str, data: list[dict]) -> dict[str, Any] | None:
        """Check for drift."""
        try:
            response = self.session.post(
                f"{self.base_url}/monitoring/drift/check",
                json={
                    "model_type": model_type,
                    "data": data,
                    "dataset_name": f"simulation_{datetime.now().strftime('%H%M%S')}",
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                result: dict[str, Any] = response.json()
                return result
            return None
        except Exception as e:
            logger.debug(f"Drift check failed: {e}")
            return None

    def get_alerts(self) -> list[Any]:
        """Get active alerts."""
        try:
            response = self.session.get(f"{self.base_url}/monitoring/alerts", timeout=self.timeout)
            if response.status_code == 200:
                result: list[Any] = response.json().get("alerts", [])
                return result
            return []
        except Exception:
            return []


class TrafficSimulator:
    """Simulates API traffic patterns."""

    def __init__(
        self,
        api_url: str,
        requests_per_second: float = 1.0,
        drift_probability: float = 0.0,
    ):
        self.client = APIClient(api_url)
        self.requests_per_second = requests_per_second
        self.drift_probability = drift_probability

        # Statistics
        self.stats: dict[str, int | float] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fraud_predictions": 0,
            "price_predictions": 0,
            "churn_predictions": 0,
            "drift_checks": 0,
            "frauds_detected": 0,
            "churns_detected": 0,
            "drift_detected": 0,
        }
        self.start_time: float = 0.0

    def should_inject_drift(self) -> bool:
        """Determine if drift should be injected."""
        return random.random() < self.drift_probability

    def make_request(self) -> dict[str, Any] | None:
        """Make a single API request."""
        drift = self.should_inject_drift()
        request_type = random.choices(
            ["fraud", "price", "churn"],
            weights=[0.5, 0.3, 0.2],
            k=1,
        )[0]

        result = None

        if request_type == "fraud":
            data = generate_fraud_request(drift=drift)
            result = self.client.predict_fraud(data)
            if result:
                self.stats["fraud_predictions"] = int(self.stats["fraud_predictions"]) + 1
                if result.get("is_fraud"):
                    self.stats["frauds_detected"] = int(self.stats["frauds_detected"]) + 1

        elif request_type == "price":
            data = generate_price_request(drift=drift)
            result = self.client.predict_price(data)
            if result:
                self.stats["price_predictions"] = int(self.stats["price_predictions"]) + 1

        elif request_type == "churn":
            data = generate_churn_request(drift=drift)
            result = self.client.predict_churn(data)
            if result:
                self.stats["churn_predictions"] = int(self.stats["churn_predictions"]) + 1
                if result.get("will_churn"):
                    self.stats["churns_detected"] = int(self.stats["churns_detected"]) + 1

        self.stats["total_requests"] = int(self.stats["total_requests"]) + 1
        if result:
            self.stats["successful_requests"] = int(self.stats["successful_requests"]) + 1
        else:
            self.stats["failed_requests"] = int(self.stats["failed_requests"]) + 1

        return result

    def run_drift_check(self) -> None:
        """Run drift check with batch data."""
        model_type = random.choice(["fraud", "price", "churn"])
        drift = self.should_inject_drift()

        if model_type == "fraud":
            data = [generate_fraud_request(drift=drift) for _ in range(50)]
        elif model_type == "price":
            data = [generate_price_request(drift=drift) for _ in range(50)]
        else:
            data = [generate_churn_request(drift=drift) for _ in range(50)]

        result = self.client.check_drift(model_type, data)
        if result:
            self.stats["drift_checks"] = int(self.stats["drift_checks"]) + 1
            if result.get("dataset_drift_detected"):
                self.stats["drift_detected"] = int(self.stats["drift_detected"]) + 1
                drift_share = result.get("drift_share", 0)
                logger.warning(f"Drift detected for {model_type}: {drift_share:.2%}")

    def print_stats(self) -> None:
        """Print current statistics."""
        elapsed = time.time() - self.start_time
        total = int(self.stats["total_requests"])
        rps = total / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print(" TRAFFIC SIMULATION STATISTICS")
        print("=" * 60)
        print(f"  Duration:              {elapsed:.1f}s")
        print(f"  Total Requests:        {self.stats['total_requests']}")
        print(f"  Successful:            {self.stats['successful_requests']}")
        print(f"  Failed:                {self.stats['failed_requests']}")
        print(f"  Requests/sec:          {rps:.2f}")
        print("-" * 60)
        print(f"  Fraud Predictions:     {self.stats['fraud_predictions']}")
        print(f"  Price Predictions:     {self.stats['price_predictions']}")
        print(f"  Churn Predictions:     {self.stats['churn_predictions']}")
        print("-" * 60)
        print(f"  Frauds Detected:       {self.stats['frauds_detected']}")
        print(f"  Churns Detected:       {self.stats['churns_detected']}")
        print(f"  Drift Checks:          {self.stats['drift_checks']}")
        print(f"  Drift Detected:        {self.stats['drift_detected']}")
        print("=" * 60 + "\n")

    def run(self, duration: int, workers: int = 4) -> None:
        """Run the traffic simulation."""
        if not self.client.health_check():
            logger.error("API is not available. Please start the API first.")
            logger.info("Run: make run-api-dev")
            return

        logger.info(f"Starting traffic simulation for {duration} seconds...")
        logger.info(f"Target rate: {self.requests_per_second} req/s")
        logger.info(f"Drift probability: {self.drift_probability:.0%}")

        self.start_time = time.time()
        end_time = self.start_time + duration

        interval = 1.0 / self.requests_per_second
        drift_check_interval = 30
        last_drift_check = 0.0

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []

                while time.time() < end_time:
                    current_time = time.time()

                    futures.append(executor.submit(self.make_request))

                    if current_time - last_drift_check > drift_check_interval:
                        executor.submit(self.run_drift_check)
                        last_drift_check = current_time

                    futures = [f for f in futures if not f.done()]

                    time.sleep(interval)

                    elapsed = current_time - self.start_time
                    if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                        remaining = duration - elapsed
                        total = self.stats["total_requests"]
                        logger.info(
                            f"Progress: {elapsed:.0f}s elapsed, "
                            f"{remaining:.0f}s remaining, {total} requests"
                        )

                for _ in as_completed(futures, timeout=10):
                    pass

        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")

        self.print_stats()

        alerts = self.client.get_alerts()
        if alerts:
            print(f"Active Alerts ({len(alerts)}):")
            for alert in alerts[:5]:
                print(f"  - [{alert['severity']}] {alert['title']}")


def main() -> int:
    """Run the traffic simulator."""
    parser = argparse.ArgumentParser(
        description="Simulate API traffic for ML Observability Platform"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Simulation duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Requests per second (default: 2.0)",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.3,
        help="Probability of injecting drift (0-1, default: 0.3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads (default: 4)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" ML OBSERVABILITY PLATFORM - TRAFFIC SIMULATOR")
    print("=" * 60)
    print(f"  API URL:      {args.api_url}")
    print(f"  Duration:     {args.duration}s")
    print(f"  Rate:         {args.rate} req/s")
    print(f"  Drift Prob:   {args.drift:.0%}")
    print("=" * 60 + "\n")

    simulator = TrafficSimulator(
        api_url=args.api_url,
        requests_per_second=args.rate,
        drift_probability=args.drift,
    )

    simulator.run(duration=args.duration, workers=args.workers)

    return 0


if __name__ == "__main__":
    sys.exit(main())
