# 🔍 ML Observability Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade **ML Monitoring and Observability Platform** that provides comprehensive model monitoring, drift detection, explainability, and automated alerting for machine learning systems in production.

![Architecture Overview](docs/images/architecture.png)

---

## 🎯 Features

### 📊 Model Monitoring
- **Drift Detection** - Track prediction and feature drift using PSI, KL Divergence, Wasserstein Distance
- **Performance Tracking** - Monitor accuracy, F1, AUC-ROC, RMSE degradation over time
- **Data Quality** - Detect missing values, outliers, schema violations

### 🔍 Explainability
- **SHAP Analysis** - Global and local feature importance with interactive visualizations
- **LIME Explanations** - Local interpretable model explanations
- **Counterfactuals** - "What-if" analysis for individual predictions

### 📈 Observability Stack
- **OpenTelemetry** - Unified traces, metrics, and logs
- **Prometheus + Grafana** - Metrics visualization and dashboards
- **Jaeger** - Distributed tracing
- **AlertManager** - Intelligent alerting with severity-based routing

### 🔄 Orchestration
- **Prefect** - Workflow orchestration for drift checks and retraining pipelines
- **Automated Retraining** - Trigger model retraining when drift exceeds thresholds

### 🎮 Interactive Dashboard
- **Streamlit UI** - Real-time monitoring dashboard
- **Drift Simulator** - Inject drift scenarios for demos and testing
- **Explainability Explorer** - Interactive SHAP/LIME visualizations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ML OBSERVABILITY PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   FastAPI    │    │   Streamlit  │    │   Prefect    │                   │
│  │   (API)      │    │  (Dashboard) │    │(Orchestrator)│                   │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                   │                            │
│         └───────────────────┴───────────────────┘                            │
│                             │                                                │
│  ┌──────────────────────────┴──────────────────────────┐                    │
│  │                   MONITORING LAYER                   │                    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │                    │
│  │  │ Evidently AI│  │ SHAP/LIME   │  │   MLflow    │  │                    │
│  │  │  (Drift)    │  │(Explainable)│  │  (Registry) │  │                    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │                    │
│  └─────────────────────────┬───────────────────────────┘                    │
│                            │                                                 │
│  ┌─────────────────────────┴───────────────────────────┐                    │
│  │                 OBSERVABILITY STACK                  │                    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │                    │
│  │  │OpenTelemetry│ │Prometheus│ │ Grafana  │ │Jaeger│ │                    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────┘ │                    │
│  └─────────────────────────────────────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **uv** (recommended) - Fast Python package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker** & **Docker Compose** (v2.0+)
- **Python 3.10** (uv will install this automatically if needed)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-observability-platform.git
cd ml-observability-platform
```

### 2. Set Up Environment

```bash
# Copy environment file
cp .env.example .env

# (Optional) Review and modify .env settings
```

### 3. Start All Services

```bash
# Start everything with Docker Compose
make up

# Or manually:
docker compose up -d
```

### 4. Access the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | - |
| **Dashboard** | http://localhost:8501 | - |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | - |
| **Jaeger** | http://localhost:16686 | - |
| **MLflow** | http://localhost:5000 | - |
| **Prefect** | http://localhost:4200 | - |

### 5. Run Demo

```bash
# Train models and generate sample data
make train
make generate-data

# Run drift simulation demo
make demo
```

---

## 📖 Documentation

- [Setup Guide](docs/setup_guide.md) - Detailed installation instructions
- [Architecture](docs/architecture.md) - System design documentation
- [API Reference](docs/api_reference.md) - API endpoint documentation
- [Demo Guide](docs/demo_guide.md) - How to run demos for interviews

---

## 🛠️ Development

### Local Development Setup

```bash
# Create virtual environment with uv (Python 3.10)
uv venv --python 3.10 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev

# Run API locally
make api

# Run dashboard locally
make dashboard
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
make coverage
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type-check
```

---

## 📊 Metrics Tracked

### Drift Metrics
| Metric | Description |
|--------|-------------|
| `ml_prediction_drift_psi` | Population Stability Index for predictions |
| `ml_prediction_drift_kl` | KL Divergence for prediction distribution |
| `ml_feature_drift_count` | Number of features with detected drift |
| `ml_concept_drift_score` | Change in feature-target relationships |

### Performance Metrics
| Metric | Description |
|--------|-------------|
| `ml_model_accuracy` | Current model accuracy |
| `ml_model_f1_score` | F1 score |
| `ml_model_auc_roc` | Area Under ROC Curve |
| `ml_model_rmse` | Root Mean Square Error (regression) |

### System Metrics
| Metric | Description |
|--------|-------------|
| `ml_prediction_latency_seconds` | Prediction response time |
| `ml_predictions_total` | Total predictions made |
| `ml_api_requests_total` | Total API requests |

---

## 🔔 Alerting

Pre-configured alerts for:

- **Drift Detection** - PSI > 0.2 (warning), > 0.3 (critical)
- **Performance Degradation** - Accuracy drop > 10%
- **Data Quality Issues** - Missing values > 5%, Outliers > 10%
- **System Health** - P99 latency > 200ms, Error rate > 5%

---

## 📁 Project Structure

```
ml-observability-platform/
├── src/                    # Source code
│   ├── api/               # FastAPI application
│   ├── data/              # Data generation & validation
│   ├── models/            # ML models
│   ├── monitoring/        # Drift detection & metrics
│   ├── explainability/    # SHAP/LIME integration
│   ├── workflows/         # Prefect workflows
│   └── utils/             # Utilities
├── dashboard/             # Streamlit dashboard
├── infrastructure/        # Docker & config files
│   ├── prometheus/        # Prometheus config
│   ├── grafana/          # Grafana dashboards
│   ├── alertmanager/     # Alert configuration
│   └── kubernetes/       # K8s manifests (ready)
├── scripts/              # Utility scripts
├── tests/                # Test suite
├── notebooks/            # Jupyter notebooks
└── docs/                 # Documentation
```

---

## 🎯 Models Included

| Model | Type | Use Case |
|-------|------|----------|
| **Fraud Detector** | Classification (XGBoost) | Detect fraudulent transactions |
| **Price Predictor** | Regression (LightGBM) | Predict product/house prices |
| **Churn Predictor** | Classification (Random Forest) | Predict customer churn |

---

## 🔮 Future Enhancements

- [ ] Kubernetes deployment with Helm charts
- [ ] AWS deployment with Terraform
- [ ] A/B testing framework
- [ ] Feature store integration
- [ ] Multi-model comparison dashboard
- [ ] Slack/Email alert integration

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ashutosh**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!
