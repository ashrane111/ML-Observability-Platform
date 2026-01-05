-- =============================================================================
-- PostgreSQL Initialization Script
-- =============================================================================
-- Creates required databases and schemas for ML Observability Platform
-- =============================================================================

-- Create additional databases
CREATE DATABASE mlflow;
CREATE DATABASE prefect;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlobs;
GRANT ALL PRIVILEGES ON DATABASE prefect TO mlobs;

-- Connect to main database and create schema
\c mlobs;

-- Create schema for predictions
CREATE SCHEMA IF NOT EXISTS predictions;

-- Create schema for monitoring
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions.prediction_logs (
    id SERIAL PRIMARY KEY,
    prediction_id UUID NOT NULL UNIQUE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    features JSONB NOT NULL,
    prediction JSONB NOT NULL,
    confidence FLOAT,
    latency_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for common queries
    INDEX idx_model_name (model_name),
    INDEX idx_created_at (created_at),
    INDEX idx_model_created (model_name, created_at)
);

-- Drift reports table
CREATE TABLE IF NOT EXISTS monitoring.drift_reports (
    id SERIAL PRIMARY KEY,
    report_id UUID NOT NULL UNIQUE,
    model_name VARCHAR(100) NOT NULL,
    report_type VARCHAR(50) NOT NULL,  -- 'prediction_drift', 'feature_drift', 'data_quality'
    metrics JSONB NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE,
    severity VARCHAR(20),  -- 'low', 'medium', 'high', 'critical'
    reference_start TIMESTAMP WITH TIME ZONE,
    reference_end TIMESTAMP WITH TIME ZONE,
    current_start TIMESTAMP WITH TIME ZONE,
    current_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_drift_model (model_name),
    INDEX idx_drift_type (report_type),
    INDEX idx_drift_detected (drift_detected),
    INDEX idx_drift_created (created_at)
);

-- Model performance table
CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    baseline_value FLOAT,
    sample_size INTEGER,
    evaluation_start TIMESTAMP WITH TIME ZONE,
    evaluation_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_perf_model (model_name),
    INDEX idx_perf_metric (metric_name),
    INDEX idx_perf_created (created_at)
);

-- Alerts history table
CREATE TABLE IF NOT EXISTS monitoring.alert_history (
    id SERIAL PRIMARY KEY,
    alert_id UUID NOT NULL,
    alert_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50),
    model_name VARCHAR(100),
    status VARCHAR(20) NOT NULL,  -- 'firing', 'resolved'
    summary TEXT,
    description TEXT,
    labels JSONB,
    annotations JSONB,
    fired_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_alert_name (alert_name),
    INDEX idx_alert_status (status),
    INDEX idx_alert_model (model_name),
    INDEX idx_alert_fired (fired_at)
);

-- Feature importance tracking table
CREATE TABLE IF NOT EXISTS monitoring.feature_importance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    feature_name VARCHAR(100) NOT NULL,
    importance_value FLOAT NOT NULL,
    importance_method VARCHAR(50),  -- 'shap', 'permutation', 'gain'
    rank INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_fi_model (model_name),
    INDEX idx_fi_feature (feature_name),
    INDEX idx_fi_created (created_at)
);

-- Create user for read-only access (e.g., for Grafana)
-- CREATE USER grafana_reader WITH PASSWORD 'readonly_password';
-- GRANT USAGE ON SCHEMA predictions, monitoring TO grafana_reader;
-- GRANT SELECT ON ALL TABLES IN SCHEMA predictions, monitoring TO grafana_reader;

-- Print success message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
END $$;
