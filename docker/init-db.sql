-- Database initialization script for ML Pipeline Framework
-- Creates necessary databases and tables for MLflow and application data

-- Create MLflow database
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mluser;

-- Connect to MLflow database and create tables
\connect mlflow;

-- Create schema for MLflow tracking
CREATE SCHEMA IF NOT EXISTS public;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create sample tables for demonstration
\connect mlpipeline;

-- Create schema for ML pipeline data
CREATE SCHEMA IF NOT EXISTS ml_data;

-- Sample customer data table
CREATE TABLE IF NOT EXISTS ml_data.customers (
    customer_id SERIAL PRIMARY KEY,
    age INTEGER,
    income DECIMAL(12,2),
    credit_score INTEGER,
    account_balance DECIMAL(12,2),
    tenure_months INTEGER,
    product_count INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    churn_flag BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample transaction data table
CREATE TABLE IF NOT EXISTS ml_data.transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES ml_data.customers(customer_id),
    transaction_date DATE,
    amount DECIMAL(12,2),
    transaction_type VARCHAR(50),
    merchant_category VARCHAR(100),
    is_fraud BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sample product data table
CREATE TABLE IF NOT EXISTS ml_data.products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    price DECIMAL(10,2),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model metadata table
CREATE TABLE IF NOT EXISTS ml_data.model_metadata (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    algorithm VARCHAR(100),
    framework VARCHAR(50),
    metrics JSONB,
    parameters JSONB,
    training_date TIMESTAMP,
    deployment_date TIMESTAMP,
    status VARCHAR(50) DEFAULT 'training',
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS ml_data.predictions (
    prediction_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_data.model_metadata(model_id),
    input_data JSONB,
    prediction JSONB,
    confidence_score DECIMAL(5,4),
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    batch_id UUID
);

-- Data quality metrics table
CREATE TABLE IF NOT EXISTS ml_data.data_quality_metrics (
    metric_id SERIAL PRIMARY KEY,
    table_name VARCHAR(255),
    column_name VARCHAR(255),
    metric_type VARCHAR(100),
    metric_value DECIMAL(15,6),
    threshold_min DECIMAL(15,6),
    threshold_max DECIMAL(15,6),
    status VARCHAR(20),
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

-- Experiment tracking table
CREATE TABLE IF NOT EXISTS ml_data.experiments (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(255) NOT NULL,
    description TEXT,
    tags JSONB,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

-- Model performance monitoring table
CREATE TABLE IF NOT EXISTS ml_data.model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_data.model_metadata(model_id),
    metric_name VARCHAR(100),
    metric_value DECIMAL(15,6),
    data_slice VARCHAR(255),
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    period_start TIMESTAMP,
    period_end TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customers_churn ON ml_data.customers(churn_flag);
CREATE INDEX IF NOT EXISTS idx_customers_active ON ml_data.customers(is_active);
CREATE INDEX IF NOT EXISTS idx_transactions_customer ON ml_data.transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON ml_data.transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_fraud ON ml_data.transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON ml_data.predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON ml_data.predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_batch ON ml_data.predictions(batch_id);
CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON ml_data.model_metadata(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metadata_status ON ml_data.model_metadata(status);
CREATE INDEX IF NOT EXISTS idx_performance_model ON ml_data.model_performance(model_id);
CREATE INDEX IF NOT EXISTS idx_performance_date ON ml_data.model_performance(measurement_date);

-- Insert sample data for testing
INSERT INTO ml_data.customers (age, income, credit_score, account_balance, tenure_months, product_count, churn_flag)
SELECT 
    20 + (random() * 60)::integer as age,
    30000 + (random() * 120000)::decimal(12,2) as income,
    300 + (random() * 550)::integer as credit_score,
    (random() * 50000)::decimal(12,2) as account_balance,
    1 + (random() * 120)::integer as tenure_months,
    1 + (random() * 8)::integer as product_count,
    (random() < 0.15)::boolean as churn_flag
FROM generate_series(1, 1000);

-- Insert sample products
INSERT INTO ml_data.products (product_name, category, price) VALUES
    ('Checking Account', 'Banking', 0.00),
    ('Savings Account', 'Banking', 0.00),
    ('Credit Card', 'Credit', 0.00),
    ('Personal Loan', 'Lending', 0.00),
    ('Mortgage', 'Lending', 0.00),
    ('Investment Account', 'Investment', 0.00),
    ('Insurance Policy', 'Insurance', 120.00),
    ('Premium Banking', 'Banking', 25.00);

-- Insert sample transactions
INSERT INTO ml_data.transactions (customer_id, transaction_date, amount, transaction_type, merchant_category, is_fraud)
SELECT 
    (random() * 1000 + 1)::integer as customer_id,
    CURRENT_DATE - (random() * 365)::integer as transaction_date,
    (random() * 2000 + 10)::decimal(12,2) as amount,
    CASE (random() * 4)::integer
        WHEN 0 THEN 'purchase'
        WHEN 1 THEN 'withdrawal'
        WHEN 2 THEN 'deposit'
        ELSE 'transfer'
    END as transaction_type,
    CASE (random() * 6)::integer
        WHEN 0 THEN 'grocery'
        WHEN 1 THEN 'gas'
        WHEN 2 THEN 'restaurant'
        WHEN 3 THEN 'retail'
        WHEN 4 THEN 'online'
        ELSE 'other'
    END as merchant_category,
    (random() < 0.02)::boolean as is_fraud
FROM generate_series(1, 5000);

-- Create views for common queries
CREATE OR REPLACE VIEW ml_data.customer_summary AS
SELECT 
    c.customer_id,
    c.age,
    c.income,
    c.credit_score,
    c.account_balance,
    c.tenure_months,
    c.product_count,
    c.churn_flag,
    COUNT(t.transaction_id) as transaction_count,
    AVG(t.amount) as avg_transaction_amount,
    SUM(t.amount) as total_transaction_amount,
    COUNT(CASE WHEN t.is_fraud THEN 1 END) as fraud_count
FROM ml_data.customers c
LEFT JOIN ml_data.transactions t ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.age, c.income, c.credit_score, c.account_balance, 
         c.tenure_months, c.product_count, c.churn_flag;

-- Create a view for model performance summary
CREATE OR REPLACE VIEW ml_data.model_performance_summary AS
SELECT 
    mm.model_id,
    mm.model_name,
    mm.version,
    mm.algorithm,
    mm.framework,
    mm.status,
    mm.training_date,
    mm.deployment_date,
    COUNT(DISTINCT mp.metric_name) as metric_count,
    COUNT(p.prediction_id) as prediction_count,
    MAX(p.prediction_date) as last_prediction_date
FROM ml_data.model_metadata mm
LEFT JOIN ml_data.model_performance mp ON mm.model_id = mp.model_id
LEFT JOIN ml_data.predictions p ON mm.model_id = p.model_id
GROUP BY mm.model_id, mm.model_name, mm.version, mm.algorithm, 
         mm.framework, mm.status, mm.training_date, mm.deployment_date;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_data TO mluser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_data TO mluser;
GRANT USAGE ON SCHEMA ml_data TO mluser;

-- Update statistics
ANALYZE;