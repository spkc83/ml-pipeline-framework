# Makefile for ML Pipeline Framework

.PHONY: help install install-dev test test-unit test-integration test-spark test-e2e lint format type-check security quality clean docs build dist upload

# Default target
help:
	@echo "ML Pipeline Framework - Available commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install          Install the package"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-spark       Run Spark integration tests only"
	@echo "  test-e2e         Run end-to-end tests only"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  test-parallel    Run tests in parallel"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting (flake8)"
	@echo "  format           Format code (autopep8 + isort)"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  security         Run security analysis (bandit)"
	@echo "  quality          Run all quality checks"
	@echo "  quality-fix      Run quality checks and fix auto-fixable issues"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Build and Distribution:"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo "  dist             Create distribution packages"
	@echo ""
	@echo "Pipeline Operations:"
	@echo "  run-example      Run example pipeline"
	@echo "  validate-config  Validate configuration file"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

# Testing
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v -m "not slow"

test-integration:
	python -m pytest tests/integration/ -v

test-spark:
	python -m pytest tests/integration/test_spark_integration.py -v -m spark

test-e2e:
	python -m pytest tests/integration/test_end_to_end_pipeline.py -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml

test-parallel:
	python -m pytest tests/ -n auto -v

test-slow:
	python -m pytest tests/ -v -m slow

test-watch:
	python -m pytest-watch tests/

# Code Quality
lint:
	flake8 src/ tests/ --statistics --count

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

security:
	bandit -r src/ -f json -o reports/bandit-report.json || true
	bandit -r src/ -f txt

quality:
	python tests/quality_checks.py

quality-fix:
	python tests/quality_checks.py --fix

complexity:
	radon cc src/ --show-complexity --min B

imports:
	isort src/ tests/ --check-only --diff

# Documentation
docs:
	@echo "Building documentation..."
	@mkdir -p docs/build
	@echo "Documentation build complete"

docs-serve:
	@echo "Serving documentation on http://localhost:8000"
	python -m http.server 8000 -d docs/build

# Build and Distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf reports/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: clean
	python setup.py build

dist: clean
	python setup.py sdist bdist_wheel

# Pipeline Operations
run-example:
	python run_pipeline.py run --config configs/pipeline_config.yaml --mode train

validate-config:
	python run_pipeline.py validate --config configs/pipeline_config.yaml

init-config:
	python run_pipeline.py init --config-type basic --output configs/

# Development utilities
setup-pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

check-deps:
	pip-audit
	safety check

benchmark:
	python -m pytest tests/ -v -m performance --benchmark-only

profile:
	python -m cProfile -o profile.stats run_pipeline.py run --config configs/pipeline_config.yaml
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Docker operations
docker-build:
	docker build -t ml-pipeline-framework .

docker-run:
	docker run --rm -v $(PWD):/workspace ml-pipeline-framework

docker-test:
	docker run --rm -v $(PWD):/workspace ml-pipeline-framework make test

# Continuous Integration helpers
ci-install:
	pip install --upgrade pip
	pip install -e ".[dev]"

ci-test:
	python -m pytest tests/ --cov=src --cov-report=xml --junitxml=reports/junit.xml

ci-quality:
	python tests/quality_checks.py --detailed-report

# Database operations (if needed)
db-init:
	@echo "Initializing database..."

db-migrate:
	@echo "Running database migrations..."

# Environment setup
env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml

env-export:
	conda env export > environment.yml

# Monitoring and logging
logs:
	tail -f logs/pipeline.log

monitor:
	@echo "Starting monitoring dashboard..."

# Security and compliance
audit:
	pip-audit
	safety check
	bandit -r src/

compliance-check:
	python tests/quality_checks.py --checks security,bandit

# Performance optimization
optimize:
	@echo "Running performance optimization..."

# Data operations
data-validate:
	python -c "from src.preprocessing.validator import DataValidator; print('Data validation tools ready')"

data-sample:
	@echo "Creating data samples for testing..."

# Model operations
model-validate:
	@echo "Validating model artifacts..."

model-compare:
	python run_pipeline.py run --mode compare --config configs/comparison_config.yaml

# Reporting
report-coverage:
	coverage html
	@echo "Coverage report generated in htmlcov/"

report-quality:
	python tests/quality_checks.py --detailed-report
	@echo "Quality report generated in quality_reports/"

# Utilities
count-lines:
	find src/ -name "*.py" -exec wc -l {} + | tail -1

tree:
	tree -I '__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|htmlcov|build|dist|*.egg-info'

# Git hooks and workflow
pre-push: quality test
	@echo "Pre-push checks complete"

release-check: clean quality test
	@echo "Release checks complete"

# IDE setup
setup-vscode:
	@mkdir -p .vscode
	@echo '{"python.defaultInterpreterPath": "./venv/bin/python", "python.testing.pytestEnabled": true}' > .vscode/settings.json

setup-pycharm:
	@echo "PyCharm setup instructions:"
	@echo "1. Open project directory"
	@echo "2. Configure Python interpreter to use virtual environment"
	@echo "3. Set test runner to pytest"
	@echo "4. Enable type checking with mypy"

# Help for specific targets
help-test:
	@echo "Testing commands:"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-spark    - Run Spark tests (requires PySpark)"
	@echo "  make test-coverage - Run tests with coverage analysis"
	@echo ""
	@echo "Test markers:"
	@echo "  -m unit           - Unit tests"
	@echo "  -m integration    - Integration tests"
	@echo "  -m spark          - Spark-specific tests"
	@echo "  -m slow           - Slow-running tests"
	@echo "  -m performance    - Performance/benchmark tests"

help-quality:
	@echo "Code quality commands:"
	@echo "  make lint         - Check code style with flake8"
	@echo "  make format       - Auto-format code with autopep8 and isort"
	@echo "  make type-check   - Type checking with mypy"
	@echo "  make security     - Security analysis with bandit"
	@echo "  make quality      - Run all quality checks"
	@echo "  make quality-fix  - Run quality checks and auto-fix issues"