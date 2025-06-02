# Makefile
.PHONY: help install test lint format train evaluate deploy clean docker-build docker-run

# Variables
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := sentiment-classifier
DOCKER_TAG := latest

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

install-prod: ## Install production dependencies only
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linting
	flake8 src/ tests/ scripts/
	black --check src/ tests/ scripts/
	isort --check-only src/ tests/ scripts/

format: ## Format code
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

validate-data: ## Validate data quality
	$(PYTHON) scripts/validate_data.py
	$(PYTHON) scripts/data_quality_checks.py

train: ## Train the model
	$(PYTHON) scripts/train_model.py

evaluate: ## Evaluate the model
	$(PYTHON) scripts/evaluate_model.py

monitor: ## Run model monitoring
	$(PYTHON) scripts/model_monitoring.py

save-metrics: ## Save model metrics
	$(PYTHON) scripts/save_metrics.py

pipeline: validate-data train evaluate save-metrics ## Run full ML pipeline

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-run: ## Run Docker container
	docker run -p 8000:8000 $(DOCKER_IMAGE):$(DOCKER_TAG)

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

start-api: ## Start API server locally
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

start-monitoring: ## Start monitoring stack
	docker-compose up -d prometheus grafana alertmanager

integration-test: ## Run integration tests
	$(PYTHON) tests/integration_tests.py --env=local

security-scan: ## Run security scan on dependencies
	pip-audit
	bandit -r src/

pre-commit: lint test ## Run pre-commit checks

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf reports/*.png reports/*.json
	rm -rf models/*.pkl

setup-dev: install ## Setup development environment
	pre-commit install
	mkdir -p data models reports monitoring notebooks
	echo "Development environment setup complete!"

setup-prod: install-prod ## Setup production environment
	mkdir -p models reports monitoring
	echo "Production environment setup complete!"

# Data preparation targets
prepare-data: ## Prepare training data
	$(PYTHON) scripts/prepare_data.py

download-models: ## Download pre-trained models (if applicable)
	echo "Downloading pre-trained models..."
	# Add commands to download models

# Deployment targets
deploy-staging: docker-build ## Deploy to staging
	echo "Deploying to staging..."
	# Add staging deployment commands

deploy-prod: docker-build ## Deploy to production
	echo "Deploying to production..."
	# Add production deployment commands

# Monitoring targets
view-metrics: ## Open Grafana dashboard
	open http://localhost:3000

view-prometheus: ## Open Prometheus UI
	open http://localhost:9090

# Notebook targets
jupyter: ## Start Jupyter notebook
	docker-compose up -d jupyter
	echo "Jupyter notebook available at http://localhost:8888"

# Backup targets
backup-models: ## Backup trained models
	tar -czf models_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz models/
	echo "Models backed up successfully"

backup-data: ## Backup training data
	tar -czf data_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/
	echo "Data backed up successfully"

# Documentation targets
docs: ## Generate documentation
	echo "Generating documentation..."
	# Add documentation generation commands

# Health checks
health-check: ## Check API health
	curl -f http://localhost:8000/health || exit 1

check-dependencies: ## Check for dependency updates
	pip list --outdated

# Full setup targets
setup-ci: ## Setup for CI environment
	$(MAKE) install
	$(MAKE) validate-data
	$(MAKE) test

setup-local: ## Setup for local development
	$(MAKE) setup-dev
	$(MAKE) docker-compose-up
	sleep 10
	$(MAKE) health-check
	echo "Local environment ready!"