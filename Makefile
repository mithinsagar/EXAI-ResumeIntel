# EXAI-ResumeIntel: Development Makefile
# Author: Mithin Sagar S
# GitHub: https://github.com/mithinsagar

.PHONY: help install install-dev clean lint format test test-cov \
        serve serve-ui serve-app train evaluate docker docker-build \
        docker-run download-data all

# Colours for pretty output
BLUE   := \033[0;34m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RESET  := \033[0m

help:
	@echo "$(BLUE)EXAI-ResumeIntel$(RESET) — Available targets"
	@echo ""
	@echo "$(GREEN)Setup$(RESET)"
	@echo "  install         Install runtime dependencies"
	@echo "  install-dev     Install runtime + development dependencies"
	@echo "  download-data   Download datasets and models from Hugging Face"
	@echo ""
	@echo "$(GREEN)Code quality$(RESET)"
	@echo "  lint            Run ruff + mypy"
	@echo "  format          Apply black + isort"
	@echo "  test            Run pytest test suite"
	@echo "  test-cov        Run pytest with coverage report"
	@echo ""
	@echo "$(GREEN)Run$(RESET)"
	@echo "  serve           Start the FastAPI backend"
	@echo "  serve-ui        Start the static HTML dashboard"
	@echo "  serve-app       Start the Streamlit application"
	@echo ""
	@echo "$(GREEN)Training$(RESET)"
	@echo "  train           Train the embedding engine"
	@echo "  evaluate        Run LinearSVC 5-fold cross-validation"
	@echo ""
	@echo "$(GREEN)Deployment$(RESET)"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-run      Run docker-compose stack"
	@echo ""
	@echo "$(GREEN)Housekeeping$(RESET)"
	@echo "  clean           Remove caches, build artifacts, __pycache__"

install:
	@echo "$(BLUE)==> Installing runtime dependencies$(RESET)"
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	@echo "$(BLUE)==> Installing development dependencies$(RESET)"
	pip install --upgrade pip
	pip install -r requirements-dev.txt

download-data:
	@echo "$(BLUE)==> Downloading datasets and models$(RESET)"
	bash scripts/download_data.sh

lint:
	@echo "$(BLUE)==> Running ruff$(RESET)"
	ruff check core/ xai/ api/ training/ tests/
	@echo "$(BLUE)==> Running mypy$(RESET)"
	mypy core/ xai/ api/ --ignore-missing-imports

format:
	@echo "$(BLUE)==> Formatting with black$(RESET)"
	black core/ xai/ api/ training/ tests/
	@echo "$(BLUE)==> Sorting imports with isort$(RESET)"
	isort core/ xai/ api/ training/ tests/

test:
	@echo "$(BLUE)==> Running pytest$(RESET)"
	pytest tests/ -v

test-cov:
	@echo "$(BLUE)==> Running pytest with coverage$(RESET)"
	pytest tests/ -v --cov=core --cov=xai --cov=api --cov-report=term --cov-report=html
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

serve:
	@echo "$(BLUE)==> Starting FastAPI server on http://localhost:8765$(RESET)"
	uvicorn api.server:app --host 0.0.0.0 --port 8765 --reload

serve-ui:
	@echo "$(BLUE)==> Starting static UI on http://localhost:8080$(RESET)"
	cd ui && python -m http.server 8080

serve-app:
	@echo "$(BLUE)==> Starting Streamlit app on http://localhost:8501$(RESET)"
	streamlit run app/main.py

train:
	@echo "$(BLUE)==> Training embedding engine$(RESET)"
	python -m training.train

evaluate:
	@echo "$(BLUE)==> Running LinearSVC 5-fold cross-validation$(RESET)"
	python -m training.evaluate

docker-build:
	@echo "$(BLUE)==> Building Docker images$(RESET)"
	docker-compose -f deployment/docker-compose.yml build

docker-run:
	@echo "$(BLUE)==> Starting docker-compose stack$(RESET)"
	docker-compose -f deployment/docker-compose.yml up -d
	@echo "$(GREEN)Services running:$(RESET)"
	@echo "  API:       http://localhost:8765"
	@echo "  UI:        http://localhost:8080"
	@echo "  Streamlit: http://localhost:8501"

clean:
	@echo "$(BLUE)==> Cleaning caches and build artifacts$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist htmlcov .coverage
	@echo "$(GREEN)Clean complete.$(RESET)"

all: format lint test
