# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentiment-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A complete ML pipeline for sentiment classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sentiment-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.3",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=6.5.4",
            "matplotlib>=3.7.1",
            "seaborn>=0.12.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-train=scripts.train_model:main",
            "sentiment-evaluate=scripts.evaluate_model:main",
            "sentiment-monitor=scripts.model_monitoring:main",
        ],
    },
)


# scripts/setup_environment.py
import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'models',
        'reports',
        'monitoring',
        'notebooks',
        'logs',
        'src/data',
        'src/models',
        'src/utils',
        'tests',
        'docs',
        'k8s/staging',
        'k8s/production'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/utils/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        logger.info(f"Created file: {init_file}")

def setup_git_hooks():
    """Setup git hooks for development"""
    try:
        subprocess.run(['pre-commit', 'install'], check=True)
        logger.info("Pre-commit hooks installed successfully")
    except subprocess.CalledProcessError:
        logger.warning("Failed to install pre-commit hooks. Run 'pre-commit install' manually.")
    except FileNotFoundError:
        logger.warning("Pre-commit not found. Install with 'pip install pre-commit'")

def create_sample_data():
    """Create sample data file"""
    sample_data = """statement,status
"Je me sens anxieux et inquiet",Anxiety
"Je suis vraiment heureux aujourd'hui",Happy
"Tout va bien dans ma vie",Normal
"Je me sens stressé par le travail",Stress
"J'ai l'impression d'être triste tout le temps",Depression
"Cette situation me rend nerveux",Anxiety
"Je suis content de ce qui m'arrive",Happy
"Ma journée se passe normalement",Normal
"Je n'arrive pas à gérer la pression",Stress
"Je me sens déprimé depuis quelques jours",Depression"""
    
    data_file = Path('data/sentiment_data.csv')
    if not data_file.exists():
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        logger.info("Created sample data file: data/sentiment_data.csv")

def create_config_files():
    """Create configuration files"""
    
    # .env.example
    env_example = """# Environment variables
ENVIRONMENT=development
ALERT_WEBHOOK_URL=http://localhost:3000/webhook
MODEL_REGISTRY=your-registry-url
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Database
MONITORING_DB_PATH=monitoring/model_logs.db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Prometheus
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    
    # pyproject.toml
    pyproject_toml = """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "sentiment-classifier"
dynamic = ["version"]
description = "A complete ML pipeline for sentiment classification"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
requires-python = ">=3.9"

[tool.setuptools_scm]

[tool.black]
line-length = 88
target-version = ["py39"]
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
include_trailing_comma = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "setup.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
]"""
    
    with open('pyproject.toml', 'w') as f:
        f.write(pyproject_toml)
    
    logger.info("Created configuration files")

def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'], check=True)
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    return True

def main():
    """Main setup function"""
    logger.info("Setting up sentiment classifier environment...")
    
    # Create directory structure
    create_directories()
    
    # Create configuration files
    create_config_files()
    
    # Create sample data
    create_sample_data()
    
    # Install dependencies (optional)
    if '--install-deps' in sys.argv:
        if not install_dependencies():
            logger.error("Setup completed with errors")
            return 1
    
    # Setup git hooks
    if '--setup-git' in sys.argv:
        setup_git_hooks()
    
    logger.info("Environment setup completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Copy .env.example to .env and configure variables")
    logger.info("2. Add your training data to data/sentiment_data.csv")
    logger.info("3. Run 'make train' to train the model")
    logger.info("4. Run 'make docker-compose-up' to start services")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


