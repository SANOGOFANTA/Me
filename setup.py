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


