# tests/test_model.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from scripts.train_model import preprocess_text, prepare_features, train_model

class TestPreprocessing:
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "  Hello WORLD!  "
        result = preprocess_text(text)
        assert result == "hello world!"
    
    def test_preprocess_text_empty(self):
        """Test preprocessing empty text"""
        result = preprocess_text("")
        assert result == ""
    
    def test_preprocess_text_nan(self):
        """Test preprocessing NaN values"""
        result = preprocess_text(pd.NA)
        assert result == ""

class TestFeaturePreparation:
    def test_prepare_features(self):
        """Test feature preparation"""
        df = pd.DataFrame({
            'statement': ['Hello world', 'This is a test', ''],
            'status': ['Positive', 'Neutral', 'Negative']
        })
        
        X, y = prepare_features(df)
        
        # Should remove empty texts
        assert len(X) == 2
        assert len(y) == 2
        assert 'hello world' in X.values
        assert 'this is a test' in X.values

class TestModelTraining:
    @patch('scripts.train_model.logger')
    def test_train_model_logistic(self, mock_logger):
        """Test logistic regression training"""
        X_train = pd.Series(['hello world', 'this is positive', 'this is negative'])
        y_train = pd.Series(['Positive', 'Positive', 'Negative'])
        
        model, vectorizer = train_model(X_train, y_train, model_type='logistic')
        
        assert model is not None
        assert vectorizer is not None
        assert hasattr(model, 'predict')
        assert hasattr(vectorizer, 'transform')



# tests/integration_tests.py
import requests
import time
import argparse
import sys

def test_api_endpoints(base_url):
    """Test API endpoints"""
    print(f"Testing API at {base_url}")
    
    # Test health check
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200, f"Health check failed: {response.status_code}"
    print("✓ Health check passed")
    
    # Test root endpoint
    response = requests.get(base_url)
    assert response.status_code == 200, f"Root endpoint failed: {response.status_code}"
    print("✓ Root endpoint passed")
    
    # Test prediction endpoint
    test_data = {"text": "I feel anxious and worried"}
    response = requests.post(f"{base_url}/predict", json=test_data)
    assert response.status_code == 200, f"Prediction failed: {response.status_code}"
    
    result = response.json()
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    print("✓ Prediction endpoint passed")
    
    # Test batch prediction
    batch_data = {"texts": ["I'm happy", "I'm sad", "I'm worried"]}
    response = requests.post(f"{base_url}/predict_batch", json=batch_data)
    assert response.status_code == 200, f"Batch prediction failed: {response.status_code}"
    
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 3
    print("✓ Batch prediction endpoint passed")
    
    print("All integration tests passed!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="staging", help="Environment to test")
    args = parser.parse_args()
    
    if args.env == "staging":
        base_url = "http://staging-api-url"
    elif args.env == "production":
        base_url = "http://production-api-url"
    else:
        base_url = "http://localhost:8000"
    
    # Wait for service to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                break
        except:
            pass
        
        if i == max_retries - 1:
            print("Service not ready after 30 attempts")
            sys.exit(1)
        
        print(f"Waiting for service... ({i+1}/{max_retries})")
        time.sleep(10)
    
    test_api_endpoints(base_url)

if __name__ == "__main__":
    main()