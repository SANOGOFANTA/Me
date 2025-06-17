# tests/test_model.py
import pytest # type: ignore
import pandas as pd # type: ignore
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



