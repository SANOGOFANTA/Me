# scripts/save_metrics.py
import json
import os
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model_metadata():
    """Save model metadata and metrics"""
    try:
        # Load model for metadata
        model = joblib.load('models/sentiment_model.pkl')
        
        # Load evaluation results if available
        eval_path = 'reports/evaluation_report.json'
        if os.path.exists(eval_path):
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
        else:
            eval_data = {}
        
        # Create comprehensive metadata
        metadata = {
            'model_info': {
                'model_type': type(model).__name__,
                'n_features': getattr(model, 'n_features_in_', 'unknown'),
                'classes': model.classes_.tolist() if hasattr(model, 'classes_') else 'unknown',
                'training_timestamp': datetime.now().isoformat()
            },
            'performance_metrics': eval_data.get('model_performance', {}),
            'model_artifacts': {
                'model_path': 'models/sentiment_model.pkl',
                'vectorizer_path': 'models/vectorizer.pkl',
                'metadata_path': 'models/model_metadata.json'
            },
            'deployment_info': {
                'version': '1.0.0',
                'environment': os.getenv('ENVIRONMENT', 'development'),
                'commit_hash': os.getenv('GITHUB_SHA', 'unknown')
            }
        }
        
        # Save metadata
        os.makedirs('models', exist_ok=True)
        with open('models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model metadata saved successfully")
        
        # Save metrics for monitoring
        metrics_for_monitoring = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': eval_data.get('model_performance', {}).get('accuracy', 0),
            'f1_macro': eval_data.get('model_performance', {}).get('f1_macro', 0),
            'f1_weighted': eval_data.get('model_performance', {}).get('f1_weighted', 0)
        }
        
        os.makedirs('monitoring', exist_ok=True)
        with open('monitoring/latest_metrics.json', 'w') as f:
            json.dump(metrics_for_monitoring, f, indent=2)
        
        logger.info("Metrics saved for monitoring")
        
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        raise

def main():
    """Main function"""
    save_model_metadata()

if __name__ == "__main__":
    main()