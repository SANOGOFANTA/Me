# scripts/evaluate_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(file_path='data/test_data.csv'):
    """Load test data"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logger.info(f"Loaded test data with {len(df)} samples")
            return df
        else:
            logger.warning("No test data file found, using train data split")
            return None
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return None

def evaluate_model_performance(model, vectorizer, X_test, y_test):
    """Comprehensive model evaluation"""
    # Make predictions
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'metrics': metrics,
        'class_report': class_report,
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist(),
        'true_labels': y_test.tolist()
    }

def create_evaluation_plots(evaluation_results, output_dir='reports'):
    """Create evaluation visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = np.array(evaluation_results['confusion_matrix'])
    class_names = list(evaluation_results['class_report'].keys())[:-3]  # Exclude avg metrics
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class metrics
    classes = [cls for cls in class_names if cls in evaluation_results['class_report']]
    precision = [evaluation_results['class_report'][cls]['precision'] for cls in classes]
    recall = [evaluation_results['class_report'][cls]['recall'] for cls in classes]
    f1 = [evaluation_results['class_report'][cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence distribution
    probabilities = np.array(evaluation_results['probabilities'])
    max_probs = np.max(probabilities, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_probs):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_evaluation_report(evaluation_results, output_path='reports/evaluation_report.json'):
    """Save detailed evaluation report"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add metadata
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_performance': evaluation_results['metrics'],
        'per_class_metrics': evaluation_results['class_report'],
        'confusion_matrix': evaluation_results['confusion_matrix'],
        'summary': {
            'total_samples': len(evaluation_results['true_labels']),
            'num_classes': len(set(evaluation_results['true_labels'])),
            'average_confidence': float(np.mean(np.max(evaluation_results['probabilities'], axis=1))),
            'low_confidence_predictions': int(np.sum(np.max(evaluation_results['probabilities'], axis=1) < 0.7))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Evaluation report saved to {output_path}")

def main():
    """Main evaluation function"""
    logger.info("Starting model evaluation...")
    
    try:
        # Load model and vectorizer
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        logger.info("Model and vectorizer loaded successfully")
        
        # Load test data
        test_df = load_test_data()
        
        if test_df is None:
            # Use training data for evaluation (split it)
            from scripts.train_model import load_data, prepare_features
            from sklearn.model_selection import train_test_split
            
            df = load_data()
            X, y = prepare_features(df)
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            from scripts.train_model import prepare_features
            X_test, y_test = prepare_features(test_df)
        
        # Evaluate model
        evaluation_results = evaluate_model_performance(model, vectorizer, X_test, y_test)
        
        # Create visualizations
        create_evaluation_plots(evaluation_results)
        
        # Save detailed report
        save_evaluation_report(evaluation_results)
        
        # Log summary
        metrics = evaluation_results['metrics']
        logger.info("Model Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
        
        # Check for model quality thresholds
        if metrics['accuracy'] < 0.7:
            logger.warning("Model accuracy is below 70%")
        
        if metrics['f1_macro'] < 0.6:
            logger.warning("Macro F1-score is below 60%")
        
        logger.info("Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()


