# scripts/train_model.py
import pandas as pd # type: ignore
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import logging
import mlflow # type: ignore
import mlflow.sklearn # type: ignore
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_data(file_path='data/Mentalhealth.csv'):
    """Load and preprocess data"""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def prepare_features(df):
    """Prepare features and target variables"""
    # Preprocess text
    df['statement_clean'] = df['statement'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['statement_clean'].str.len() > 0]
    
    X = df['statement_clean']
    y = df['status']
    
    logger.info(f"Feature preparation complete. Shape: {X.shape}")
    logger.info(f"Classes: {y.unique()}")
    logger.info(f"Class distribution:\n{y.value_counts()}")
    
    return X, y

def train_model(X_train, y_train, model_type='logistic'):
    """Train the model"""
    if model_type == 'logistic':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train model
    logger.info(f"Training {model_type} model...")
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate the model"""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_model(model, vectorizer, metrics, model_dir='models'):
    """Save model and artifacts"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and vectorizer
    joblib.dump(model, os.path.join(model_dir, 'sentiment_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    
    # Save metrics
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save model metadata
    metadata = {
        'model_type': type(model).__name__,
        'training_date': datetime.now().isoformat(),
        'accuracy': metrics['accuracy'],
        'classes': model.classes_.tolist()
    }
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_dir}")

def main():
    """Main training pipeline"""
    try:
        # Start MLflow run
        with mlflow.start_run():
            # Load data
            df = load_data()
            
            # Prepare features
            X, y = prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Train model
            model, vectorizer = train_model(X_train, y_train, model_type='logistic')
            
            # Evaluate model
            metrics = evaluate_model(model, vectorizer, X_test, y_test)
            
            # Log to MLflow
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("f1_macro", metrics['classification_report']['macro avg']['f1-score'])
            
            # Log model
            mlflow.sklearn.log_model(model, artifact_path="C:\\Users\\hp\\Documents\\GitHub\\Me\\artifacts")
            
            # Save model
            save_model(model, vectorizer, metrics)
            
            logger.info("Training pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()