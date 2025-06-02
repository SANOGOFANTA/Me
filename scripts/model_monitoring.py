# scripts/model_monitoring.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import json
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple
import requests
import os
from prometheus_client import Gauge, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
MODEL_ACCURACY = Gauge('model_accuracy', 'Current model accuracy')
DRIFT_SCORE = Gauge('model_drift_score', 'Model drift score')
PREDICTION_CONFIDENCE = Gauge('prediction_confidence_avg', 'Average prediction confidence')

class ModelMonitor:
    def __init__(self, model_path='models/sentiment_model.pkl', 
                 vectorizer_path='models/vectorizer.pkl',
                 db_path='monitoring/model_logs.db'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.db_path = db_path
        self.model = None
        self.vectorizer = None
        self._initialize_db()
        self._load_model()
    
    def _initialize_db(self):
        """Initialize monitoring database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_text TEXT,
                prediction TEXT,
                confidence REAL,
                actual_label TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                additional_info TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_model(self):
        """Load model and vectorizer"""
        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def log_prediction(self, text: str, prediction: str, 
                      confidence: float, actual_label: str = None):
        """Log prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (input_text, prediction, confidence, actual_label)
            VALUES (?, ?, ?, ?)
        ''', (text, prediction, confidence, actual_label))
        
        conn.commit()
        conn.close()
    
    def log_metric(self, metric_name: str, value: float, info: str = None):
        """Log model metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (metric_name, metric_value, additional_info)
            VALUES (?, ?, ?)
        ''', (metric_name, value, info))
        
        conn.commit()
        conn.close()
    
    def calculate_accuracy(self, days: int = 7) -> float:
        """Calculate accuracy over last N days"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT prediction, actual_label
            FROM predictions
            WHERE actual_label IS NOT NULL
            AND timestamp >= datetime('now', '-{} days')
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            logger.warning("No labeled data found for accuracy calculation")
            return None
        
        accuracy = accuracy_score(df['actual_label'], df['prediction'])
        
        # Update Prometheus metric
        MODEL_ACCURACY.set(accuracy)
        
        # Log to database
        self.log_metric('accuracy', accuracy, f'calculated over {days} days')
        
        return accuracy
    
    def detect_drift(self, reference_period_days: int = 30, 
                     current_period_days: int = 7) -> Dict:
        """Detect model drift by comparing prediction distributions"""
        conn = sqlite3.connect(self.db_path)
        
        # Get reference period data
        ref_query = '''
            SELECT prediction, confidence
            FROM predictions
            WHERE timestamp BETWEEN datetime('now', '-{} days') AND datetime('now', '-{} days')
        '''.format(reference_period_days, current_period_days)
        
        ref_df = pd.read_sql_query(ref_query, conn)
        
        # Get current period data
        curr_query = '''
            SELECT prediction, confidence
            FROM predictions
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(current_period_days)
        
        curr_df = pd.read_sql_query(curr_query, conn)
        conn.close()
        
        if len(ref_df) == 0 or len(curr_df) == 0:
            logger.warning("Insufficient data for drift detection")
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Compare prediction distributions
        ref_dist = ref_df['prediction'].value_counts(normalize=True)
        curr_dist = curr_df['prediction'].value_counts(normalize=True)
        
        # Calculate KL divergence
        drift_score = self._calculate_kl_divergence(ref_dist, curr_dist)
        
        # Compare confidence distributions
        ref_conf_mean = ref_df['confidence'].mean()
        curr_conf_mean = curr_df['confidence'].mean()
        confidence_drift = abs(curr_conf_mean - ref_conf_mean)
        
        # Update Prometheus metrics
        DRIFT_SCORE.set(drift_score)
        PREDICTION_CONFIDENCE.set(curr_conf_mean)
        
        # Log metrics
        self.log_metric('drift_score', drift_score)
        self.log_metric('confidence_drift', confidence_drift)
        
        drift_threshold = 0.1
        confidence_threshold = 0.05
        
        drift_detected = (drift_score > drift_threshold or 
                         confidence_drift > confidence_threshold)
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'confidence_drift': confidence_drift,
            'reference_confidence': ref_conf_mean,
            'current_confidence': curr_conf_mean
        }
    
    def _calculate_kl_divergence(self, p, q):
        """Calculate KL divergence between two distributions"""
        # Ensure all classes are present in both distributions
        all_classes = set(p.index) | set(q.index)
        
        p_aligned = pd.Series(index=all_classes, dtype=float).fillna(1e-10)
        q_aligned = pd.Series(index=all_classes, dtype=float).fillna(1e-10)
        
        for cls in all_classes:
            if cls in p.index:
                p_aligned[cls] = p[cls]
            if cls in q.index:
                q_aligned[cls] = q[cls]
        
        # Normalize
        p_aligned = p_aligned / p_aligned.sum()
        q_aligned = q_aligned / q_aligned.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(p_aligned * np.log(p_aligned / q_aligned))
        return kl_div
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_path': self.model_path,
                'model_type': type(self.model).__name__ if self.model else None
            }
        }
        
        # Calculate recent accuracy
        accuracy = self.calculate_accuracy(days=7)
        if accuracy is not None:
            report['accuracy_7_days'] = accuracy
        
        # Detect drift
        drift_info = self.detect_drift()
        report['drift_analysis'] = drift_info
        
        # Get prediction statistics
        conn = sqlite3.connect(self.db_path)
        
        # Recent prediction volume
        volume_query = '''
            SELECT COUNT(*) as count
            FROM predictions
            WHERE timestamp >= datetime('now', '-1 days')
        '''
        volume_df = pd.read_sql_query(volume_query, conn)
        report['daily_prediction_volume'] = int(volume_df['count'].iloc[0])
        
        # Average confidence
        conf_query = '''
            SELECT AVG(confidence) as avg_confidence
            FROM predictions
            WHERE timestamp >= datetime('now', '-7 days')
        '''
        conf_df = pd.read_sql_query(conf_query, conn)
        if not conf_df['avg_confidence'].isna().iloc[0]:
            report['avg_confidence_7_days'] = float(conf_df['avg_confidence'].iloc[0])
        
        conn.close()
        
        return report
    
    def send_alert(self, message: str, severity: str = 'warning'):
        """Send alert notification"""
        # This could be integrated with Slack, email, or other notification systems
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': severity,
            'service': 'sentiment-classifier'
        }
        
        # Log alert
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        # Send to webhook (if configured)
        webhook_url = os.getenv('ALERT_WEBHOOK_URL')
        if webhook_url:
            try:
                requests.post(webhook_url, json=alert_data)
            except Exception as e:
                logger.error(f"Failed to send alert webhook: {e}")

def main():
    """Main monitoring function"""
    monitor = ModelMonitor()
    
    logger.info("Starting model monitoring check...")
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    
    # Check for alerts
    if 'drift_analysis' in report and report['drift_analysis']['drift_detected']:
        monitor.send_alert(
            f"Model drift detected! Drift score: {report['drift_analysis']['drift_score']:.4f}",
            severity='warning'
        )
    
    if 'accuracy_7_days' in report and report['accuracy_7_days'] < 0.8:
        monitor.send_alert(
            f"Model accuracy dropped to {report['accuracy_7_days']:.4f}",
            severity='critical'
        )
    
    if 'avg_confidence_7_days' in report and report['avg_confidence_7_days'] < 0.7:
        monitor.send_alert(
            f"Average prediction confidence dropped to {report['avg_confidence_7_days']:.4f}",
            severity='warning'
        )
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    with open(f"reports/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("Monitoring check completed")
    
    return report

if __name__ == "__main__":
    main()


