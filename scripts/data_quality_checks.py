# scripts/data_quality_checks.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_text_patterns(df: pd.DataFrame) -> Dict:
    """Analyze text patterns and characteristics"""
    
    # Basic text statistics
    text_stats = {
        'avg_length': df['statement'].str.len().mean(),
        'min_length': df['statement'].str.len().min(),
        'max_length': df['statement'].str.len().max(),
        'std_length': df['statement'].str.len().std()
    }
    
    # Word count statistics
    word_counts = df['statement'].str.split().str.len()
    text_stats.update({
        'avg_words': word_counts.mean(),
        'min_words': word_counts.min(),
        'max_words': word_counts.max(),
        'std_words': word_counts.std()
    })
    
    # Language patterns
    all_text = ' '.join(df['statement'].values).lower()
    
    # Count common patterns
    patterns = {
        'exclamation_marks': all_text.count('!'),
        'question_marks': all_text.count('?'),
        'periods': all_text.count('.'),
        'commas': all_text.count(','),
        'uppercase_words': len(re.findall(r'\b[A-Z]{2,}\b', ' '.join(df['statement'].values)))
    }
    
    text_stats.update(patterns)
    
    return text_stats

def detect_anomalies(df: pd.DataFrame) -> Dict:
    """Detect potential data anomalies"""
    anomalies = {
        'suspicious_patterns': [],
        'outliers': [],
        'inconsistencies': []
    }
    
    # Check for repeated patterns
    text_counts = df['statement'].value_counts()
    highly_repeated = text_counts[text_counts > 5]
    if len(highly_repeated) > 0:
        anomalies['suspicious_patterns'].append(
            f"Found {len(highly_repeated)} texts repeated more than 5 times"
        )
    
    # Check for very short/long texts
    lengths = df['statement'].str.len()
    q1, q3 = lengths.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    
    outliers = df[(lengths < outlier_threshold_low) | (lengths > outlier_threshold_high)]
    if len(outliers) > 0:
        anomalies['outliers'].append(
            f"Found {len(outliers)} text length outliers"
        )
    
    # Check for inconsistent labels
    status_patterns = df.groupby('status')['statement'].apply(
        lambda x: x.str.len().mean()
    )
    
    if status_patterns.std() > 50:
        anomalies['inconsistencies'].append(
            "Large variation in text length between classes"
        )
    
    return anomalies

def generate_visualizations(df: pd.DataFrame, output_dir: str = "reports"):
    """Generate data quality visualizations"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Class distribution
    plt.figure(figsize=(10, 6))
    df['status'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png")
    plt.close()
    
    # Text length distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df['statement'].str.len().hist(bins=30)
    plt.title('Text Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df['statement'].str.split().str.len().hist(bins=30)
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_distribution.png")
    plt.close()
    
    # Text length by class
    plt.figure(figsize=(10, 6))
    df.boxplot(column='statement', by='status', 
               ax=plt.gca(), 
               figsize=(10, 6))
    plt.title('Text Length Distribution by Class')
    plt.suptitle('')
    plt.ylabel('Character Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_by_class.png")
    plt.close()

def save_quality_report(report: Dict, output_path: str = "reports/quality_report.json"):
    """Save quality report to JSON file"""
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def main():
    """Main data quality check function"""
    data_file = "data/sentiment_data.csv"
    
    logger.info("Starting comprehensive data quality checks...")
    
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Analyze text patterns
    text_analysis = analyze_text_patterns(df)
    logger.info("Text pattern analysis completed")
    
    # Detect anomalies
    anomalies = detect_anomalies(df)
    logger.info("Anomaly detection completed")
    
    # Generate visualizations
    generate_visualizations(df)
    logger.info("Visualizations generated")
    
    # Compile full report
    quality_report = {
        'data_summary': {
            'total_samples': len(df),
            'unique_samples': df['statement'].nunique(),
            'classes': df['status'].unique().tolist(),
            'class_counts': df['status'].value_counts().to_dict()
        },
        'text_analysis': text_analysis,
        'anomalies': anomalies,
        'recommendations': []
    }
    
    # Generate recommendations
    if quality_report['anomalies']['suspicious_patterns']:
        quality_report['recommendations'].append(
            "Review repeated patterns for potential data collection issues"
        )
    
    if quality_report['anomalies']['outliers']:
        quality_report['recommendations'].append(
            "Consider removing or investigating text length outliers"
        )
    
    if quality_report['data_summary']['total_samples'] < 1000:
        quality_report['recommendations'].append(
            "Consider collecting more data for better model performance"
        )
    
    # Save report
    save_quality_report(quality_report)
    
    # Log summary
    logger.info("Data Quality Summary:")
    logger.info(f"  Average text length: {text_analysis['avg_length']:.1f} characters")
    logger.info(f"  Average word count: {text_analysis['avg_words']:.1f} words")
    logger.info(f"  Anomalies detected: {len(anomalies['suspicious_patterns']) + len(anomalies['outliers'])}")
    
    if quality_report['recommendations']:
        logger.info("Recommendations:")
        for rec in quality_report['recommendations']:
            logger.info(f"  - {rec}")
    
    logger.info("Data quality checks completed successfully!")

if __name__ == "__main__":
    main()