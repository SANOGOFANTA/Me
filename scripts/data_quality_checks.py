# scripts/data_quality_checks.py
from typing import Dict
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from collections import Counter # type: ignore
import re
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_text_patterns(df: pd.DataFrame) -> Dict:
    """Analyze text patterns and characteristics"""
    
    # Filter out null values first
    valid_statements = df['statement'].dropna()
    
    if len(valid_statements) == 0:
        logger.warning("No valid text data found!")
        return {}
    
    # Basic text statistics
    text_stats = {
        'total_samples': len(df),
        'valid_samples': len(valid_statements),
        'null_samples': len(df) - len(valid_statements),
        'avg_length': valid_statements.str.len().mean(),
        'min_length': valid_statements.str.len().min(),
        'max_length': valid_statements.str.len().max(),
        'std_length': valid_statements.str.len().std()
    }
    
    # Word count statistics
    word_counts = valid_statements.str.split().str.len()
    text_stats.update({
        'avg_words': word_counts.mean(),
        'min_words': word_counts.min(),
        'max_words': word_counts.max(),
        'std_words': word_counts.std()
    })
    
    # Language patterns - only use valid statements
    all_text = ' '.join(valid_statements.values).lower()
    
    # Count common patterns
    patterns = {
        'exclamation_marks': all_text.count('!'),
        'question_marks': all_text.count('?'),
        'periods': all_text.count('.'),
        'commas': all_text.count(','),
        'uppercase_words': len(re.findall(r'\b[A-Z]{2,}\b', ' '.join(valid_statements.values)))
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
    
    # Filter out null values
    valid_statements = df['statement'].dropna()
    
    if len(valid_statements) == 0:
        anomalies['suspicious_patterns'].append("All statements are null!")
        return anomalies
    
    # Report null values if any
    null_count = len(df) - len(valid_statements)
    if null_count > 0:
        anomalies['suspicious_patterns'].append(
            f"Found {null_count} null/empty statements ({null_count/len(df)*100:.1f}%)"
        )
    
    # Check for repeated patterns
    text_counts = valid_statements.value_counts()
    highly_repeated = text_counts[text_counts > 5]
    if len(highly_repeated) > 0:
        anomalies['suspicious_patterns'].append(
            f"Found {len(highly_repeated)} texts repeated more than 5 times"
        )
    
    # Check for very short/long texts
    lengths = valid_statements.str.len()
    q1, q3 = lengths.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_threshold_low = q1 - 1.5 * iqr
    outlier_threshold_high = q3 + 1.5 * iqr
    
    outliers_mask = (lengths < outlier_threshold_low) | (lengths > outlier_threshold_high)
    outlier_count = outliers_mask.sum()
    if outlier_count > 0:
        anomalies['outliers'].append(
            f"Found {outlier_count} text length outliers"
        )
    
    # Check for inconsistent labels - only use rows with valid statements
    valid_df = df.dropna(subset=['statement'])
    if len(valid_df) > 0:
        status_patterns = valid_df.groupby('status')['statement'].apply(
            lambda x: x.str.len().mean()
        )
        
        if len(status_patterns) > 1 and status_patterns.std() > 50:
            anomalies['inconsistencies'].append(
                "Large variation in text length between classes"
            )
    
    return anomalies

def generate_visualizations(df: pd.DataFrame, output_dir: str = "reports"):
    """Generate data quality visualizations"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Filter out null values for visualizations
    valid_df = df.dropna(subset=['statement'])
    
    if len(valid_df) == 0:
        logger.warning("No valid data for visualizations!")
        return
    
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
    
    # Text length distribution (only valid statements)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    valid_df['statement'].str.len().hist(bins=30)
    plt.title('Text Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    valid_df['statement'].str.split().str.len().hist(bins=30)
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/text_length_distribution.png")
    plt.close()
    
    # Text length by class (only valid statements)
    plt.figure(figsize=(10, 6))
    
    # Create a temporary column for plotting
    valid_df_copy = valid_df.copy()
    valid_df_copy['text_length'] = valid_df_copy['statement'].str.len()
    
    # Use seaborn for better boxplot handling
    sns.boxplot(data=valid_df_copy, x='status', y='text_length')
    plt.title('Text Length Distribution by Class')
    plt.ylabel('Character Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_by_class.png")
    plt.close()
    
    # Add a data quality summary plot
    plt.figure(figsize=(10, 6))
    quality_stats = {
        'Valid Samples': len(valid_df),
        'Null Samples': len(df) - len(valid_df),
        'Unique Samples': valid_df['statement'].nunique()
    }
    
    plt.bar(quality_stats.keys(), quality_stats.values())
    plt.title('Data Quality Overview')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    for i, v in enumerate(quality_stats.values()):
        plt.text(i, v + max(quality_stats.values()) * 0.01, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_quality_overview.png")
    plt.close()

def save_quality_report(report: Dict, output_path: str = "reports/quality_report.json"):
    """Save quality report to JSON file"""
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

def main():
    """Main data quality check function"""
    data_file = "data/Mentalhealth.csv"
    
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