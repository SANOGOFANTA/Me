# scripts/prepare_data.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def prepare_training_data(input_file, output_dir='data', test_size=0.2):
    """Prepare and split training data"""
    logger.info(f"Loading data from {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} samples")
    
    # Clean text
    df['statement'] = df['statement'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['statement'].str.len() > 0]
    logger.info(f"After cleaning: {len(df)} samples")
    
    # Check class distribution
    class_counts = df['status'].value_counts()
    logger.info(f"Class distribution:\n{class_counts}")
    
    # Split data
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df['status']
    )
    
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save splits
    train_file = f"{output_dir}/train_data.csv"
    test_file = f"{output_dir}/test_data.csv"
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    logger.info(f"Training data saved: {train_file} ({len(train_df)} samples)")
    logger.info(f"Test data saved: {test_file} ({len(test_df)} samples)")
    
    return train_file, test_file

def validate_data_quality(df):
    """Validate data quality"""
    issues = []
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['statement']).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate texts")
    
    # Check class balance
    class_counts = df['status'].value_counts()
    min_count = class_counts.min()
    max_count = class_counts.max()
    
    if max_count / min_count > 10:
        issues.append(f"Severe class imbalance detected (ratio: {max_count/min_count:.2f})")
    
    # Check text lengths
    lengths = df['statement'].str.len()
    very_short = (lengths < 10).sum()
    very_long = (lengths > 500).sum()
    
    if very_short > 0:
        issues.append(f"Found {very_short} very short texts (< 10 chars)")
    
    if very_long > 0:
        issues.append(f"Found {very_long} very long texts (> 500 chars)")
    
    return issues

def main():
    """Main data preparation function"""
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--input', default='data/sentiment_data.csv', 
                       help='Input data file')
    parser.add_argument('--output-dir', default='data', 
                       help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Test set size')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate data quality')
    
    args = parser.parse_args()
    
    # Load data for validation
    if args.validate:
        df = pd.read_csv(args.input)
        issues = validate_data_quality(df)
        
        if issues:
            logger.warning("Data quality issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No data quality issues found")
    
    # Prepare data
    train_file, test_file = prepare_training_data(
        args.input, 
        args.output_dir, 
        args.test_size
    )
    
    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()