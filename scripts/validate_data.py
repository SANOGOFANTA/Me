# scripts/validate_data.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import jsonschema
from jsonschema import validate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data schema definition
DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "statement": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000
        },
        "status": {
            "type": "string",
            "enum": ["Anxiety", "Depression", "Stress", "Normal", "Happy"]
        }
    },
    "required": ["statement", "status"]
}

def validate_data_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate data against predefined schema"""
    errors = []
    
    # Check required columns
    required_columns = ["statement", "status"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors
    
    # Check data types
    if not df['statement'].dtype == 'object':
        errors.append("Column 'statement' should be of type string/object")
    
    if not df['status'].dtype == 'object':
        errors.append("Column 'status' should be of type string/object")
    
    # Check for empty values
    empty_statements = df['statement'].isna().sum() + (df['statement'] == '').sum()
    if empty_statements > 0:
        errors.append(f"Found {empty_statements} empty statements")
    
    empty_status = df['status'].isna().sum()
    if empty_status > 0:
        errors.append(f"Found {empty_status} empty status values")
    
    # Check valid status values
    valid_statuses = ["Anxiety", "Depression", "Stress", "Normal", "Happy"]
    invalid_statuses = set(df['status'].dropna()) - set(valid_statuses)
    if invalid_statuses:
        errors.append(f"Found invalid status values: {invalid_statuses}")
    
    # Check text length
    long_texts = df[df['statement'].str.len() > 1000].shape[0]
    if long_texts > 0:
        errors.append(f"Found {long_texts} texts longer than 1000 characters")
    
    return len(errors) == 0, errors

def check_data_quality(df: pd.DataFrame) -> Dict:
    """Perform comprehensive data quality checks"""
    quality_report = {
        'total_samples': len(df),
        'duplicate_count': df.duplicated().sum(),
        'missing_values': df.isnull().sum().to_dict(),
        'class_distribution': df['status'].value_counts().to_dict(),
        'text_length_stats': df['statement'].str.len().describe().to_dict(),
        'unique_texts': df['statement'].nunique(),
        'warnings': [],
        'errors': []
    }
    
    # Check for class imbalance
    class_counts = df['status'].value_counts()
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    
    if max_class_count / min_class_count > 10:
        quality_report['warnings'].append(
            f"Severe class imbalance detected. Ratio: {max_class_count/min_class_count:.2f}"
        )
    
    # Check for duplicates
    if quality_report['duplicate_count'] > 0:
        quality_report['warnings'].append(
            f"Found {quality_report['duplicate_count']} duplicate samples"
        )
    
    # Check minimum samples per class
    if min_class_count < 10:
        quality_report['errors'].append(
            f"Insufficient samples for class {class_counts.idxmin()}: {min_class_count}"
        )
    
    # Check text diversity
    unique_ratio = quality_report['unique_texts'] / quality_report['total_samples']
    if unique_ratio < 0.8:
        quality_report['warnings'].append(
            f"Low text diversity. Unique ratio: {unique_ratio:.2f}"
        )
    
    return quality_report

def validate_file_format(file_path: str) -> Tuple[bool, str]:
    """Validate file format and structure"""
    try:
        if not Path(file_path).exists():
            return False, f"File does not exist: {file_path}"
        
        # Try to read the file
        df = pd.read_csv(file_path)
        
        if df.empty:
            return False, "File is empty"
        
        if len(df.columns) < 2:
            return False, f"Insufficient columns. Expected at least 2, got {len(df.columns)}"
        
        return True, "File format is valid"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def main():
    """Main validation function"""
    data_file = "data/sentiment_data.csv"
    
    logger.info("Starting data validation...")
    
    # Validate file format
    is_valid_format, format_message = validate_file_format(data_file)
    if not is_valid_format:
        logger.error(f"File format validation failed: {format_message}")
        sys.exit(1)
    
    logger.info("File format validation passed")
    
    # Load data
    try:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} samples from {data_file}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Validate schema
    is_valid_schema, schema_errors = validate_data_schema(df)
    if not is_valid_schema:
        logger.error("Schema validation failed:")
        for error in schema_errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    logger.info("Schema validation passed")
    
    # Quality checks
    quality_report = check_data_quality(df)
    
    logger.info("Data Quality Report:")
    logger.info(f"  Total samples: {quality_report['total_samples']}")
    logger.info(f"  Unique texts: {quality_report['unique_texts']}")
    logger.info(f"  Duplicates: {quality_report['duplicate_count']}")
    logger.info(f"  Class distribution: {quality_report['class_distribution']}")
    
    # Log warnings
    if quality_report['warnings']:
        logger.warning("Data quality warnings:")
        for warning in quality_report['warnings']:
            logger.warning(f"  - {warning}")
    
    # Check for critical errors
    if quality_report['errors']:
        logger.error("Critical data quality errors:")
        for error in quality_report['errors']:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    logger.info("Data validation completed successfully!")

if __name__ == "__main__":
    main()


