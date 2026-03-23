"""
Data Models for Dataset Management and Validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np


class VariableType(str, Enum):
    """Variable type classification"""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    UNKNOWN = "unknown"


class MissingValueStrategy(str, Enum):
    """Strategies for handling missing values"""
    DROP_ROWS = "drop_rows"
    MEAN_IMPUTATION = "mean_imputation"
    MEDIAN_IMPUTATION = "median_imputation"
    MODE_IMPUTATION = "mode_imputation"
    KNN_IMPUTATION = "knn_imputation"
    MICE_IMPUTATION = "mice_imputation"


class NormalizationMethod(str, Enum):
    """Normalization methods"""
    STANDARD = "standard"  # Z-score
    MINMAX = "minmax"  # Min-Max scaling
    ROBUST = "robust"  # Robust scaling
    NONE = "none"


class VariableInfo(BaseModel):
    """Information about a single variable"""
    name: str = Field(..., description="Variable name")
    type: VariableType = Field(..., description="Detected variable type")
    dtype: str = Field(..., description="Pandas dtype")
    n_unique: int = Field(..., description="Number of unique values")
    n_missing: int = Field(default=0, description="Number of missing values")
    missing_percentage: float = Field(default=0.0, description="Percentage of missing values")
    
    # Statistics for continuous variables
    mean: Optional[float] = Field(default=None, description="Mean value")
    std: Optional[float] = Field(default=None, description="Standard deviation")
    min: Optional[float] = Field(default=None, description="Minimum value")
    max: Optional[float] = Field(default=None, description="Maximum value")
    median: Optional[float] = Field(default=None, description="Median value")
    q1: Optional[float] = Field(default=None, description="First quartile")
    q3: Optional[float] = Field(default=None, description="Third quartile")
    skewness: Optional[float] = Field(default=None, description="Skewness")
    kurtosis: Optional[float] = Field(default=None, description="Kurtosis")
    
    # Information for categorical variables
    categories: Optional[List[str]] = Field(default=None, description="Category labels")
    value_counts: Optional[Dict[str, int]] = Field(default=None, description="Value counts")
    
    # Quality indicators
    is_constant: bool = Field(default=False, description="Whether variable is constant")
    has_outliers: bool = Field(default=False, description="Whether outliers detected")
    outlier_percentage: float = Field(default=0.0, description="Percentage of outliers")
    
    class Config:
        use_enum_values = True


class DataStatistics(BaseModel):
    """Overall dataset statistics"""
    n_rows: int = Field(..., description="Number of rows")
    n_columns: int = Field(..., description="Number of columns")
    n_numeric: int = Field(..., description="Number of numeric columns")
    n_categorical: int = Field(..., description="Number of categorical columns")
    n_binary: int = Field(..., description="Number of binary columns")
    total_missing: int = Field(default=0, description="Total missing values")
    missing_percentage: float = Field(default=0.0, description="Overall missing percentage")
    duplicate_rows: int = Field(default=0, description="Number of duplicate rows")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    
    # Correlation summary
    high_correlation_pairs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pairs of highly correlated variables"
    )
    
    # Data quality score
    quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall data quality score (0-1)"
    )


class ValidationWarning(BaseModel):
    """Validation warning message"""
    variable: Optional[str] = Field(default=None, description="Variable name")
    message: str = Field(..., description="Warning message")
    severity: str = Field(default="warning", description="Severity level")
    suggestion: Optional[str] = Field(default=None, description="Suggested action")


class ValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool = Field(..., description="Whether data passed validation")
    variables: List[VariableInfo] = Field(default_factory=list, description="Variable information")
    statistics: Optional[DataStatistics] = Field(default=None, description="Dataset statistics")
    warnings: List[ValidationWarning] = Field(default_factory=list, description="Validation warnings")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    
    # Preprocessing recommendations
    recommended_preprocessing: List[str] = Field(
        default_factory=list,
        description="Recommended preprocessing steps"
    )


class DatasetUpload(BaseModel):
    """Dataset upload request"""
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (csv, xlsx, etc.)")
    data_base64: Optional[str] = Field(default=None, description="Base64 encoded data")
    
    # Preprocessing options
    missing_value_strategy: MissingValueStrategy = Field(
        default=MissingValueStrategy.MEAN_IMPUTATION,
        description="Strategy for handling missing values"
    )
    normalization: NormalizationMethod = Field(
        default=NormalizationMethod.NONE,
        description="Normalization method"
    )
    detect_types: bool = Field(default=True, description="Auto-detect variable types")
    remove_constants: bool = Field(default=True, description="Remove constant variables")
    handle_outliers: bool = Field(default=False, description="Handle outliers")
    
    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v):
        allowed = ['csv', 'xlsx', 'xls', 'json', 'parquet']
        if v.lower() not in allowed:
            raise ValueError(f"File type must be one of: {allowed}")
        return v.lower()


class DatasetInfo(BaseModel):
    """Information about uploaded dataset"""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    session_id: str = Field(..., description="Session identifier")
    filename: str = Field(..., description="Original filename")
    upload_time: datetime = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    n_rows: int = Field(..., description="Number of rows")
    n_columns: int = Field(..., description="Number of columns")
    column_names: List[str] = Field(..., description="Column names")
    file_size_bytes: int = Field(..., description="File size in bytes")
    
    # Processing status
    is_processed: bool = Field(default=False, description="Whether preprocessing is complete")
    preprocessing_applied: List[str] = Field(default_factory=list, description="Applied preprocessing steps")


class PreprocessRequest(BaseModel):
    """Request for data preprocessing"""
    dataset_id: str = Field(..., description="Dataset identifier")
    missing_value_strategy: MissingValueStrategy = Field(
        default=MissingValueStrategy.MEAN_IMPUTATION
    )
    normalization: NormalizationMethod = Field(
        default=NormalizationMethod.STANDARD
    )
    columns_to_normalize: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to normalize (None = all numeric)"
    )
    columns_to_drop: Optional[List[str]] = Field(
        default=None,
        description="Columns to drop"
    )
    encode_categorical: bool = Field(default=True, description="Encode categorical variables")
    remove_outliers: bool = Field(default=False, description="Remove outliers")
    outlier_method: str = Field(default="iqr", description="Outlier detection method")
    outlier_threshold: float = Field(default=1.5, description="Outlier threshold")


class PreprocessResult(BaseModel):
    """Result of preprocessing"""
    dataset_id: str = Field(..., description="Dataset identifier")
    original_shape: tuple = Field(..., description="Original data shape")
    processed_shape: tuple = Field(..., description="Processed data shape")
    steps_applied: List[str] = Field(..., description="Applied preprocessing steps")
    variables_removed: List[str] = Field(default_factory=list, description="Removed variables")
    rows_removed: int = Field(default=0, description="Number of rows removed")
    encoding_map: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Categorical encoding mappings"
    )
    normalization_params: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Normalization parameters for each column"
    )


class DataPreview(BaseModel):
    """Preview of dataset"""
    dataset_id: str = Field(..., description="Dataset identifier")
    columns: List[str] = Field(..., description="Column names")
    data: List[Dict[str, Any]] = Field(..., description="Preview rows")
    dtypes: Dict[str, str] = Field(..., description="Data types")
    total_rows: int = Field(..., description="Total number of rows")