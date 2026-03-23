"""
Data Service for Dataset Validation and Preprocessing
Handles data upload, validation, preprocessing, and storage
"""
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import os
import io
import base64
import logging
from scipy import stats

from app.config import settings
from app.models.data_models import (
    ValidationResult, VariableInfo, DataStatistics, 
    ValidationWarning, VariableType, DatasetInfo,
    PreprocessResult, DataPreview,
    MissingValueStrategy, NormalizationMethod
)

logger = logging.getLogger(__name__)


class DataService:
    """
    Service for data validation and preprocessing
    """
    
    def __init__(self):
        self._storage_path = settings.DATA_DIR
        
    async def upload_dataset(
        self, 
        session_id: str,
        filename: str,
        content: bytes,
        file_type: str
    ) -> Tuple[str, pd.DataFrame]:
        """
        Upload and parse a dataset
        
        Returns:
            Tuple of (dataset_id, dataframe)
        """
        dataset_id = str(uuid.uuid4())
        
        # Parse based on file type
        if file_type == 'csv':
            df = pd.read_csv(io.BytesIO(content))
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(content))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(content))
        elif file_type == 'parquet':
            df = pd.read_parquet(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Store the dataframe
        session_dir = os.path.join(self._storage_path, session_id, "datasets")
        os.makedirs(session_dir, exist_ok=True)
        
        file_path = os.path.join(session_dir, f"{dataset_id}.parquet")
        df.to_parquet(file_path)
        
        logger.info(f"Dataset {dataset_id} uploaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return dataset_id, df
    
    async def load_dataset(self, session_id: str, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load a dataset from storage"""
        file_path = os.path.join(
            self._storage_path, session_id, "datasets", f"{dataset_id}.parquet"
        )
        
        if not os.path.exists(file_path):
            return None
            
        return pd.read_parquet(file_path)
    
    async def save_dataset(self, session_id: str, dataset_id: str, df: pd.DataFrame):
        """Save a dataset to storage"""
        file_path = os.path.join(
            self._storage_path, session_id, "datasets", f"{dataset_id}.parquet"
        )
        df.to_parquet(file_path)
    
    async def validate_dataset(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate dataset and compute statistics
        """
        warnings = []
        errors = []
        variables = []
        
        # Check minimum size
        if len(df) < 10:
            errors.append("Dataset has fewer than 10 rows. Causal inference may be unreliable.")
        if len(df.columns) < 2:
            errors.append("Dataset must have at least 2 variables.")
        
        # Analyze each variable
        for col in df.columns:
            var_info = await self._analyze_variable(df, col)
            variables.append(var_info)
            
            # Generate warnings
            if var_info.missing_percentage > 0.2:
                warnings.append(ValidationWarning(
                    variable=col,
                    message=f"Variable has {var_info.missing_percentage*100:.1f}% missing values",
                    severity="warning",
                    suggestion="Consider imputation or removing this variable"
                ))
            
            if var_info.is_constant:
                warnings.append(ValidationWarning(
                    variable=col,
                    message="Variable is constant (no variance)",
                    severity="warning",
                    suggestion="Remove this variable as it provides no information"
                ))
            
            if var_info.has_outliers and var_info.outlier_percentage > 0.05:
                warnings.append(ValidationWarning(
                    variable=col,
                    message=f"Variable has {var_info.outlier_percentage*100:.1f}% outliers",
                    severity="info",
                    suggestion="Consider outlier treatment or robust methods"
                ))
        
        # Compute overall statistics
        statistics = await self._compute_statistics(df, variables)
        
        # Generate preprocessing recommendations
        recommendations = self._generate_recommendations(variables, statistics)
        
        # Determine if valid
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            variables=variables,
            statistics=statistics,
            warnings=warnings,
            errors=errors,
            recommended_preprocessing=recommendations
        )
    
    async def _analyze_variable(self, df: pd.DataFrame, col: str) -> VariableInfo:
        """Analyze a single variable"""
        series = df[col]
        n_total = len(series)
        n_missing = series.isna().sum()
        n_unique = series.nunique(dropna=True)
        
        # Detect variable type
        var_type = self._detect_variable_type(series)
        
        # Base info
        info = VariableInfo(
            name=col,
            type=var_type,
            dtype=str(series.dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            missing_percentage=n_missing / n_total if n_total > 0 else 0,
            is_constant=n_unique <= 1
        )
        
        # Numeric statistics
        if var_type in [VariableType.CONTINUOUS, VariableType.BINARY]:
            numeric = pd.to_numeric(series, errors='coerce')
            valid = numeric.dropna()
            
            if len(valid) > 0:
                info.mean = float(valid.mean())
                info.std = float(valid.std())
                info.min = float(valid.min())
                info.max = float(valid.max())
                info.median = float(valid.median())
                info.q1 = float(valid.quantile(0.25))
                info.q3 = float(valid.quantile(0.75))
                info.skewness = float(valid.skew())
                info.kurtosis = float(valid.kurtosis())
                
                # Outlier detection (IQR method)
                iqr = info.q3 - info.q1
                lower = info.q1 - 1.5 * iqr
                upper = info.q3 + 1.5 * iqr
                outliers = ((valid < lower) | (valid > upper)).sum()
                info.has_outliers = outliers > 0
                info.outlier_percentage = outliers / len(valid) if len(valid) > 0 else 0
        
        # Categorical statistics
        if var_type in [VariableType.CATEGORICAL, VariableType.ORDINAL, VariableType.BINARY]:
            value_counts = series.value_counts(dropna=True)
            info.categories = list(value_counts.index.astype(str))
            info.value_counts = dict(value_counts)
        
        return info
    
    def _detect_variable_type(self, series: pd.Series) -> VariableType:
        """Detect the type of a variable"""
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            n_unique = series.nunique(dropna=True)
            
            # Binary check
            if n_unique == 2:
                return VariableType.BINARY
            
            # Check if it might be categorical (few unique values)
            if n_unique <= 10 and n_unique / len(series) < 0.05:
                return VariableType.CATEGORICAL
            
            return VariableType.CONTINUOUS
        
        # String/object type
        n_unique = series.nunique(dropna=True)
        
        # Check for boolean-like strings
        unique_vals = set(series.dropna().astype(str).str.lower())
        bool_vals = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
        if unique_vals.issubset(bool_vals) or unique_vals == {'0', '1'}:
            return VariableType.BINARY
        
        # Check if ordinal (contains numbers or ordered categories)
        if pd.api.types.is_categorical_dtype(series) and series.cat.ordered:
            return VariableType.ORDINAL
        
        return VariableType.CATEGORICAL
    
    async def _compute_statistics(
        self, 
        df: pd.DataFrame, 
        variables: List[VariableInfo]
    ) -> DataStatistics:
        """Compute overall dataset statistics"""
        n_numeric = sum(1 for v in variables if v.type == VariableType.CONTINUOUS)
        n_categorical = sum(1 for v in variables if v.type in [VariableType.CATEGORICAL, VariableType.ORDINAL])
        n_binary = sum(1 for v in variables if v.type == VariableType.BINARY)
        
        total_missing = sum(v.n_missing for v in variables)
        duplicate_rows = df.duplicated().sum()
        
        # Memory usage
        memory_bytes = df.memory_usage(deep=True).sum()
        
        # High correlation pairs
        high_corr_pairs = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = abs(corr_matrix.loc[col1, col2])
                    if corr > 0.8:
                        high_corr_pairs.append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr)
                        })
        
        # Quality score
        quality_score = self._compute_quality_score(df, variables)
        
        return DataStatistics(
            n_rows=len(df),
            n_columns=len(df.columns),
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_binary=n_binary,
            total_missing=total_missing,
            missing_percentage=total_missing / (len(df) * len(df.columns)) if len(df) > 0 else 0,
            duplicate_rows=duplicate_rows,
            memory_usage_mb=memory_bytes / (1024 * 1024),
            high_correlation_pairs=high_corr_pairs,
            quality_score=quality_score
        )
    
    def _compute_quality_score(self, df: pd.DataFrame, variables: List[VariableInfo]) -> float:
        """Compute overall data quality score (0-1)"""
        scores = []
        
        # Completeness score
        total_cells = len(df) * len(df.columns)
        missing_cells = sum(v.n_missing for v in variables)
        completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 1
        scores.append(completeness)
        
        # Uniqueness score (low duplicates is good)
        uniqueness = 1 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 1
        scores.append(uniqueness)
        
        # Variability score (variables should have variance)
        n_constant = sum(1 for v in variables if v.is_constant)
        variability = 1 - (n_constant / len(variables)) if len(variables) > 0 else 1
        scores.append(variability)
        
        # Size adequacy score
        size_score = min(1.0, len(df) / 100)  # 100+ rows is ideal
        scores.append(size_score)
        
        return np.mean(scores)
    
    def _generate_recommendations(
        self, 
        variables: List[VariableInfo], 
        statistics: DataStatistics
    ) -> List[str]:
        """Generate preprocessing recommendations"""
        recommendations = []
        
        # Missing values
        if statistics.missing_percentage > 0:
            if statistics.missing_percentage < 0.05:
                recommendations.append("Drop rows with missing values (low missing rate)")
            else:
                recommendations.append("Apply imputation strategy for missing values")
        
        # Constant variables
        constant_vars = [v.name for v in variables if v.is_constant]
        if constant_vars:
            recommendations.append(f"Remove constant variables: {', '.join(constant_vars)}")
        
        # Highly correlated variables
        if statistics.high_correlation_pairs:
            recommendations.append(
                "Review highly correlated variables for potential multicollinearity"
            )
        
        # Outliers
        outlier_vars = [v.name for v in variables if v.has_outliers and v.outlier_percentage > 0.05]
        if outlier_vars:
            recommendations.append(f"Consider outlier treatment for: {', '.join(outlier_vars)}")
        
        # Normalization
        numeric_vars = [v for v in variables if v.type == VariableType.CONTINUOUS]
        if numeric_vars:
            ranges = [(v.max - v.min) for v in numeric_vars if v.max is not None and v.min is not None]
            if ranges and (max(ranges) / min(ranges) > 10 if min(ranges) > 0 else False):
                recommendations.append("Normalize numeric variables (different scales detected)")
        
        return recommendations
    
    async def preprocess_dataset(
        self,
        session_id: str,
        dataset_id: str,
        df: pd.DataFrame,
        missing_strategy: MissingValueStrategy = MissingValueStrategy.MEAN_IMPUTATION,
        normalization: NormalizationMethod = NormalizationMethod.STANDARD,
        columns_to_normalize: Optional[List[str]] = None,
        columns_to_drop: Optional[List[str]] = None,
        encode_categorical: bool = True,
        remove_outliers: bool = False,
        outlier_threshold: float = 1.5
    ) -> Tuple[pd.DataFrame, PreprocessResult]:
        """
        Preprocess dataset according to specified options
        """
        original_shape = df.shape
        steps_applied = []
        variables_removed = []
        rows_removed = 0
        encoding_map = {}
        normalization_params = {}
        
        # Make a copy
        df_processed = df.copy()
        
        # Drop specified columns
        if columns_to_drop:
            for col in columns_to_drop:
                if col in df_processed.columns:
                    df_processed = df_processed.drop(columns=[col])
                    variables_removed.append(col)
            steps_applied.append(f"Dropped columns: {', '.join(columns_to_drop)}")
        
        # Remove constant variables
        constant_cols = [col for col in df_processed.columns 
                        if df_processed[col].nunique(dropna=True) <= 1]
        if constant_cols:
            df_processed = df_processed.drop(columns=constant_cols)
            variables_removed.extend(constant_cols)
            steps_applied.append(f"Removed constant variables: {', '.join(constant_cols)}")
        
        # Handle missing values
        if df_processed.isna().any().any():
            df_processed, missing_steps = await self._handle_missing(
                df_processed, missing_strategy
            )
            steps_applied.extend(missing_steps)
        
        # Remove outliers
        if remove_outliers:
            df_processed, n_removed = await self._remove_outliers(
                df_processed, outlier_threshold
            )
            rows_removed = n_removed
            steps_applied.append(f"Removed {n_removed} outlier rows")
        
        # Normalize numeric columns
        if normalization != NormalizationMethod.NONE:
            df_processed, norm_params = await self._normalize(
                df_processed, normalization, columns_to_normalize
            )
            normalization_params = norm_params
            steps_applied.append(f"Applied {normalization} normalization")
        
        # Encode categorical variables
        if encode_categorical:
            df_processed, enc_map = await self._encode_categorical(df_processed)
            encoding_map = enc_map
            if enc_map:
                steps_applied.append("Encoded categorical variables")
        
        # Save processed dataset
        await self.save_dataset(session_id, dataset_id, df_processed)
        
        result = PreprocessResult(
            dataset_id=dataset_id,
            original_shape=original_shape,
            processed_shape=df_processed.shape,
            steps_applied=steps_applied,
            variables_removed=variables_removed,
            rows_removed=rows_removed,
            encoding_map=encoding_map,
            normalization_params=normalization_params
        )
        
        return df_processed, result
    
    async def _handle_missing(
        self, 
        df: pd.DataFrame, 
        strategy: MissingValueStrategy
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values"""
        steps = []
        
        if strategy == MissingValueStrategy.DROP_ROWS:
            n_before = len(df)
            df = df.dropna()
            n_after = len(df)
            steps.append(f"Dropped {n_before - n_after} rows with missing values")
            
        elif strategy == MissingValueStrategy.MEAN_IMPUTATION:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
            # Mode for categorical
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in cat_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode()[0])
            steps.append("Imputed missing values (mean for numeric, mode for categorical)")
            
        elif strategy == MissingValueStrategy.MEDIAN_IMPUTATION:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for col in cat_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode()[0])
            steps.append("Imputed missing values (median for numeric, mode for categorical)")
            
        elif strategy == MissingValueStrategy.MODE_IMPUTATION:
            for col in df.columns:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mode()[0])
            steps.append("Imputed missing values with mode")
        
        return df, steps
    
    async def _remove_outliers(
        self, 
        df: pd.DataFrame, 
        threshold: float
    ) -> Tuple[pd.DataFrame, int]:
        """Remove outliers using IQR method"""
        n_before = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        n_removed = n_before - len(df)
        return df, n_removed
    
    async def _normalize(
        self, 
        df: pd.DataFrame, 
        method: NormalizationMethod,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Normalize numeric columns"""
        params = {}
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == NormalizationMethod.STANDARD:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                params[col] = {"mean": mean, "std": std}
                
            elif method == NormalizationMethod.MINMAX:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                params[col] = {"min": min_val, "max": max_val}
                
            elif method == NormalizationMethod.ROBUST:
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df[col] = (df[col] - median) / iqr
                params[col] = {"median": median, "iqr": iqr}
        
        return df, params
    
    async def _encode_categorical(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """Encode categorical variables"""
        encoding_map = {}
        
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in cat_cols:
            unique_vals = df[col].dropna().unique()
            mapping = {val: i for i, val in enumerate(sorted(unique_vals))}
            df[col] = df[col].map(mapping)
            encoding_map[col] = mapping
        
        return df, encoding_map
    
    async def get_preview(
        self, 
        df: pd.DataFrame, 
        dataset_id: str, 
        n_rows: int = 10
    ) -> DataPreview:
        """Get a preview of the dataset"""
        preview_df = df.head(n_rows)
        
        return DataPreview(
            dataset_id=dataset_id,
            columns=list(df.columns),
            data=preview_df.to_dict(orient='records'),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            total_rows=len(df)
        )
    
    async def get_dataset_info(
        self,
        session_id: str,
        dataset_id: str,
        filename: str
    ) -> DatasetInfo:
        """Get dataset information"""
        df = await self.load_dataset(session_id, dataset_id)
        
        if df is None:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        file_path = os.path.join(
            self._storage_path, session_id, "datasets", f"{dataset_id}.parquet"
        )
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        return DatasetInfo(
            dataset_id=dataset_id,
            session_id=session_id,
            filename=filename,
            upload_time=datetime.utcnow(),
            n_rows=len(df),
            n_columns=len(df.columns),
            column_names=list(df.columns),
            file_size_bytes=file_size
        )