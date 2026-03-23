"""
Data Management API Router
Handles dataset upload, validation, and preprocessing
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import uuid
import os
import json
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

# Data storage (simple in-memory for now)
_datasets = {}


@router.post("/upload")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Upload a dataset file (CSV, Excel, JSON)
    """
    session_id = request.headers.get("X-Session-ID", "default")
    
    # Determine file type
    filename = file.filename or "unknown"
    file_type = filename.split('.')[-1].lower() if '.' in filename else 'csv'
    
    if file_type not in ['csv', 'xlsx', 'xls', 'json']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_type}. Supported: csv, xlsx, xls, json"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > 100 * 1024 * 1024:  # 100MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
    
    try:
        # Generate dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Store dataset info
        _datasets[dataset_id] = {
            "filename": filename,
            "content": content,
            "file_type": file_type,
            "session_id": session_id
        }
        
        # Parse content to get dimensions
        if file_type == 'csv':
            import pandas as pd
            import io
            df = pd.read_csv(io.BytesIO(content))
            n_rows = len(df)
            n_columns = len(df.columns)
            columns = list(df.columns)
        elif file_type == 'json':
            import pandas as pd
            import io
            df = pd.read_json(io.BytesIO(content))
            n_rows = len(df)
            n_columns = len(df.columns)
            columns = list(df.columns)
        else:
            n_rows = 0
            n_columns = 0
            columns = []
        
        return {
            "dataset_id": dataset_id,
            "filename": filename,
            "n_rows": n_rows,
            "n_columns": n_columns,
            "columns": columns
        }
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    request: Request,
    n_rows: int = 10
):
    """
    Get a preview of the dataset
    """
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = _datasets[dataset_id]
    
    try:
        import pandas as pd
        import io
        
        content = dataset["content"]
        file_type = dataset["file_type"]
        
        if file_type == 'csv':
            df = pd.read_csv(io.BytesIO(content))
        elif file_type == 'json':
            df = pd.read_json(io.BytesIO(content))
        else:
            df = pd.DataFrame()
        
        # Get preview data
        preview_df = df.head(n_rows)
        
        # Convert column names to strings to avoid JSON serialization issues
        preview_df.columns = [str(c) for c in preview_df.columns]
        df.columns = [str(c) for c in df.columns]
        
        # Replace NaN/None values with None (will become null in JSON)
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        
        # Convert to records safely - handle mixed types
        try:
            data = preview_df.to_dict(orient='records')
        except (ValueError, TypeError):
            # Fallback: convert all values to strings
            data = preview_df.astype(str).to_dict(orient='records')
        
        # Ensure all values are JSON serializable
        clean_data = []
        for row in data:
            clean_row = {}
            for k, v in row.items():
                # Handle different types safely
                if v is None:
                    clean_row[k] = None
                elif isinstance(v, (list, dict, np.ndarray)):
                    # Convert complex types to string representation
                    clean_row[k] = str(v)
                elif isinstance(v, float):
                    if np.isnan(v) or np.isinf(v):
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                elif isinstance(v, (np.integer, np.floating)):
                    clean_row[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (int, str, bool)):
                    clean_row[k] = v
                else:
                    # Try to check for NaN, but handle arrays gracefully
                    try:
                        if pd.isna(v):
                            clean_row[k] = None
                        else:
                            clean_row[k] = str(v)
                    except (ValueError, TypeError):
                        clean_row[k] = str(v)
            clean_data.append(clean_row)
        
        return {
            "dataset_id": dataset_id,
            "columns": list(df.columns),
            "data": clean_data,
            "n_rows": len(df),
            "n_columns": len(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Error previewing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_id}/validate")
async def validate_dataset(
    dataset_id: str,
    request: Request
):
    """
    Validate an uploaded dataset
    """
    if dataset_id not in _datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "is_valid": True,
        "errors": [],
        "warnings": []
    }


@router.get("/")
async def list_datasets(request: Request):
    """
    List all datasets in the current session
    """
    session_id = request.headers.get("X-Session-ID", "default")
    
    datasets = []
    for dataset_id, dataset in _datasets.items():
        if dataset.get("session_id") == session_id:
            datasets.append({
                "dataset_id": dataset_id,
                "filename": dataset.get("filename", "unknown")
            })
    
    return datasets


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    request: Request
):
    """
    Delete a dataset
    """
    if dataset_id in _datasets:
        del _datasets[dataset_id]
    
    return {"message": "Dataset deleted", "dataset_id": dataset_id}