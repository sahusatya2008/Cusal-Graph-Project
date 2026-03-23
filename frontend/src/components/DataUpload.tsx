import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileSpreadsheet, AlertCircle, CheckCircle } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { useSessionStore } from '../store/sessionStore';

interface UploadResponse {
  dataset_id: string;
  filename: string;
  n_rows: number;
  n_columns: number;
}

async function uploadFile(file: File, sessionId: string): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/data/upload', {
    method: 'POST',
    headers: {
      'X-Session-ID': sessionId,
    },
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Upload failed');
  }
  
  return response.json();
}

export default function DataUpload() {
  const { sessionId, setSessionId, datasetId, setDatasetId } = useSessionStore();
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const queryClient = useQueryClient();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploadStatus('uploading');
    
    try {
      const result = await uploadFile(file, sessionId || '');
      setDatasetId(result.dataset_id);
      setUploadStatus('success');
      toast.success(`Uploaded ${result.filename}: ${result.n_rows} rows, ${result.n_columns} columns`);
      queryClient.invalidateQueries({ queryKey: ['dataset'] });
    } catch (error) {
      setUploadStatus('error');
      toast.error('Upload failed. Please try again.');
    }
  }, [sessionId, setDatasetId, queryClient]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024,
  });

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="card-header">Upload Dataset</h2>
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-indigo-500 bg-indigo-50'
              : 'border-gray-300 hover:border-indigo-400'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          {isDragActive ? (
            <p className="text-indigo-600 font-medium">Drop the file here...</p>
          ) : (
            <div>
              <p className="text-gray-600 font-medium">
                Drag & drop a dataset file here, or click to select
              </p>
              <p className="text-gray-400 text-sm mt-2">
                Supported formats: CSV, Excel (.xlsx), JSON (max 100MB)
              </p>
            </div>
          )}
        </div>

        {uploadStatus === 'success' && datasetId && (
          <div className="mt-4 p-4 bg-green-50 rounded-lg flex items-center space-x-3">
            <CheckCircle className="text-green-500" size={20} />
            <span className="text-green-700">Dataset uploaded successfully!</span>
          </div>
        )}

        {uploadStatus === 'error' && (
          <div className="mt-4 p-4 bg-red-50 rounded-lg flex items-center space-x-3">
            <AlertCircle className="text-red-500" size={20} />
            <span className="text-red-700">Upload failed. Please try again.</span>
          </div>
        )}
      </div>

      {datasetId && <DatasetPreview datasetId={datasetId} />}
    </div>
  );
}

function DatasetPreview({ datasetId }: { datasetId: string }) {
  const { sessionId } = useSessionStore();
  
  const { data: preview, isLoading } = useQuery({
    queryKey: ['dataset', datasetId, 'preview'],
    queryFn: async () => {
      const response = await fetch(`/api/data/${datasetId}/preview?n_rows=10`, {
        headers: { 'X-Session-ID': sessionId || '' },
      });
      return response.json();
    },
  });

  if (isLoading) {
    return <div className="card">Loading preview...</div>;
  }

  return (
    <div className="card">
      <h2 className="card-header">Data Preview</h2>
      <div className="overflow-x-auto">
        <table className="table">
          <thead>
            <tr>
              {preview?.columns?.map((col: string) => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview?.data?.map((row: Record<string, unknown>, idx: number) => (
              <tr key={idx}>
                {preview?.columns?.map((col: string) => (
                  <td key={col}>{String(row[col])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}