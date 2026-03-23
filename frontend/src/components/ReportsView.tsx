import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { FileText, Download, Play, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { useSessionStore } from '../store/sessionStore';

export default function ReportsView() {
  const { sessionId, modelId } = useSessionStore();
  const [reportType, setReportType] = useState<'full' | 'summary' | 'technical'>('full');
  const [includeSections, setIncludeSections] = useState({
    methodology: true,
    results: true,
    sensitivity: true,
    recommendations: true,
    appendix: false,
  });
  const [generatedReport, setGeneratedReport] = useState<any>(null);

  const generateMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/reports/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId || '',
        },
        body: JSON.stringify({
          model_id: modelId,
          report_type: reportType,
          include_sections: includeSections,
          format: 'json',
        }),
      });
      if (!response.ok) throw new Error('Report generation failed');
      return response.json();
    },
    onSuccess: (data) => {
      setGeneratedReport(data);
      toast.success('Report generated successfully!');
    },
    onError: () => {
      toast.error('Failed to generate report');
    },
  });

  const downloadReport = async (format: 'pdf' | 'html' | 'markdown') => {
    if (!generatedReport) return;
    
    try {
      const response = await fetch(`/api/reports/${generatedReport.report_id}/download?format=${format}`, {
        headers: { 'X-Session-ID': sessionId || '' },
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `causal_report.${format}`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      toast.error('Download failed');
    }
  };

  if (!modelId) {
    return (
      <div className="card">
        <div className="text-center py-12">
          <p className="text-gray-500">Please learn a causal model first to generate reports.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="card-header flex items-center">
          <FileText className="mr-2" size={20} />
          Generate Report
        </h2>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-3">Report Type</h3>
            <div className="space-y-2">
              {[
                { value: 'full', label: 'Full Report', desc: 'Complete analysis with all details' },
                { value: 'summary', label: 'Executive Summary', desc: 'Key findings and recommendations' },
                { value: 'technical', label: 'Technical Report', desc: 'Methodology and statistical details' },
              ].map((type) => (
                <label
                  key={type.value}
                  className={`flex items-start p-3 border rounded-lg cursor-pointer transition-colors ${
                    reportType === type.value ? 'border-indigo-500 bg-indigo-50' : 'hover:bg-gray-50'
                  }`}
                >
                  <input
                    type="radio"
                    name="reportType"
                    value={type.value}
                    checked={reportType === type.value}
                    onChange={() => setReportType(type.value as any)}
                    className="mt-1 mr-3"
                  />
                  <div>
                    <p className="font-medium">{type.label}</p>
                    <p className="text-sm text-gray-500">{type.desc}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div>
            <h3 className="font-semibold mb-3">Include Sections</h3>
            <div className="space-y-2">
              {Object.entries(includeSections).map(([section, checked]) => (
                <label key={section} className="flex items-center p-2 hover:bg-gray-50 rounded">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(e) => setIncludeSections(prev => ({ ...prev, [section]: e.target.checked }))}
                    className="mr-3 rounded"
                  />
                  <span className="capitalize">{section.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={() => generateMutation.mutate()}
            disabled={generateMutation.isPending}
            className="btn btn-primary flex items-center space-x-2"
          >
            <Play size={18} />
            <span>{generateMutation.isPending ? 'Generating...' : 'Generate Report'}</span>
          </button>
        </div>
      </div>

      {generatedReport && (
        <div className="space-y-6">
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="card-header mb-0">Report Preview</h3>
              <div className="flex space-x-2">
                <button onClick={() => downloadReport('pdf')} className="btn btn-secondary flex items-center space-x-1">
                  <Download size={16} />
                  <span>PDF</span>
                </button>
                <button onClick={() => downloadReport('html')} className="btn btn-secondary flex items-center space-x-1">
                  <Download size={16} />
                  <span>HTML</span>
                </button>
                <button onClick={() => downloadReport('markdown')} className="btn btn-secondary flex items-center space-x-1">
                  <Download size={16} />
                  <span>MD</span>
                </button>
              </div>
            </div>

            <div className="prose max-w-none">
              {generatedReport.sections?.map((section: any, idx: number) => (
                <div key={idx} className="mb-6">
                  <h4 className="text-lg font-semibold text-indigo-700">{section.title}</h4>
                  <div className="text-gray-600 mt-2" dangerouslySetInnerHTML={{ __html: section.content }} />
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h3 className="card-header">Report Metadata</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500">Generated</p>
                <p className="font-medium">{new Date(generatedReport.created_at).toLocaleString()}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500">Model ID</p>
                <p className="font-mono text-sm">{generatedReport.model_id?.slice(0, 12)}...</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500">Sections</p>
                <p className="font-medium">{generatedReport.sections?.length || 0}</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-xs text-gray-500">Status</p>
                <p className="flex items-center text-green-600">
                  <CheckCircle size={16} className="mr-1" />
                  Complete
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}