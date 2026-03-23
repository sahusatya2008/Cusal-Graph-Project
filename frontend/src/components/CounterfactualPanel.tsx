import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Play, ArrowRight } from 'lucide-react';
import toast from 'react-hot-toast';
import { useSessionStore } from '../store/sessionStore';

export default function CounterfactualPanel() {
  const { sessionId, modelId } = useSessionStore();
  const [evidence, setEvidence] = useState<Record<string, number>>({});
  const [intervention, setIntervention] = useState<Record<string, number>>({});
  const [targetVariable, setTargetVariable] = useState<string>('');
  const [result, setResult] = useState<any>(null);

  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: async () => {
      if (!modelId) return null;
      const response = await fetch(`/api/causal/model/${modelId}`, {
        headers: { 'X-Session-ID': sessionId || '' },
      });
      return response.json();
    },
    enabled: !!modelId,
  });

  const counterfactualMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/intervention/counterfactual', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId || '',
        },
        body: JSON.stringify({
          model_id: modelId,
          evidence,
          intervention,
          target_variable: targetVariable,
          n_samples: 10000,
        }),
      });
      if (!response.ok) throw new Error('Counterfactual failed');
      return response.json();
    },
    onSuccess: (data) => {
      setResult(data);
      toast.success('Counterfactual computed successfully!');
    },
    onError: () => {
      toast.error('Failed to compute counterfactual');
    },
  });

  if (!modelId) {
    return (
      <div className="card">
        <div className="text-center py-12">
          <p className="text-gray-500">Please learn a causal model first.</p>
        </div>
      </div>
    );
  }

  const variables = model?.graph?.nodes?.map((n: any) => n.name || n.id) || [];

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="card-header">Counterfactual Analysis</h2>
        <p className="text-gray-600 mb-4">
          Compute counterfactual queries: "What would Y have been if X had been x', given we observed X=x?"
        </p>

        <div className="grid grid-cols-3 gap-6">
          <div>
            <h3 className="font-semibold mb-3 text-blue-600">Factual Evidence (Observed)</h3>
            {Object.entries(evidence).map(([varName, val]) => (
              <div key={varName} className="flex items-center space-x-2 mb-2">
                <span className="font-mono bg-blue-100 px-2 py-1 rounded text-sm">{varName}</span>
                <span>=</span>
                <input
                  type="number"
                  value={val}
                  onChange={(e) => setEvidence(prev => ({ ...prev, [varName]: parseFloat(e.target.value) }))}
                  className="input w-24"
                />
                <button onClick={() => {
                  const next = { ...evidence };
                  delete next[varName];
                  setEvidence(next);
                }} className="text-red-500 text-sm">✕</button>
              </div>
            ))}
            <select 
              className="select mt-2"
              onChange={(e) => {
                if (e.target.value) setEvidence(prev => ({ ...prev, [e.target.value]: 0 }));
                e.target.value = '';
              }}
            >
              <option value="">Add evidence...</option>
              {variables.filter((v: string) => !evidence[v]).map((v: string) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div>
            <h3 className="font-semibold mb-3 text-green-600">Counterfactual Intervention</h3>
            {Object.entries(intervention).map(([varName, val]) => (
              <div key={varName} className="flex items-center space-x-2 mb-2">
                <span className="font-mono bg-green-100 px-2 py-1 rounded text-sm">{varName}</span>
                <span>=</span>
                <input
                  type="number"
                  value={val}
                  onChange={(e) => setIntervention(prev => ({ ...prev, [varName]: parseFloat(e.target.value) }))}
                  className="input w-24"
                />
                <button onClick={() => {
                  const next = { ...intervention };
                  delete next[varName];
                  setIntervention(next);
                }} className="text-red-500 text-sm">✕</button>
              </div>
            ))}
            <select 
              className="select mt-2"
              onChange={(e) => {
                if (e.target.value) setIntervention(prev => ({ ...prev, [e.target.value]: 0 }));
                e.target.value = '';
              }}
            >
              <option value="">Add intervention...</option>
              {variables.filter((v: string) => !intervention[v]).map((v: string) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>

          <div>
            <h3 className="font-semibold mb-3 text-purple-600">Target Variable</h3>
            <select 
              className="select"
              value={targetVariable}
              onChange={(e) => setTargetVariable(e.target.value)}
            >
              <option value="">Select target...</option>
              {variables.map((v: string) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={() => counterfactualMutation.mutate()}
            disabled={counterfactualMutation.isPending || !targetVariable || Object.keys(intervention).length === 0}
            className="btn btn-primary flex items-center space-x-2"
          >
            <Play size={18} />
            <span>{counterfactualMutation.isPending ? 'Computing...' : 'Compute Counterfactual'}</span>
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h3 className="card-header">Counterfactual Results</h3>
          
          <div className="grid grid-cols-2 gap-6">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h4 className="font-semibold text-blue-700">Factual Value</h4>
              <p className="text-3xl font-bold mt-2">{result.factual_value?.toFixed(4)}</p>
              <p className="text-sm text-gray-500 mt-1">Observed outcome</p>
            </div>
            <div className="p-4 bg-green-50 rounded-lg">
              <h4 className="font-semibold text-green-700">Counterfactual Value</h4>
              <p className="text-3xl font-bold mt-2">{result.counterfactual_value?.toFixed(4)}</p>
              <p className="text-sm text-gray-500 mt-1">Hypothetical outcome</p>
            </div>
          </div>

          <div className="mt-4 p-4 bg-purple-50 rounded-lg">
            <h4 className="font-semibold text-purple-700">Causal Effect</h4>
            <div className="flex items-center space-x-4 mt-2">
              <span className="text-lg">{result.factual_value?.toFixed(3)}</span>
              <ArrowRight className="text-purple-500" />
              <span className="text-lg font-bold">{result.counterfactual_value?.toFixed(3)}</span>
              <span className="text-sm text-gray-500">
                (Δ = {((result.counterfactual_value || 0) - (result.factual_value || 0)).toFixed(4)})
              </span>
            </div>
          </div>

          {result.explanation && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold">Explanation</h4>
              <p className="text-sm text-gray-600 mt-2">{result.explanation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}