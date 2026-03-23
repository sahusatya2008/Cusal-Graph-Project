import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Play, Plus, Trash2, BarChart2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { useSessionStore } from '../store/sessionStore';

export default function InterventionPanel() {
  const { sessionId, modelId } = useSessionStore();
  const [interventions, setInterventions] = useState<Record<string, number>>({});
  const [targetVariables, setTargetVariables] = useState<string[]>([]);
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

  const interventionMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/intervention/intervene', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId || '',
        },
        body: JSON.stringify({
          model_id: modelId,
          intervention_variables: interventions,
          target_variables: targetVariables,
          n_samples: 5000,
          compute_confidence_intervals: true,
        }),
      });
      if (!response.ok) throw new Error('Intervention failed');
      return response.json();
    },
    onSuccess: (data) => {
      setResult(data);
      toast.success('Intervention computed successfully!');
    },
    onError: () => {
      toast.error('Failed to compute intervention');
    },
  });

  const addIntervention = (variable: string, value: number) => {
    setInterventions(prev => ({ ...prev, [variable]: value }));
  };

  const removeIntervention = (variable: string) => {
    setInterventions(prev => {
      const next = { ...prev };
      delete next[variable];
      return next;
    });
  };

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
        <h2 className="card-header">Intervention Simulator</h2>
        <p className="text-gray-600 mb-4">
          Compute interventional distributions P(Y | do(X=x)) using the truncated factorization formula.
        </p>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-3">Intervention Variables (do)</h3>
            {Object.entries(interventions).map(([varName, val]) => (
              <div key={varName} className="flex items-center space-x-2 mb-2">
                <span className="font-mono bg-indigo-100 px-2 py-1 rounded">{varName}</span>
                <span>=</span>
                <input
                  type="number"
                  value={val}
                  onChange={(e) => addIntervention(varName, parseFloat(e.target.value))}
                  className="input w-24"
                />
                <button onClick={() => removeIntervention(varName)} className="text-red-500">
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
            
            <div className="flex items-center space-x-2 mt-4">
              <select 
                className="select flex-1"
                onChange={(e) => {
                  if (e.target.value) addIntervention(e.target.value, 0);
                  e.target.value = '';
                }}
              >
                <option value="">Add variable...</option>
                {variables.filter((v: string) => !interventions[v]).map((v: string) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <h3 className="font-semibold mb-3">Target Variables</h3>
            {variables.map((v: string) => (
              <label key={v} className="flex items-center space-x-2 mb-2">
                <input
                  type="checkbox"
                  checked={targetVariables.includes(v)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setTargetVariables(prev => [...prev, v]);
                    } else {
                      setTargetVariables(prev => prev.filter(t => t !== v));
                    }
                  }}
                  className="rounded"
                />
                <span>{v}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={() => interventionMutation.mutate()}
            disabled={interventionMutation.isPending || Object.keys(interventions).length === 0}
            className="btn btn-primary flex items-center space-x-2"
          >
            <Play size={18} />
            <span>{interventionMutation.isPending ? 'Computing...' : 'Compute Intervention'}</span>
          </button>
        </div>
      </div>

      {result && (
        <div className="card">
          <h3 className="card-header">Intervention Results</h3>
          
          <div className="mb-4 p-4 bg-indigo-50 rounded-lg">
            <p className="font-mono text-sm">{result.intervention_formula}</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(result.distribution_stats || {}).map(([varName, stats]: [string, any]) => (
              <div key={varName} className="p-4 bg-gray-50 rounded-lg">
                <h4 className="font-semibold text-indigo-600">{varName}</h4>
                <div className="mt-2 space-y-1 text-sm">
                  <p>Mean: {stats.mean?.toFixed(4)}</p>
                  <p>Std: {stats.std?.toFixed(4)}</p>
                  <p>Median: {stats.median?.toFixed(4)}</p>
                  <p>95% CI: [{stats.q1?.toFixed(2)}, {stats.q3?.toFixed(2)}]</p>
                </div>
              </div>
            ))}
          </div>

          {result.causal_effects && (
            <div className="mt-6">
              <h4 className="font-semibold mb-2">Causal Effects</h4>
              <div className="space-y-2">
                {Object.entries(result.causal_effects).map(([varName, effect]: [string, any]) => (
                  <div key={varName} className="flex justify-between p-2 bg-gray-50 rounded">
                    <span>{varName}</span>
                    <span className="font-mono">{effect.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}