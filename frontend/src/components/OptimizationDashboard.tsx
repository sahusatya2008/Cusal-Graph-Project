import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Play, Target, TrendingUp, AlertTriangle } from 'lucide-react';
import toast from 'react-hot-toast';
import { useSessionStore } from '../store/sessionStore';

export default function OptimizationDashboard() {
  const { sessionId, modelId } = useSessionStore();
  const [targetVariable, setTargetVariable] = useState<string>('');
  const [objective, setObjective] = useState<'maximize' | 'minimize'>('maximize');
  const [constraints, setConstraints] = useState<any[]>([]);
  const [budget, setBudget] = useState<number>(1000);
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

  const optimizeMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/optimization/optimize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId || '',
        },
        body: JSON.stringify({
          model_id: modelId,
          target_variable: targetVariable,
          objective,
          intervention_variables: constraints.filter(c => c.is_actionable).map(c => c.variable),
          constraints: constraints.reduce((acc, c) => {
            if (c.min !== undefined || c.max !== undefined) {
              acc[c.variable] = { min: c.min, max: c.max };
            }
            return acc;
          }, {} as Record<string, any>),
          budget,
          method: 'bayesian',
        }),
      });
      if (!response.ok) throw new Error('Optimization failed');
      return response.json();
    },
    onSuccess: (data) => {
      setResult(data);
      toast.success('Optimization completed!');
    },
    onError: () => {
      toast.error('Optimization failed');
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
        <h2 className="card-header">Policy Optimization</h2>
        <p className="text-gray-600 mb-4">
          Find optimal intervention values to maximize or minimize a target variable while respecting constraints.
        </p>

        <div className="grid grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-3">Objective</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-500 mb-1">Target Variable</label>
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
              <div>
                <label className="block text-sm text-gray-500 mb-1">Goal</label>
                <div className="flex space-x-4">
                  <label className="flex items-center">
                    <input
                      type="radio"
                      checked={objective === 'maximize'}
                      onChange={() => setObjective('maximize')}
                      className="mr-2"
                    />
                    <TrendingUp size={16} className="mr-1 text-green-500" />
                    Maximize
                  </label>
                  <label className="flex items-center">
                    <input
                      type="radio"
                      checked={objective === 'minimize'}
                      onChange={() => setObjective('minimize')}
                      className="mr-2"
                    />
                    <TrendingUp size={16} className="mr-1 text-red-500 rotate-180" />
                    Minimize
                  </label>
                </div>
              </div>
              <div>
                <label className="block text-sm text-gray-500 mb-1">Budget Constraint</label>
                <input
                  type="number"
                  value={budget}
                  onChange={(e) => setBudget(parseFloat(e.target.value))}
                  className="input"
                />
              </div>
            </div>
          </div>

          <div>
            <h3 className="font-semibold mb-3">Intervention Constraints</h3>
            {constraints.map((c, idx) => (
              <div key={idx} className="flex items-center space-x-2 mb-2 text-sm">
                <input
                  type="checkbox"
                  checked={c.is_actionable}
                  onChange={(e) => {
                    const next = [...constraints];
                    next[idx].is_actionable = e.target.checked;
                    setConstraints(next);
                  }}
                  className="rounded"
                  title="Actionable"
                />
                <span className="font-mono bg-gray-100 px-2 py-1 rounded">{c.variable}</span>
                <input
                  type="number"
                  placeholder="min"
                  value={c.min || ''}
                  onChange={(e) => {
                    const next = [...constraints];
                    next[idx].min = parseFloat(e.target.value);
                    setConstraints(next);
                  }}
                  className="input w-20"
                />
                <span>≤</span>
                <span className="text-gray-500">{c.variable}</span>
                <span>≤</span>
                <input
                  type="number"
                  placeholder="max"
                  value={c.max || ''}
                  onChange={(e) => {
                    const next = [...constraints];
                    next[idx].max = parseFloat(e.target.value);
                    setConstraints(next);
                  }}
                  className="input w-20"
                />
                <button
                  onClick={() => setConstraints(prev => prev.filter((_, i) => i !== idx))}
                  className="text-red-500"
                >
                  ✕
                </button>
              </div>
            ))}
            <select
              className="select mt-2"
              onChange={(e) => {
                if (e.target.value) {
                  setConstraints(prev => [...prev, { 
                    variable: e.target.value, 
                    is_actionable: true,
                    min: undefined,
                    max: undefined 
                  }]);
                  e.target.value = '';
                }
              }}
            >
              <option value="">Add constraint...</option>
              {variables.filter((v: string) => !constraints.find(c => c.variable === v)).map((v: string) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-6">
          <button
            onClick={() => optimizeMutation.mutate()}
            disabled={optimizeMutation.isPending || !targetVariable}
            className="btn btn-primary flex items-center space-x-2"
          >
            <Target size={18} />
            <span>{optimizeMutation.isPending ? 'Optimizing...' : 'Find Optimal Policy'}</span>
          </button>
        </div>
      </div>

      {result && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="card-header">Optimal Policy</h3>
            
            <div className="grid grid-cols-2 gap-6">
              <div className="p-4 bg-green-50 rounded-lg">
                <h4 className="font-semibold text-green-700">Optimal {targetVariable}</h4>
                <p className="text-3xl font-bold mt-2">
                  {result.optimal_value?.toFixed(4)}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {objective === 'maximize' ? 'Maximum' : 'Minimum'} achievable value
                </p>
              </div>
              <div className="p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-700">Baseline Value</h4>
                <p className="text-3xl font-bold mt-2">
                  {result.baseline_value?.toFixed(4)}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  Without intervention
                </p>
              </div>
            </div>

            <div className="mt-4">
              <h4 className="font-semibold mb-2">Optimal Interventions</h4>
              <div className="space-y-2">
                {Object.entries(result.optimal_interventions || {}).map(([varName, value]: [string, any]) => (
                  <div key={varName} className="flex justify-between p-3 bg-gray-50 rounded">
                    <span className="font-mono">{varName}</span>
                    <span className="font-bold">{value.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 p-4 bg-purple-50 rounded-lg">
              <h4 className="font-semibold text-purple-700">Expected Improvement</h4>
              <p className="text-2xl font-bold mt-2">
                +{((result.optimal_value || 0) - (result.baseline_value || 0)).toFixed(4)}
                <span className="text-sm text-gray-500 ml-2">
                  ({(((result.optimal_value || 0) - (result.baseline_value || 0)) / (result.baseline_value || 1) * 100).toFixed(1)}%)
                </span>
              </p>
            </div>
          </div>

          {result.sensitivity && (
            <div className="card">
              <h3 className="card-header">Sensitivity Analysis</h3>
              <div className="space-y-2">
                {Object.entries(result.sensitivity).map(([varName, sens]: [string, any]) => (
                  <div key={varName} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span>{varName}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-32 h-2 bg-gray-200 rounded">
                        <div 
                          className="h-full bg-indigo-500 rounded"
                          style={{ width: `${Math.min(Math.abs(sens) * 100, 100)}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono">{sens.toFixed(4)}</span>
                    </div>
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