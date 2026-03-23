import React, { useCallback, useMemo, useState } from 'react';
import ReactFlow, {
  Node, Edge, Background, Controls, MiniMap,
  useNodesState, useEdgesState, Position
} from 'reactflow';
import dagre from 'dagre';
import 'reactflow/dist/style.css';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Play, Settings, Download, Info } from 'lucide-react';
import toast from 'react-hot-toast';
import { useSessionStore } from '../store/sessionStore';

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 150;
const nodeHeight = 50;

function getLayoutedElements(nodes: Node[], edges: Edge[], direction = 'TB') {
  const isHorizontal = direction === 'LR';
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = isHorizontal ? Position.Left : Position.Top;
    node.sourcePosition = isHorizontal ? Position.Right : Position.Bottom;
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };
  });

  return { nodes, edges };
}

export default function CausalGraphView() {
  const { sessionId, datasetId, modelId, setModelId } = useSessionStore();
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [learningMethod, setLearningMethod] = useState('hybrid');

  const learnMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`/api/causal/learn/${datasetId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Session-ID': sessionId || '',
        },
        body: JSON.stringify({
          method: learningMethod,
          use_bootstrap: true,
          n_bootstrap: 50,
        }),
      });
      if (!response.ok) throw new Error('Learning failed');
      return response.json();
    },
    onSuccess: (data) => {
      setModelId(data.model_id);
      toast.success('Causal structure learned successfully!');
    },
    onError: () => {
      toast.error('Failed to learn causal structure');
    },
  });

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

  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    if (!model?.graph) return { nodes: [], edges: [] };

    const nodes: Node[] = model.graph.nodes.map((n: { id?: string; name?: string }) => ({
      id: n.id || n.name || '',
      data: { label: n.name || n.id },
      style: {
        background: selectedNode === n.id ? '#e0e7ff' : '#fff',
        border: '2px solid #6366f1',
        borderRadius: '8px',
        padding: '10px',
      },
    }));

    const edges: Edge[] = model.graph.edges.map((e: { source: string; target: string; coefficient?: number }) => ({
      id: `${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      animated: true,
      style: { stroke: '#6366f1', strokeWidth: 2 },
      label: e.coefficient ? `${e.coefficient.toFixed(2)}` : undefined,
      labelStyle: { fill: '#6366f1', fontWeight: 700 },
    }));

    return getLayoutedElements(nodes, edges);
  }, [model, selectedNode]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node.id);
  }, []);

  if (!datasetId) {
    return (
      <div className="card">
        <div className="text-center py-12">
          <p className="text-gray-500">Please upload a dataset first to learn causal structure.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="card-header">Causal Structure Learning</h2>
          <div className="flex items-center space-x-4">
            <select
              value={learningMethod}
              onChange={(e) => setLearningMethod(e.target.value)}
              className="select w-48"
            >
              <option value="hybrid">Hybrid (Recommended)</option>
              <option value="pc">PC Algorithm</option>
              <option value="ges">Greedy Equivalence Search</option>
              <option value="notears">NOTEARS</option>
              <option value="bayesian">Bayesian</option>
            </select>
            <button
              onClick={() => learnMutation.mutate()}
              disabled={learnMutation.isPending}
              className="btn btn-primary flex items-center space-x-2"
            >
              <Play size={18} />
              <span>{learnMutation.isPending ? 'Learning...' : 'Learn Structure'}</span>
            </button>
          </div>
        </div>

        {model && (
          <div className="grid grid-cols-4 gap-4 mb-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Nodes</p>
              <p className="text-xl font-bold">{model.graph?.n_nodes || 0}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Edges</p>
              <p className="text-xl font-bold">{model.graph?.n_edges || 0}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">BIC Score</p>
              <p className="text-xl font-bold">{model.bic?.toFixed(1) || 'N/A'}</p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <p className="text-xs text-gray-500">Log-Likelihood</p>
              <p className="text-xl font-bold">{model.log_likelihood?.toFixed(1) || 'N/A'}</p>
            </div>
          </div>
        )}
      </div>

      {model?.graph && (
        <div className="card">
          <h3 className="card-header">Learned Causal Graph</h3>
          <div style={{ height: 500 }}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              fitView
            >
              <Background />
              <Controls />
              <MiniMap />
            </ReactFlow>
          </div>
        </div>
      )}
    </div>
  );
}