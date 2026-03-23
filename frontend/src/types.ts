// Graph types
export interface Node {
  id: string;
  name?: string;
}

export interface Edge {
  source: string;
  target: string;
  coefficient?: number;
}

export interface CausalGraph {
  nodes: Node[];
  edges: Edge[];
  n_nodes?: number;
  n_edges?: number;
}

// Model types
export interface CausalModelResult {
  model_id: string;
  graph: CausalGraph;
  bic?: number;
  log_likelihood?: number;
  parameters?: Record<string, number>;
}

// Data types
export interface ValidationResult {
  is_valid: boolean;
  errors?: string[];
  warnings?: string[];
}

export interface DatasetInfo {
  dataset_id: string;
  filename: string;
  n_rows: number;
  n_columns: number;
  columns: string[];
}

// Intervention types
export interface InterventionRequest {
  model_id: string;
  intervention_variables: Record<string, number>;
  target_variables: string[];
  n_samples?: number;
  compute_confidence_intervals?: boolean;
}

export interface InterventionResult {
  model_id: string;
  intervention_formula: string;
  distribution_stats: Record<string, DistributionStats>;
  causal_effects?: Record<string, number>;
}

export interface DistributionStats {
  mean: number;
  std: number;
  median: number;
  q1: number;
  q3: number;
}

// Counterfactual types
export interface CounterfactualRequest {
  model_id: string;
  evidence: Record<string, number>;
  intervention: Record<string, number>;
  target_variable: string;
  n_samples?: number;
}

export interface CounterfactualResult {
  factual_value: number;
  counterfactual_value: number;
  explanation?: string;
}

// Optimization types
export interface OptimizationRequest {
  model_id: string;
  target_variable: string;
  objective: 'maximize' | 'minimize';
  intervention_variables: string[];
  constraints: Record<string, { min?: number; max?: number }>;
  budget?: number;
  method?: string;
}

export interface OptimizationResult {
  optimal_value: number;
  baseline_value: number;
  optimal_interventions: Record<string, number>;
  sensitivity?: Record<string, number>;
}

// Report types
export interface ReportConfig {
  model_id: string;
  report_type: 'full' | 'summary' | 'technical';
  include_sections: {
    methodology?: boolean;
    results?: boolean;
    sensitivity?: boolean;
    recommendations?: boolean;
    appendix?: boolean;
  };
  format?: string;
}

export interface ReportResult {
  report_id: string;
  model_id: string;
  created_at: string;
  sections: ReportSection[];
}

export interface ReportSection {
  title: string;
  content: string;
}