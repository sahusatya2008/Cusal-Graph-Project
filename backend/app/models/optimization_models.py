from __future__ import annotations

"""
Optimization Model Schemas for Policy Decision Making
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime


class OptimizationMethod(str, Enum):
    """Optimization methods"""
    GRADIENT_DESCENT = "gradient_descent"
    SLSQP = "slsqp"
    COBYLA = "cobyla"
    TRUST_CONSTR = "trust_constr"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BAYESIAN = "bayesian_optimization"
    CMA_ES = "cma_es"


class ConstraintType(str, Enum):
    """Types of constraints"""
    INEQUALITY = "inequality"
    EQUALITY = "equality"
    BOUND = "bound"


class UtilityType(str, Enum):
    """Types of utility functions"""
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    PIECEWISE = "piecewise"
    CUSTOM = "custom"


class Constraint(BaseModel):
    """Constraint definition for optimization"""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(..., description="Constraint name")
    constraint_type: ConstraintType = Field(..., description="Type of constraint")
    expression: str = Field(..., description="Mathematical expression for the constraint")
    lower_bound: Optional[float] = Field(default=None, description="Lower bound for variable")
    upper_bound: Optional[float] = Field(default=None, description="Upper bound for variable")
    variables: List[str] = Field(default_factory=list, description="Variables involved in constraint")
    value: Optional[float] = Field(default=None, description="Right-hand side value")
    tolerance: float = Field(default=1e-6, description="Tolerance for constraint satisfaction")
    lagrange_multiplier: Optional[float] = Field(default=None, description="Lagrange multiplier")


class UtilityFunction(BaseModel):
    """Utility function definition"""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(default="U", description="Utility function name")
    utility_type: UtilityType = Field(default=UtilityType.LINEAR, description="Type of utility function")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Weights for linear utility")
    quadratic_matrix: Optional[List[List[float]]] = Field(default=None, description="Quadratic matrix Q")
    linear_coefficients: Optional[Dict[str, float]] = Field(default=None, description="Linear coefficients b")
    expression: Optional[str] = Field(default=None, description="Custom utility expression")
    target_variables: List[str] = Field(default_factory=list, description="Variables in utility function")
    risk_aversion: float = Field(default=0.0, ge=0.0, description="Risk aversion coefficient")


class OptimizationRequest(BaseModel):
    """Request for policy optimization"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str = Field(..., description="Causal model identifier")
    action_variables: List[str] = Field(..., description="Controllable action variables")
    action_bounds: Dict[str, Tuple[float, float]] = Field(..., description="Bounds for each action variable")
    utility_function: UtilityFunction = Field(..., description="Utility function to maximize")
    target_variable: str = Field(..., description="Target outcome variable")
    constraints: List[Constraint] = Field(default_factory=list, description="Optimization constraints")
    method: OptimizationMethod = Field(default=OptimizationMethod.SLSQP, description="Optimization method")
    max_iterations: int = Field(default=1000, ge=1, description="Maximum iterations")
    tolerance: float = Field(default=1e-6, description="Convergence tolerance")
    n_samples: int = Field(default=1000, description="Samples for expected utility computation")
    handle_uncertainty: bool = Field(default=True, description="Whether to handle uncertainty")
    initial_values: Optional[Dict[str, float]] = Field(default=None, description="Initial values for action variables")
    is_multi_objective: bool = Field(default=False, description="Multi-objective optimization")
    secondary_objectives: Optional[List[UtilityFunction]] = Field(default=None, description="Secondary objectives")


class SensitivityAnalysis(BaseModel):
    """Sensitivity analysis for optimization results"""
    model_config = ConfigDict(protected_namespaces=())
    
    gradient_at_optimum: Dict[str, float] = Field(default_factory=dict, description="Gradient of objective at optimum")
    hessian_eigenvalues: Optional[List[float]] = Field(default=None, description="Eigenvalues of Hessian")
    is_local_maximum: bool = Field(default=True, description="Whether solution is a local maximum")
    action_sensitivity: Dict[str, float] = Field(default_factory=dict, description="Sensitivity of utility to each action")
    shadow_prices: Dict[str, float] = Field(default_factory=dict, description="Shadow prices for constraints")
    robustness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Robustness score")
    perturbation_results: List[Dict[str, Any]] = Field(default_factory=list, description="Perturbation results")
    expected_value_perfect_information: Optional[float] = Field(default=None, description="EVPI")


class OptimizationResult(BaseModel):
    """Result of policy optimization"""
    model_config = ConfigDict(protected_namespaces=())
    
    optimization_id: str = Field(..., description="Unique optimization identifier")
    model_id: str = Field(..., description="Causal model identifier")
    optimal_actions: Dict[str, float] = Field(..., description="Optimal action values")
    optimal_utility: float = Field(..., description="Maximum expected utility achieved")
    expected_outcome: Dict[str, float] = Field(..., description="Expected outcome under optimal policy")
    outcome_confidence_interval: Optional[Dict[str, Tuple[float, float]]] = Field(default=None, description="Confidence interval")
    constraint_values: Dict[str, float] = Field(default_factory=dict, description="Constraint values at optimum")
    constraints_satisfied: bool = Field(default=True, description="Whether all constraints are satisfied")
    active_constraints: List[str] = Field(default_factory=list, description="Active constraints at optimum")
    method_used: OptimizationMethod = Field(..., description="Method used")
    n_iterations: int = Field(..., description="Number of iterations")
    converged: bool = Field(..., description="Whether optimization converged")
    computation_time: float = Field(..., description="Computation time in seconds")
    sensitivity_analysis: Optional[SensitivityAnalysis] = Field(default=None, description="Sensitivity analysis results")
    objective_formulation: str = Field(..., description="Mathematical formulation of objective")
    lagrangian_formulation: Optional[str] = Field(default=None, description="Lagrangian formulation")
    pareto_front: Optional[List[Dict[str, float]]] = Field(default=None, description="Pareto front")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PolicyEvaluationRequest(BaseModel):
    """Request for evaluating a specific policy"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str = Field(..., description="Causal model identifier")
    policy_actions: Dict[str, float] = Field(..., description="Action values to evaluate")
    n_samples: int = Field(default=1000, description="Number of samples for evaluation")
    compute_confidence_intervals: bool = Field(default=True, description="Compute confidence intervals")
    utility_function: Optional[UtilityFunction] = Field(default=None, description="Utility function for evaluation")


class PolicyEvaluationResult(BaseModel):
    """Result of policy evaluation"""
    model_config = ConfigDict(protected_namespaces=())
    
    evaluation_id: str = Field(..., description="Evaluation identifier")
    model_id: str = Field(..., description="Causal model identifier")
    policy_actions: Dict[str, float] = Field(..., description="Actions evaluated")
    outcome_distribution: Dict[str, Dict[str, float]] = Field(..., description="Distribution statistics")
    expected_utility: Optional[float] = Field(default=None, description="Expected utility")
    utility_confidence_interval: Optional[Tuple[float, float]] = Field(default=None, description="Utility CI")
    outcome_samples: Optional[Dict[str, List[float]]] = Field(default=None, description="Outcome samples")
    utility_gap: Optional[float] = Field(default=None, description="Gap from optimal utility")
    computation_time: float = Field(..., description="Computation time")


class MultiPolicyComparison(BaseModel):
    """Comparison of multiple policies"""
    model_config = ConfigDict(protected_namespaces=())
    
    comparison_id: str = Field(..., description="Comparison identifier")
    model_id: str = Field(..., description="Causal model identifier")
    policies: List[Dict[str, float]] = Field(..., description="Policies compared")
    policy_names: List[str] = Field(..., description="Policy names")
    expected_utilities: List[float] = Field(..., description="Expected utilities")
    utility_ranking: List[int] = Field(..., description="Ranking by utility")
    outcome_comparison: Dict[str, List[Dict[str, float]]] = Field(..., description="Outcome statistics")
    pairwise_tests: Optional[Dict[str, Dict[str, float]]] = Field(default=None, description="Pairwise tests")
    dominated_policies: List[int] = Field(default_factory=list, description="Dominated policies")