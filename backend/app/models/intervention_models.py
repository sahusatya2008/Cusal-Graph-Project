"""
Intervention and Counterfactual Model Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


class InterventionType(str, Enum):
    """Types of interventions"""
    DO = "do"  # Perfect intervention: do(X=x)
    SOFT = "soft"  # Soft intervention: shift in distribution
    POLICY = "policy"  # Policy intervention: conditional intervention
    STOCHASTIC = "stochastic"  # Stochastic intervention


class CounterfactualType(str, Enum):
    """Types of counterfactual queries"""
    POINT = "point"  # Point counterfactual
    DISTRIBUTIONAL = "distributional"  # Distributional counterfactual
    PROBABILITY = "probability"  # Probability of necessity/sufficiency


class InterventionRequest(BaseModel):
    """Request for computing interventional distribution"""
    model_id: str = Field(..., description="Causal model identifier")
    
    # Intervention specification
    intervention_type: InterventionType = Field(
        default=InterventionType.DO,
        description="Type of intervention"
    )
    intervention_variables: Dict[str, Any] = Field(
        ...,
        description="Variables and values to intervene on {var: value}"
    )
    
    # Target variables
    target_variables: List[str] = Field(
        ...,
        description="Variables to compute distribution for"
    )
    
    # Computation options
    n_samples: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Number of samples for Monte Carlo estimation"
    )
    compute_confidence_intervals: bool = Field(
        default=True,
        description="Whether to compute confidence intervals"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Confidence level for intervals"
    )
    
    # Conditioning (optional)
    conditioning_variables: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Variables to condition on"
    )


class InterventionResult(BaseModel):
    """Result of intervention computation"""
    intervention_id: str = Field(..., description="Unique intervention identifier")
    model_id: str = Field(..., description="Causal model identifier")
    
    # Intervention specification
    intervention_type: InterventionType = Field(..., description="Type of intervention")
    intervention_variables: Dict[str, Any] = Field(..., description="Intervention specification")
    target_variables: List[str] = Field(..., description="Target variables")
    
    # Results - Distribution statistics
    distribution_stats: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Statistics for each target variable distribution"
    )
    # Format: {var: {"mean": ..., "std": ..., "median": ..., "q1": ..., "q3": ...}}
    
    # Samples
    samples: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Monte Carlo samples for each target variable"
    )
    
    # Confidence intervals
    confidence_intervals: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = Field(
        default=None,
        description="Confidence intervals for each variable"
    )
    # Format: {var: {"mean": (lower, upper), "std": (lower, upper)}}
    
    # Comparison with observational
    observational_comparison: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Comparison with observational distribution"
    )
    
    # Causal effect estimates
    causal_effects: Optional[Dict[str, float]] = Field(
        default=None,
        description="Average causal effect for each target"
    )
    
    # Computation metadata
    computation_time: float = Field(..., description="Computation time in seconds")
    n_samples_used: int = Field(..., description="Number of samples used")
    
    # Mathematical formulation
    intervention_formula: str = Field(
        ...,
        description="Mathematical formula for the intervention"
    )
    truncated_factorization: str = Field(
        ...,
        description="Truncated factorization formula"
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchInterventionRequest(BaseModel):
    """Request for batch intervention analysis"""
    model_id: str = Field(..., description="Causal model identifier")
    
    # Multiple interventions
    interventions: List[Dict[str, Any]] = Field(
        ...,
        description="List of intervention specifications"
    )
    target_variables: List[str] = Field(..., description="Target variables")
    
    # Options
    n_samples: int = Field(default=1000, description="Samples per intervention")
    parallel: bool = Field(default=True, description="Run in parallel")


class BatchInterventionResult(BaseModel):
    """Result of batch intervention analysis"""
    batch_id: str = Field(..., description="Batch identifier")
    model_id: str = Field(..., description="Causal model identifier")
    
    # Individual results
    results: List[InterventionResult] = Field(..., description="Individual intervention results")
    
    # Summary statistics
    effect_comparison: Dict[str, List[float]] = Field(
        ...,
        description="Comparison of effects across interventions"
    )
    
    # Computation metadata
    total_computation_time: float = Field(..., description="Total computation time")
    n_interventions: int = Field(..., description="Number of interventions computed")


class CounterfactualRequest(BaseModel):
    """Request for counterfactual reasoning"""
    model_id: str = Field(..., description="Causal model identifier")
    
    # Factual observation
    factual_evidence: Dict[str, float] = Field(
        ...,
        description="Observed factual values {var: value}"
    )
    
    # Counterfactual intervention
    counterfactual_intervention: Dict[str, float] = Field(
        ...,
        description="Counterfactual intervention {var: value}"
    )
    
    # Target variable
    target_variable: str = Field(..., description="Variable to compute counterfactual for")
    
    # Computation options
    n_samples: int = Field(default=1000, description="Number of counterfactual samples")
    compute_probability: bool = Field(
        default=False,
        description="Compute probability of counterfactual"
    )
    counterfactual_type: CounterfactualType = Field(
        default=CounterfactualType.POINT,
        description="Type of counterfactual query"
    )


class CounterfactualResult(BaseModel):
    """Result of counterfactual computation"""
    counterfactual_id: str = Field(..., description="Unique counterfactual identifier")
    model_id: str = Field(..., description="Causal model identifier")
    
    # Input specification
    factual_evidence: Dict[str, float] = Field(..., description="Factual observations")
    counterfactual_intervention: Dict[str, float] = Field(..., description="Counterfactual intervention")
    target_variable: str = Field(..., description="Target variable")
    
    # Results
    factual_value: float = Field(..., description="Factual value of target")
    counterfactual_value: float = Field(..., description="Counterfactual value of target")
    counterfactual_effect: float = Field(..., description="Counterfactual effect (CF - Factual)")
    
    # Distribution (if distributional)
    counterfactual_distribution: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full counterfactual distribution"
    )
    
    # Samples
    samples: Optional[List[float]] = Field(
        default=None,
        description="Counterfactual samples"
    )
    
    # Probability metrics
    probability_of_necessity: Optional[float] = Field(
        default=None,
        description="Probability of necessity (PN)"
    )
    probability_of_sufficiency: Optional[float] = Field(
        default=None,
        description="Probability of sufficiency (PS)"
    )
    probability_of_necessity_and_sufficiency: Optional[float] = Field(
        default=None,
        description="Probability of necessity and sufficiency (PNS)"
    )
    
    # Uncertainty
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Confidence interval for counterfactual effect"
    )
    
    # Computation metadata
    computation_time: float = Field(..., description="Computation time in seconds")
    n_samples_used: int = Field(..., description="Number of samples used")
    
    # Explanation
    noise_estimates: Dict[str, float] = Field(
        ...,
        description="Estimated noise values for each variable"
    )
    explanation: str = Field(..., description="Human-readable explanation")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UncertaintyEstimate(BaseModel):
    """Uncertainty quantification for causal estimates"""
    estimate_id: str = Field(..., description="Unique estimate identifier")
    model_id: str = Field(..., description="Causal model identifier")
    
    # Point estimate
    point_estimate: float = Field(..., description="Point estimate value")
    
    # Interval estimates
    confidence_interval_95: Tuple[float, float] = Field(
        ...,
        description="95% confidence interval"
    )
    confidence_interval_99: Tuple[float, float] = Field(
        ...,
        description="99% confidence interval"
    )
    credible_interval_95: Optional[Tuple[float, float]] = Field(
        default=None,
        description="95% Bayesian credible interval"
    )
    
    # Variance components
    total_variance: float = Field(..., description="Total variance")
    epistemic_variance: Optional[float] = Field(
        default=None,
        description="Epistemic (model) uncertainty"
    )
    aleatoric_variance: Optional[float] = Field(
        default=None,
        description="Aleatoric (data) uncertainty"
    )
    
    # Distribution information
    posterior_mean: Optional[float] = Field(default=None, description="Posterior mean")
    posterior_std: Optional[float] = Field(default=None, description="Posterior std")
    posterior_samples: Optional[List[float]] = Field(
        default=None,
        description="Posterior samples"
    )
    
    # Stability metrics
    kl_divergence: Optional[float] = Field(
        default=None,
        description="KL divergence from reference"
    )
    wasserstein_distance: Optional[float] = Field(
        default=None,
        description="Wasserstein distance from reference"
    )
    
    # Sensitivity analysis
    sensitivity_indices: Optional[Dict[str, float]] = Field(
        default=None,
        description="Sensitivity indices for each parameter"
    )
    
    computation_method: str = Field(
        ...,
        description="Method used for uncertainty quantification"
    )
    n_samples: int = Field(..., description="Number of samples used")


class SensitivityAnalysisResult(BaseModel):
    """Result of sensitivity analysis for causal estimates"""
    analysis_id: str = Field(..., description="Analysis identifier")
    model_id: str = Field(..., description="Causal model identifier")
    
    # Target estimate
    target_estimate: str = Field(..., description="Target causal estimate")
    base_value: float = Field(..., description="Base estimate value")
    
    # Parameter perturbations
    perturbation_results: List[Dict[str, Any]] = Field(
        ...,
        description="Results under parameter perturbations"
    )
    
    # Sensitivity indices
    first_order_indices: Dict[str, float] = Field(
        ...,
        description="First-order Sobol indices"
    )
    total_order_indices: Dict[str, float] = Field(
        ...,
        description="Total-order Sobol indices"
    )
    
    # Robustness metrics
    robustness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall robustness score"
    )
    critical_parameters: List[str] = Field(
        ...,
        description="Parameters with high sensitivity"
    )
    
    # Visualization data
    tornado_data: Dict[str, Tuple[float, float]] = Field(
        ...,
        description="Data for tornado diagram"
    )
    
    computation_time: float = Field(..., description="Computation time in seconds")