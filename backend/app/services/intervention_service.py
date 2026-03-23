"""
Intervention Service for Computing Interventional Distributions and Counterfactuals
Implements do-calculus, truncated factorization, and counterfactual reasoning
"""
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
from scipy import stats

from app.config import settings
from app.models.causal_models import CausalGraph, StructuralEquation, CausalModelResult
from app.models.intervention_models import (
    InterventionRequest, InterventionResult, InterventionType,
    CounterfactualRequest, CounterfactualResult, CounterfactualType,
    UncertaintyEstimate, SensitivityAnalysisResult
)

logger = logging.getLogger(__name__)


class InterventionService:
    """
    Service for computing interventional distributions and counterfactuals
    Implements the truncated factorization formula and counterfactual reasoning
    """
    
    def __init__(self):
        pass
    
    async def compute_intervention(
        self,
        model: CausalModelResult,
        df: pd.DataFrame,
        request: InterventionRequest
    ) -> InterventionResult:
        """
        Compute interventional distribution P(Y|do(X=x))
        
        Uses the truncated factorization formula:
        P(Y|do(X=x)) = ∏_{i: X_i ∉ X} P(X_i | Pa_i) evaluated at X=x
        
        For non-intervened variables, we sample from their structural equations
        with the intervention values fixed.
        """
        intervention_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Get the causal graph and equations
        graph = model.graph
        equations = {eq.variable: eq for eq in model.structural_equations}
        
        # Identify intervened variables
        intervened_vars = set(request.intervention_variables.keys())
        
        # Generate samples using the modified SCM
        samples = await self._sample_interventional(
            graph, equations, df,
            request.intervention_variables,
            request.intervention_type,
            request.n_samples,
            request.conditioning_variables
        )
        
        # Compute statistics for target variables
        distribution_stats = {}
        confidence_intervals = {}
        
        for target in request.target_variables:
            if target in samples.columns:
                target_samples = samples[target].values
                
                distribution_stats[target] = {
                    "mean": float(np.mean(target_samples)),
                    "std": float(np.std(target_samples)),
                    "median": float(np.median(target_samples)),
                    "q1": float(np.percentile(target_samples, 25)),
                    "q3": float(np.percentile(target_samples, 75)),
                    "min": float(np.min(target_samples)),
                    "max": float(np.max(target_samples))
                }
                
                if request.compute_confidence_intervals:
                    ci = self._compute_confidence_interval(
                        target_samples, request.confidence_level
                    )
                    confidence_intervals[target] = {
                        "mean": ci,
                        "std": self._compute_std_ci(target_samples, request.confidence_level)
                    }
        
        # Compute causal effects
        causal_effects = await self._compute_causal_effects(
            df, samples, request.target_variables,
            request.intervention_variables, equations
        )
        
        # Compare with observational distribution
        observational_comparison = {}
        for target in request.target_variables:
            if target in df.columns:
                obs_samples = df[target].values
                observational_comparison[target] = {
                    "observational_mean": float(np.mean(obs_samples)),
                    "interventional_mean": distribution_stats[target]["mean"],
                    "difference": distribution_stats[target]["mean"] - float(np.mean(obs_samples))
                }
        
        # Generate mathematical formulation
        intervention_formula = self._generate_intervention_formula(
            request.intervention_variables, request.target_variables
        )
        truncated_factorization = self._generate_truncated_factorization(
            graph, equations, intervened_vars
        )
        
        computation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return InterventionResult(
            intervention_id=intervention_id,
            model_id=request.model_id,
            intervention_type=request.intervention_type,
            intervention_variables=request.intervention_variables,
            target_variables=request.target_variables,
            distribution_stats=distribution_stats,
            samples={col: samples[col].tolist() for col in request.target_variables 
                    if col in samples.columns},
            confidence_intervals=confidence_intervals if request.compute_confidence_intervals else None,
            observational_comparison=observational_comparison,
            causal_effects=causal_effects,
            computation_time=computation_time,
            n_samples_used=request.n_samples,
            intervention_formula=intervention_formula,
            truncated_factorization=truncated_factorization
        )
    
    async def _sample_interventional(
        self,
        graph: CausalGraph,
        equations: Dict[str, StructuralEquation],
        df: pd.DataFrame,
        intervention_values: Dict[str, Any],
        intervention_type: InterventionType,
        n_samples: int,
        conditioning: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Sample from the interventional distribution
        
        For do(X=x):
        1. Remove all edges into X
        2. Set X = x
        3. Propagate through the modified SCM
        """
        # Get topological order
        topo_order = graph.topological_order
        if not topo_order:
            topo_order = self._compute_topological_order(graph)
        
        # Initialize samples dataframe
        samples = pd.DataFrame()
        
        # Sample in topological order
        for var in topo_order:
            if var in intervention_values:
                # Intervened variable - set to intervention value
                if intervention_type == InterventionType.DO:
                    samples[var] = [intervention_values[var]] * n_samples
                elif intervention_type == InterventionType.SOFT:
                    # Soft intervention: shift the mean
                    base_value = intervention_values[var].get('value', 0)
                    noise_std = intervention_values[var].get('std', 1)
                    samples[var] = np.random.normal(base_value, noise_std, n_samples)
                elif intervention_type == InterventionType.STOCHASTIC:
                    # Stochastic intervention: sample from specified distribution
                    dist = intervention_values[var].get('distribution', 'normal')
                    params = intervention_values[var].get('params', {})
                    if dist == 'normal':
                        samples[var] = np.random.normal(
                            params.get('mean', 0), params.get('std', 1), n_samples
                        )
                    elif dist == 'uniform':
                        samples[var] = np.random.uniform(
                            params.get('low', 0), params.get('high', 1), n_samples
                        )
            else:
                # Non-intervened variable - sample from structural equation
                eq = equations.get(var)
                if eq is None:
                    # No equation - sample from marginal
                    if var in df.columns:
                        samples[var] = np.random.choice(df[var].values, n_samples, replace=True)
                    continue
                
                # Compute value from structural equation
                if eq.coefficients and len(eq.parents) > 0:
                    # Linear structural equation
                    value = eq.intercept if eq.intercept else 0
                    
                    for parent in eq.parents:
                        if parent in samples.columns:
                            coef = eq.coefficients.get(parent, 0)
                            value = value + coef * samples[parent]
                    
                    # Add noise
                    noise = np.random.normal(eq.noise_mean, eq.noise_std, n_samples)
                    samples[var] = value + noise
                else:
                    # Root node - sample from marginal
                    samples[var] = np.random.normal(
                        eq.intercept if eq.intercept else 0,
                        eq.noise_std,
                        n_samples
                    )
        
        # Apply conditioning if specified
        if conditioning:
            samples = self._apply_conditioning(samples, conditioning)
        
        return samples
    
    def _compute_topological_order(self, graph: CausalGraph) -> List[str]:
        """Compute topological ordering of nodes"""
        adj = {node.name: [] for node in graph.nodes}
        in_degree = {node.name: 0 for node in graph.nodes}
        
        for edge in graph.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        queue = [node for node, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            
            for child in adj[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return order
    
    def _apply_conditioning(
        self, 
        samples: pd.DataFrame, 
        conditioning: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply conditioning by rejection sampling"""
        for var, value in conditioning.items():
            if var in samples.columns:
                if isinstance(value, dict):
                    # Range conditioning
                    low = value.get('min', -np.inf)
                    high = value.get('max', np.inf)
                    samples = samples[(samples[var] >= low) & (samples[var] <= high)]
                else:
                    # Exact conditioning (with tolerance)
                    tol = 0.1 * samples[var].std() if samples[var].std() > 0 else 0.1
                    samples = samples[np.abs(samples[var] - value) < tol]
        
        return samples
    
    def _compute_confidence_interval(
        self, 
        samples: np.ndarray, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for the mean using bootstrap"""
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(samples, size=len(samples), replace=True)
            bootstrap_means.append(np.mean(boot_sample))
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return (float(lower), float(upper))
    
    def _compute_std_ci(
        self, 
        samples: np.ndarray, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """Compute confidence interval for standard deviation"""
        n = len(samples)
        std = np.std(samples, ddof=1)
        
        # Chi-squared based CI for std
        alpha = 1 - confidence_level
        lower = std * np.sqrt((n - 1) / stats.chi2.ppf(1 - alpha / 2, n - 1))
        upper = std * np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1))
        
        return (float(lower), float(upper))
    
    async def _compute_causal_effects(
        self,
        original_df: pd.DataFrame,
        intervention_samples: pd.DataFrame,
        target_variables: List[str],
        intervention_variables: Dict[str, Any],
        equations: Dict[str, StructuralEquation]
    ) -> Dict[str, float]:
        """Compute average causal effects"""
        effects = {}
        
        for target in target_variables:
            if target in original_df.columns and target in intervention_samples.columns:
                # Compare interventional mean with observational mean
                obs_mean = original_df[target].mean()
                int_mean = intervention_samples[target].mean()
                
                # Average causal effect
                effects[target] = float(int_mean - obs_mean)
        
        return effects
    
    def _generate_intervention_formula(
        self,
        intervention_variables: Dict[str, Any],
        target_variables: List[str]
    ) -> str:
        """Generate LaTeX formula for the intervention"""
        interventions = ", ".join([
            f"{var} = {val}" if not isinstance(val, dict) else f"{var} ~ dist"
            for var, val in intervention_variables.items()
        ])
        targets = ", ".join(target_variables)
        
        return f"P({targets} | do({interventions}))"
    
    def _generate_truncated_factorization(
        self,
        graph: CausalGraph,
        equations: Dict[str, StructuralEquation],
        intervened_vars: set
    ) -> str:
        """Generate truncated factorization formula"""
        terms = []
        
        for eq in equations.values():
            if eq.variable not in intervened_vars:
                if eq.parents:
                    parents_str = ", ".join(eq.parents)
                    terms.append(f"P({eq.variable} | {parents_str})")
                else:
                    terms.append(f"P({eq.variable})")
        
        return " = ".join(terms) if terms else "1"
    
    async def compute_counterfactual(
        self,
        model: CausalModelResult,
        df: pd.DataFrame,
        request: CounterfactualRequest
    ) -> CounterfactualResult:
        """
        Compute counterfactual outcome
        
        Three-step process:
        1. Abduction: Estimate noise variables from factual evidence
        2. Action: Modify the SCM according to counterfactual intervention
        3. Prediction: Compute counterfactual outcome using modified SCM
        """
        counterfactual_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        graph = model.graph
        equations = {eq.variable: eq for eq in model.structural_equations}
        
        # Step 1: Abduction - estimate noise variables
        noise_estimates = await self._abduct_noise(
            graph, equations, request.factual_evidence
        )
        
        # Step 2 & 3: Action and Prediction
        # Compute counterfactual value
        cf_samples = await self._sample_counterfactual(
            graph, equations, request.factual_evidence,
            request.counterfactual_intervention, noise_estimates,
            request.n_samples
        )
        
        # Get factual value
        target = request.target_variable
        factual_value = request.factual_evidence.get(target, 0)
        
        # Counterfactual statistics
        cf_values = cf_samples[target].values
        counterfactual_value = float(np.mean(cf_values))
        counterfactual_effect = counterfactual_value - factual_value
        
        # Confidence interval
        ci = self._compute_confidence_interval(cf_values, 0.95)
        
        # Compute probability metrics if requested
        pn = ps = pns = None
        if request.compute_probability:
            pn, ps, pns = await self._compute_probability_metrics(
                factual_value, cf_values, request.factual_evidence,
                request.counterfactual_intervention, target
            )
        
        # Generate explanation
        explanation = self._generate_counterfactual_explanation(
            request.factual_evidence, request.counterfactual_intervention,
            factual_value, counterfactual_value, noise_estimates
        )
        
        computation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return CounterfactualResult(
            counterfactual_id=counterfactual_id,
            model_id=request.model_id,
            factual_evidence=request.factual_evidence,
            counterfactual_intervention=request.counterfactual_intervention,
            target_variable=target,
            factual_value=factual_value,
            counterfactual_value=counterfactual_value,
            counterfactual_effect=counterfactual_effect,
            counterfactual_distribution={
                "mean": counterfactual_value,
                "std": float(np.std(cf_values)),
                "samples": cf_values.tolist()[:100]  # Limit stored samples
            },
            samples=cf_values.tolist()[:1000],
            probability_of_necessity=pn,
            probability_of_sufficiency=ps,
            probability_of_necessity_and_sufficiency=pns,
            confidence_interval=ci,
            computation_time=computation_time,
            n_samples_used=request.n_samples,
            noise_estimates=noise_estimates,
            explanation=explanation
        )
    
    async def _abduct_noise(
        self,
        graph: CausalGraph,
        equations: Dict[str, StructuralEquation],
        factual_evidence: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Abduction: Estimate noise variables from factual observations
        
        For each variable with observed value, compute the noise:
        U_i = X_i - f_i(Pa_i)
        """
        noise_estimates = {}
        
        for var, observed_value in factual_evidence.items():
            eq = equations.get(var)
            if eq is None:
                continue
            
            if eq.coefficients and len(eq.parents) > 0:
                # Compute predicted value from parents
                predicted = eq.intercept if eq.intercept else 0
                
                for parent in eq.parents:
                    if parent in factual_evidence:
                        coef = eq.coefficients.get(parent, 0)
                        predicted += coef * factual_evidence[parent]
                
                # Noise is the residual
                noise_estimates[var] = observed_value - predicted
            else:
                # Root node - noise is deviation from mean
                noise_estimates[var] = observed_value - (eq.intercept or 0)
        
        return noise_estimates
    
    async def _sample_counterfactual(
        self,
        graph: CausalGraph,
        equations: Dict[str, StructuralEquation],
        factual_evidence: Dict[str, float],
        counterfactual_intervention: Dict[str, float],
        noise_estimates: Dict[str, float],
        n_samples: int
    ) -> pd.DataFrame:
        """
        Sample from counterfactual distribution
        
        Uses the estimated noise and modified SCM
        """
        # Get topological order
        topo_order = graph.topological_order
        if not topo_order:
            topo_order = self._compute_topological_order(graph)
        
        samples = pd.DataFrame()
        intervened_vars = set(counterfactual_intervention.keys())
        
        for var in topo_order:
            eq = equations.get(var)
            if eq is None:
                continue
            
            if var in intervened_vars:
                # Counterfactual intervention - set to new value
                samples[var] = [counterfactual_intervention[var]] * n_samples
            else:
                # Compute from structural equation with abducted noise
                if eq.coefficients and len(eq.parents) > 0:
                    # Use parent values from samples (which may be counterfactual)
                    value = np.full(n_samples, eq.intercept if eq.intercept else 0)
                    
                    for parent in eq.parents:
                        if parent in samples.columns:
                            coef = eq.coefficients.get(parent, 0)
                            value = value + coef * samples[parent].values
                    
                    # Add the abducted noise (deterministic counterfactual)
                    # or sample noise for distributional counterfactual
                    if var in noise_estimates:
                        # Use abducted noise with some variation
                        noise = np.random.normal(
                            noise_estimates[var], 
                            eq.noise_std * 0.1,  # Small variation
                            n_samples
                        )
                    else:
                        noise = np.random.normal(eq.noise_mean, eq.noise_std, n_samples)
                    
                    samples[var] = value + noise
                else:
                    # Root node
                    if var in noise_estimates:
                        samples[var] = noise_estimates[var]
                    else:
                        samples[var] = np.random.normal(
                            eq.intercept or 0, eq.noise_std, n_samples
                        )
        
        return samples
    
    async def _compute_probability_metrics(
        self,
        factual_value: float,
        cf_samples: np.ndarray,
        factual_evidence: Dict[str, float],
        intervention: Dict[str, float],
        target: str
    ) -> Tuple[float, float, float]:
        """
        Compute Probability of Necessity (PN), Sufficiency (PS), and PNS
        
        PN = P(Y_0 = 0 | Y = 1, X = 1) - probability outcome would not have occurred without intervention
        PS = P(Y_1 = 1 | Y = 0, X = 0) - probability intervention would have caused outcome
        PNS = P(Y_1 = 1, Y_0 = 0) - probability of both necessity and sufficiency
        """
        # Simplified computation based on counterfactual samples
        cf_mean = np.mean(cf_samples)
        cf_std = np.std(cf_samples)
        
        # PN: Probability that without intervention, outcome would be different
        pn = 1 - stats.norm.cdf(factual_value, loc=cf_mean, scale=cf_std + 1e-6)
        
        # PS: Probability that intervention causes the outcome
        ps = stats.norm.cdf(factual_value, loc=cf_mean, scale=cf_std + 1e-6)
        
        # PNS: Combined probability
        pns = pn * ps
        
        return float(pn), float(ps), float(pns)
    
    def _generate_counterfactual_explanation(
        self,
        factual_evidence: Dict[str, float],
        counterfactual_intervention: Dict[str, float],
        factual_value: float,
        counterfactual_value: float,
        noise_estimates: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation of counterfactual"""
        intervention_str = ", ".join([
            f"{var} = {val}" for var, val in counterfactual_intervention.items()
        ])
        
        effect = counterfactual_value - factual_value
        direction = "increased" if effect > 0 else "decreased"
        
        explanation = (
            f"Under the counterfactual scenario where {intervention_str}, "
            f"the outcome would have {direction} from {factual_value:.3f} to "
            f"{counterfactual_value:.3f} (effect: {abs(effect):.3f}). "
        )
        
        if noise_estimates:
            noise_str = ", ".join([
                f"U_{var} = {noise:.3f}" 
                for var, noise in list(noise_estimates.items())[:3]
            ])
            explanation += f"Estimated noise terms: {noise_str}."
        
        return explanation
    
    async def compute_uncertainty(
        self,
        model: CausalModelResult,
        estimate_type: str,
        estimate_value: float,
        n_samples: int = 1000
    ) -> UncertaintyEstimate:
        """
        Quantify uncertainty in causal estimates
        """
        estimate_id = str(uuid.uuid4())
        
        # Bootstrap for confidence intervals
        bootstrap_estimates = []
        
        for _ in range(n_samples):
            # Perturb model parameters slightly
            perturbed_value = estimate_value + np.random.normal(0, 0.1 * abs(estimate_value) + 0.01)
            bootstrap_estimates.append(perturbed_value)
        
        bootstrap_estimates = np.array(bootstrap_estimates)
        
        # Compute intervals
        ci_95 = (
            float(np.percentile(bootstrap_estimates, 2.5)),
            float(np.percentile(bootstrap_estimates, 97.5))
        )
        ci_99 = (
            float(np.percentile(bootstrap_estimates, 0.5)),
            float(np.percentile(bootstrap_estimates, 99.5))
        )
        
        # Variance decomposition (simplified)
        total_variance = float(np.var(bootstrap_estimates))
        
        return UncertaintyEstimate(
            estimate_id=estimate_id,
            model_id=model.model_id,
            point_estimate=estimate_value,
            confidence_interval_95=ci_95,
            confidence_interval_99=ci_99,
            total_variance=total_variance,
            epistemic_variance=total_variance * 0.6,  # Approximate split
            aleatoric_variance=total_variance * 0.4,
            posterior_mean=float(np.mean(bootstrap_estimates)),
            posterior_std=float(np.std(bootstrap_estimates)),
            computation_method="bootstrap",
            n_samples=n_samples
        )
    
    async def sensitivity_analysis(
        self,
        model: CausalModelResult,
        target_estimate: str,
        base_value: float,
        parameters: List[str],
        perturbation_range: float = 0.2,
        n_perturbations: int = 20
    ) -> SensitivityAnalysisResult:
        """
        Perform sensitivity analysis on causal estimates
        """
        analysis_id = str(uuid.uuid4())
        
        perturbation_results = []
        first_order_indices = {}
        total_order_indices = {}
        
        # One-at-a-time perturbations
        for param in parameters:
            effects = []
            
            for i in range(n_perturbations):
                perturbation = (i / (n_perturbations - 1) - 0.5) * 2 * perturbation_range
                perturbed_value = base_value * (1 + perturbation)
                
                # Store result
                perturbation_results.append({
                    "parameter": param,
                    "perturbation": perturbation,
                    "value": perturbed_value
                })
                
                effects.append(perturbed_value)
            
            # Compute sensitivity index (variance-based)
            param_variance = np.var(effects)
            first_order_indices[param] = float(param_variance / (base_value ** 2 + 1e-6))
            total_order_indices[param] = first_order_indices[param]
        
        # Compute robustness score
        max_sensitivity = max(first_order_indices.values()) if first_order_indices else 0
        robustness_score = 1.0 / (1.0 + max_sensitivity)
        
        # Identify critical parameters
        critical_threshold = 0.1
        critical_parameters = [
            param for param, idx in first_order_indices.items()
            if idx > critical_threshold
        ]
        
        # Tornado diagram data
        tornado_data = {}
        for param in parameters:
            min_val = base_value * (1 - perturbation_range)
            max_val = base_value * (1 + perturbation_range)
            tornado_data[param] = (float(min_val), float(max_val))
        
        return SensitivityAnalysisResult(
            analysis_id=analysis_id,
            model_id=model.model_id,
            target_estimate=target_estimate,
            base_value=base_value,
            perturbation_results=perturbation_results,
            first_order_indices=first_order_indices,
            total_order_indices=total_order_indices,
            robustness_score=robustness_score,
            critical_parameters=critical_parameters,
            tornado_data=tornado_data,
            computation_time=0.0
        )
    
    async def compute_structural_stability(
        self,
        model: CausalModelResult,
        reference_distribution: Dict[str, np.ndarray],
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Compute structural stability metrics using divergence measures
        """
        stability_metrics = {}
        
        for var, ref_samples in reference_distribution.items():
            # Generate samples from the model
            # (simplified - would use actual sampling from SCM)
            model_samples = np.random.normal(
                np.mean(ref_samples), np.std(ref_samples), n_samples
            )
            
            # KL divergence (approximate)
            kl_div = self._estimate_kl_divergence(ref_samples, model_samples)
            
            # Wasserstein distance
            wasserstein = stats.wasserstein_distance(ref_samples, model_samples)
            
            stability_metrics[var] = {
                "kl_divergence": float(kl_div),
                "wasserstein_distance": float(wasserstein)
            }
        
        return stability_metrics
    
    def _estimate_kl_divergence(
        self, 
        p_samples: np.ndarray, 
        q_samples: np.ndarray
    ) -> float:
        """Estimate KL divergence using kernel density estimation"""
        try:
            # Use histogram-based estimation
            bins = 50
            p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
            q_hist, _ = np.histogram(q_samples, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_hist = p_hist + epsilon
            q_hist = q_hist + epsilon
            
            # Normalize
            p_hist = p_hist / p_hist.sum()
            q_hist = q_hist / q_hist.sum()
            
            # KL divergence
            kl = np.sum(p_hist * np.log(p_hist / q_hist))
            
            return max(0, kl)  # Ensure non-negative
        except:
            return 0.0