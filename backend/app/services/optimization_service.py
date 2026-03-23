"""
Optimization Service for Policy Decision Making
Implements expected utility maximization under constraints
"""
from typing import Optional, List, Dict, Any, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
from scipy import optimize
from scipy.stats import norm
import warnings

from app.config import settings
from app.models.causal_models import CausalModelResult, StructuralEquation
from app.models.optimization_models import (
    OptimizationRequest, OptimizationResult, OptimizationMethod,
    Constraint, ConstraintType, UtilityFunction, UtilityType,
    SensitivityAnalysis, PolicyEvaluationRequest, PolicyEvaluationResult
)

logger = logging.getLogger(__name__)


class OptimizationService:
    """
    Service for policy optimization under constraints
    Solves: max_a E[U(Y)|do(A=a)] subject to g(a) <= c
    """
    
    def __init__(self):
        pass
    
    async def optimize_policy(
        self,
        model: CausalModelResult,
        df: pd.DataFrame,
        request: OptimizationRequest
    ) -> OptimizationResult:
        """
        Optimize policy by maximizing expected utility under constraints
        
        The optimization problem:
        max_a E[U(Y) | do(A=a)]
        subject to: g_i(a) <= c_i for all constraints
        
        Uses the causal model to compute interventional expectations
        """
        optimization_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # Get model components
        graph = model.graph
        equations = {eq.variable: eq for eq in model.structural_equations}
        
        # Build objective function
        def objective(actions: np.ndarray) -> float:
            """Compute negative expected utility (for minimization)"""
            action_dict = self._array_to_action_dict(
                actions, request.action_variables
            )
            expected_utility = self._compute_expected_utility(
                action_dict, equations, request.utility_function,
                request.target_variable, request.n_samples
            )
            return -expected_utility  # Negative for minimization
        
        # Build constraints
        constraints_list = self._build_constraints(request)
        
        # Build bounds
        bounds = self._build_bounds(request.action_variables, request.action_bounds)
        
        # Initial guess
        x0 = self._get_initial_guess(request)
        
        # Choose optimization method
        if request.method in [OptimizationMethod.SLSQP, OptimizationMethod.TRUST_CONSTR]:
            result = await self._scipy_optimize(
                objective, x0, bounds, constraints_list, request
            )
        elif request.method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
            result = await self._differential_evolution(
                objective, bounds, constraints_list, request
            )
        else:
            result = await self._scipy_optimize(
                objective, x0, bounds, constraints_list, request
            )
        
        # Extract optimal actions
        optimal_actions = self._array_to_action_dict(
            result['x'], request.action_variables
        )
        
        # Compute expected outcome under optimal policy
        expected_outcome = await self._compute_expected_outcome(
            optimal_actions, equations, request.target_variable, request.n_samples
        )
        
        # Compute confidence intervals
        outcome_ci = await self._compute_outcome_confidence_interval(
            optimal_actions, equations, request.target_variable, request.n_samples
        )
        
        # Evaluate constraints at optimum
        constraint_values = self._evaluate_constraints(
            result['x'], request.constraints
        )
        
        # Identify active constraints
        active_constraints = [
            c.name for i, c in enumerate(request.constraints)
            if abs(constraint_values.get(c.name, 0) - (c.value or 0)) < 1e-4
        ]
        
        # Sensitivity analysis
        sensitivity = await self._compute_sensitivity(
            result['x'], objective, request.action_variables
        )
        
        # Generate mathematical formulation
        objective_formulation = self._generate_objective_formulation(request)
        lagrangian = self._generate_lagrangian(request, result)
        
        computation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return OptimizationResult(
            optimization_id=optimization_id,
            model_id=request.model_id,
            optimal_actions=optimal_actions,
            optimal_utility=-result['fun'],
            expected_outcome=expected_outcome,
            outcome_confidence_interval=outcome_ci,
            constraint_values=constraint_values,
            constraints_satisfied=result['success'],
            active_constraints=active_constraints,
            method_used=request.method,
            n_iterations=result.get('nit', 0),
            converged=result['success'],
            computation_time=computation_time,
            sensitivity_analysis=sensitivity,
            objective_formulation=objective_formulation,
            lagrangian_formulation=lagrangian
        )
    
    def _array_to_action_dict(
        self, 
        actions: np.ndarray, 
        action_vars: List[str]
    ) -> Dict[str, float]:
        """Convert action array to dictionary"""
        return {var: float(actions[i]) for i, var in enumerate(action_vars)}
    
    def _compute_expected_utility(
        self,
        actions: Dict[str, float],
        equations: Dict[str, StructuralEquation],
        utility_func: UtilityFunction,
        target_variable: str,
        n_samples: int
    ) -> float:
        """
        Compute expected utility under intervention do(A=actions)
        
        E[U(Y) | do(A=a)] = ∫ U(y) P(y | do(a)) dy
        """
        # Sample outcomes under intervention
        outcomes = self._sample_outcomes(actions, equations, target_variable, n_samples)
        
        # Compute utility for each sample
        utilities = self._compute_utility(outcomes, utility_func)
        
        return float(np.mean(utilities))
    
    def _sample_outcomes(
        self,
        actions: Dict[str, float],
        equations: Dict[str, StructuralEquation],
        target_variable: str,
        n_samples: int
    ) -> np.ndarray:
        """Sample outcomes under intervention"""
        # Get topological order
        topo_order = self._get_topological_order(equations)
        
        samples = {}
        
        # Set action variables
        for var, value in actions.items():
            samples[var] = np.full(n_samples, value)
        
        # Sample in topological order
        for var in topo_order:
            if var in actions:
                continue
            
            eq = equations.get(var)
            if eq is None:
                continue
            
            if eq.coefficients and len(eq.parents) > 0:
                value = np.full(n_samples, eq.intercept or 0)
                
                for parent in eq.parents:
                    if parent in samples:
                        coef = eq.coefficients.get(parent, 0)
                        value = value + coef * samples[parent]
                
                noise = np.random.normal(eq.noise_mean, eq.noise_std, n_samples)
                samples[var] = value + noise
            else:
                samples[var] = np.random.normal(
                    eq.intercept or 0, eq.noise_std, n_samples
                )
        
        return samples.get(target_variable, np.zeros(n_samples))
    
    def _get_topological_order(
        self, 
        equations: Dict[str, StructuralEquation]
    ) -> List[str]:
        """Get topological ordering from equations"""
        # Build dependency graph
        in_degree = {var: 0 for var in equations.keys()}
        children = {var: [] for var in equations.keys()}
        
        for var, eq in equations.items():
            for parent in eq.parents:
                if parent in equations:
                    in_degree[var] += 1
                    children[parent].append(var)
        
        # Kahn's algorithm
        queue = [var for var, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            var = queue.pop(0)
            order.append(var)
            
            for child in children[var]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return order
    
    def _compute_utility(
        self, 
        outcomes: np.ndarray, 
        utility_func: UtilityFunction
    ) -> np.ndarray:
        """Compute utility for outcomes"""
        if utility_func.utility_type == UtilityType.LINEAR:
            if utility_func.weights:
                # For single outcome, use the weight
                weight = list(utility_func.weights.values())[0] if utility_func.weights else 1.0
                return weight * outcomes
            return outcomes
        
        elif utility_func.utility_type == UtilityType.QUADRATIC:
            # U = -risk_aversion * (y - target)^2 + linear term
            risk_aversion = utility_func.risk_aversion
            target = 0  # Default target
            return -risk_aversion * (outcomes - target) ** 2 + outcomes
        
        elif utility_func.utility_type == UtilityType.EXPONENTIAL:
            # U = -exp(-risk_aversion * y)
            risk_aversion = utility_func.risk_aversion
            return -np.exp(-risk_aversion * outcomes)
        
        elif utility_func.utility_type == UtilityType.LOGARITHMIC:
            # U = log(y + c) for positive outcomes
            return np.log(np.abs(outcomes) + 1)
        
        else:
            return outcomes
    
    def _build_constraints(
        self, 
        request: OptimizationRequest
    ) -> List[Dict[str, Any]]:
        """Build scipy constraint format"""
        constraints = []
        
        for constraint in request.constraints:
            if constraint.constraint_type == ConstraintType.INEQUALITY:
                # g(a) <= c  =>  g(a) - c <= 0
                def make_ineq_const(c):
                    def ineq_const(x):
                        return c.value - self._evaluate_constraint_expression(
                            x, c.expression, request.action_variables
                        )
                    return ineq_const
                
                constraints.append({
                    'type': 'ineq',
                    'fun': make_ineq_const(constraint)
                })
            
            elif constraint.constraint_type == ConstraintType.EQUALITY:
                def make_eq_const(c):
                    def eq_const(x):
                        return self._evaluate_constraint_expression(
                            x, c.expression, request.action_variables
                        ) - c.value
                    return eq_const
                
                constraints.append({
                    'type': 'eq',
                    'fun': make_eq_const(constraint)
                })
        
        return constraints
    
    def _evaluate_constraint_expression(
        self,
        actions: np.ndarray,
        expression: str,
        action_vars: List[str]
    ) -> float:
        """Evaluate constraint expression"""
        # Create local variables for evaluation
        local_vars = {var: actions[i] for i, var in enumerate(action_vars)}
        
        # Safe evaluation (basic arithmetic only)
        try:
            # Replace ^ with ** for exponentiation
            expr = expression.replace('^', '**')
            result = eval(expr, {"__builtins__": {}}, local_vars)
            return float(result)
        except:
            return 0.0
    
    def _build_bounds(
        self,
        action_vars: List[str],
        action_bounds: Dict[str, Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Build bounds for optimization"""
        bounds = []
        for var in action_vars:
            if var in action_bounds:
                bounds.append(action_bounds[var])
            else:
                bounds.append((None, None))  # Unbounded
        return bounds
    
    def _get_initial_guess(self, request: OptimizationRequest) -> np.ndarray:
        """Get initial guess for optimization"""
        if request.initial_values:
            return np.array([
                request.initial_values.get(var, 0.0)
                for var in request.action_variables
            ])
        
        # Use midpoint of bounds
        x0 = []
        for var in request.action_variables:
            if var in request.action_bounds:
                low, high = request.action_bounds[var]
                if low is not None and high is not None:
                    x0.append((low + high) / 2)
                elif low is not None:
                    x0.append(low + 1.0)
                elif high is not None:
                    x0.append(high - 1.0)
                else:
                    x0.append(0.0)
            else:
                x0.append(0.0)
        
        return np.array(x0)
    
    async def _scipy_optimize(
        self,
        objective: Callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: List[Dict],
        request: OptimizationRequest
    ) -> Dict[str, Any]:
        """Run scipy optimization"""
        method_map = {
            OptimizationMethod.SLSQP: 'SLSQP',
            OptimizationMethod.TRUST_CONSTR: 'trust-constr',
            OptimizationMethod.COBYLA: 'COBYLA',
        }
        
        method = method_map.get(request.method, 'SLSQP')
        
        options = {
            'maxiter': request.max_iterations,
            'ftol': request.tolerance
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                options=options
            )
        
        return {
            'x': result.x,
            'fun': result.fun,
            'success': result.success,
            'nit': getattr(result, 'nit', 0),
            'message': result.message if hasattr(result, 'message') else ''
        }
    
    async def _differential_evolution(
        self,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        constraints: List[Dict],
        request: OptimizationRequest
    ) -> Dict[str, Any]:
        """Run differential evolution optimization"""
        # Convert bounds format
        de_bounds = [(b[0] if b[0] is not None else -1e6,
                      b[1] if b[1] is not None else 1e6) 
                     for b in bounds]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.differential_evolution(
                objective,
                de_bounds,
                maxiter=request.max_iterations,
                tol=request.tolerance,
                seed=42
            )
        
        return {
            'x': result.x,
            'fun': result.fun,
            'success': result.success,
            'nit': result.nit if hasattr(result, 'nit') else 0,
            'message': result.message if hasattr(result, 'message') else ''
        }
    
    async def _compute_expected_outcome(
        self,
        actions: Dict[str, float],
        equations: Dict[str, StructuralEquation],
        target_variable: str,
        n_samples: int
    ) -> Dict[str, float]:
        """Compute expected outcome under intervention"""
        outcomes = self._sample_outcomes(actions, equations, target_variable, n_samples)
        
        return {
            "mean": float(np.mean(outcomes)),
            "std": float(np.std(outcomes)),
            "median": float(np.median(outcomes)),
            "q5": float(np.percentile(outcomes, 5)),
            "q95": float(np.percentile(outcomes, 95))
        }
    
    async def _compute_outcome_confidence_interval(
        self,
        actions: Dict[str, float],
        equations: Dict[str, StructuralEquation],
        target_variable: str,
        n_samples: int
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for outcomes"""
        outcomes = self._sample_outcomes(actions, equations, target_variable, n_samples)
        
        # Bootstrap for CI
        n_bootstrap = 500
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        return {
            "mean": (
                float(np.percentile(bootstrap_means, 2.5)),
                float(np.percentile(bootstrap_means, 97.5))
            )
        }
    
    def _evaluate_constraints(
        self,
        actions: np.ndarray,
        constraints: List[Constraint]
    ) -> Dict[str, float]:
        """Evaluate all constraints"""
        values = {}
        
        for constraint in constraints:
            value = self._evaluate_constraint_expression(
                actions, constraint.expression, []
            )
            values[constraint.name] = value
        
        return values
    
    async def _compute_sensitivity(
        self,
        optimal_actions: np.ndarray,
        objective: Callable,
        action_vars: List[str]
    ) -> SensitivityAnalysis:
        """Compute sensitivity analysis at optimum"""
        # Compute gradient numerically
        epsilon = 1e-5
        gradient = {}
        
        base_value = objective(optimal_actions)
        
        for i, var in enumerate(action_vars):
            x_plus = optimal_actions.copy()
            x_plus[i] += epsilon
            value_plus = objective(x_plus)
            
            gradient[var] = float(-(value_plus - base_value) / epsilon)
        
        # Compute Hessian eigenvalues (for convexity check)
        n = len(action_vars)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                x_pp = optimal_actions.copy()
                x_pm = optimal_actions.copy()
                x_mp = optimal_actions.copy()
                x_mm = optimal_actions.copy()
                
                x_pp[i] += epsilon; x_pp[j] += epsilon
                x_pm[i] += epsilon; x_pm[j] -= epsilon
                x_mp[i] -= epsilon; x_mp[j] += epsilon
                x_mm[i] -= epsilon; x_mm[j] -= epsilon
                
                hessian[i, j] = (
                    objective(x_pp) - objective(x_pm) - 
                    objective(x_mp) + objective(x_mm)
                ) / (4 * epsilon ** 2)
        
        eigenvalues = np.linalg.eigvalsh(hessian)
        is_local_max = all(eigenvalues < 0)
        
        return SensitivityAnalysis(
            gradient_at_optimum=gradient,
            hessian_eigenvalues=eigenvalues.tolist(),
            is_local_maximum=is_local_max,
            action_sensitivity={var: abs(grad) for var, grad in gradient.items()},
            shadow_prices={},  # Would need Lagrange multipliers
            robustness_score=1.0 / (1.0 + max(abs(g) for g in gradient.values()))
        )
    
    def _generate_objective_formulation(self, request: OptimizationRequest) -> str:
        """Generate LaTeX formulation of objective"""
        utility_latex = request.utility_function.to_latex()
        actions = ", ".join(request.action_variables)
        
        return f"\\max_{{{actions}}} \\mathbb{{E}}[{utility_latex} | do(A)]"
    
    def _generate_lagrangian(
        self, 
        request: OptimizationRequest, 
        result: Dict
    ) -> str:
        """Generate Lagrangian formulation"""
        utility_latex = request.utility_function.to_latex()
        
        constraint_terms = []
        for c in request.constraints:
            if c.constraint_type == ConstraintType.INEQUALITY:
                constraint_terms.append(f"\\lambda_{{{c.name}}} ({c.expression} - {c.value})")
        
        if constraint_terms:
            return f"\\mathcal{{L}} = {utility_latex} + " + " + ".join(constraint_terms)
        return f"\\mathcal{{L}} = {utility_latex}"
    
    async def evaluate_policy(
        self,
        model: CausalModelResult,
        df: pd.DataFrame,
        request: PolicyEvaluationRequest
    ) -> PolicyEvaluationResult:
        """Evaluate a specific policy"""
        evaluation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        equations = {eq.variable: eq for eq in model.structural_equations}
        
        # Sample outcomes under the policy
        outcomes = {}
        for var in equations.keys():
            if var not in request.policy_actions:
                outcomes[var] = self._sample_outcomes(
                    request.policy_actions, equations, var, request.n_samples
                )
        
        # Compute distribution statistics
        outcome_distribution = {}
        for var, samples in outcomes.items():
            outcome_distribution[var] = {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "median": float(np.median(samples)),
                "q5": float(np.percentile(samples, 5)),
                "q95": float(np.percentile(samples, 95))
            }
        
        # Compute utility if specified
        expected_utility = None
        utility_ci = None
        
        if request.utility_function:
            target_samples = list(outcomes.values())[0] if outcomes else np.zeros(request.n_samples)
            utilities = self._compute_utility(target_samples, request.utility_function)
            expected_utility = float(np.mean(utilities))
            
            if request.compute_confidence_intervals:
                bootstrap_utils = []
                for _ in range(500):
                    boot_sample = np.random.choice(utilities, size=len(utilities), replace=True)
                    bootstrap_utils.append(np.mean(boot_sample))
                utility_ci = (
                    float(np.percentile(bootstrap_utils, 2.5)),
                    float(np.percentile(bootstrap_utils, 97.5))
                )
        
        computation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return PolicyEvaluationResult(
            evaluation_id=evaluation_id,
            model_id=model.model_id,
            policy_actions=request.policy_actions,
            outcome_distribution=outcome_distribution,
            expected_utility=expected_utility,
            utility_confidence_interval=utility_ci,
            computation_time=computation_time
        )
    
    async def compute_value_of_information(
        self,
        model: CausalModelResult,
        request: OptimizationRequest,
        uncertainty_reduction: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute Expected Value of Perfect Information (EVPI)
        
        EVPI = E[max_a U(a, θ)] - max_a E[U(a, θ)]
        
        where θ represents uncertain parameters
        """
        # Current optimal utility
        base_result = await self.optimize_policy(model, pd.DataFrame(), request)
        base_utility = base_result.optimal_utility
        
        # Simulate with reduced uncertainty
        # (Simplified - would need full uncertainty propagation)
        voi = {}
        
        for var in request.action_variables:
            # Estimate value of learning this variable perfectly
            voi[var] = base_utility * uncertainty_reduction * 0.1
        
        return voi