"""
Report Service for Generating Analytical Reports
Handles report generation, experiment logging, and reproducibility
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import os
import json
import logging
from io import BytesIO

from app.config import settings
from app.models.report_models import (
    ReportConfig, ReportResult, ReportFormat, ReportSection,
    ExperimentLog, MathematicalExplanation, SessionSummary,
    ReproducibilityPackage
)
from app.models.causal_models import CausalModelResult

logger = logging.getLogger(__name__)


class ReportService:
    """Service for generating reports and managing experiments"""
    
    def __init__(self):
        self._storage_path = settings.REPORTS_DIR
        self._experiments: Dict[str, ExperimentLog] = {}
        
    async def generate_report(
        self,
        config: ReportConfig,
        model: Optional[CausalModelResult] = None,
        dataset_info: Optional[Dict] = None,
        intervention_results: Optional[List] = None,
        optimization_result: Optional[Dict] = None
    ) -> ReportResult:
        """Generate a comprehensive analytical report"""
        report_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        content = {}
        
        for section in config.sections:
            if section == ReportSection.DATA_SUMMARY and dataset_info:
                content["data_summary"] = self._generate_data_summary(dataset_info)
            elif section == ReportSection.CAUSAL_STRUCTURE and model:
                content["causal_structure"] = self._generate_causal_structure_section(model)
            elif section == ReportSection.STRUCTURAL_EQUATIONS and model:
                content["structural_equations"] = self._generate_equations_section(model)
            elif section == ReportSection.INTERVENTION_ANALYSIS and intervention_results:
                content["intervention_analysis"] = self._generate_intervention_section(intervention_results)
            elif section == ReportSection.OPTIMIZATION_RESULTS and optimization_result:
                content["optimization_results"] = self._generate_optimization_section(optimization_result)
            elif section == ReportSection.UNCERTAINTY_ANALYSIS and model:
                content["uncertainty_analysis"] = self._generate_uncertainty_section(model)
            elif section == ReportSection.MATHEMATICAL_APPENDIX and model:
                content["mathematical_appendix"] = self._generate_mathematical_appendix(model)
            elif section == ReportSection.METHODOLOGY:
                content["methodology"] = self._generate_methodology_section()
        
        if config.custom_sections:
            for custom in config.custom_sections:
                content[custom.get("title", "Custom")] = custom.get("content", "")
        
        output_content, file_path = await self._generate_output(report_id, config, content)
        
        computation_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ReportResult(
            report_id=report_id,
            session_id=config.session_id,
            title=config.report_title,
            format=config.format,
            file_path=file_path,
            file_size_bytes=len(output_content) if output_content else 0,
            sections_included=[s.value for s in config.sections],
            n_figures=self._count_figures(content),
            n_tables=self._count_tables(content),
            generation_time=computation_time
        )
    
    def _generate_data_summary(self, dataset_info: Dict) -> str:
        return f"""
## Data Summary

### Dataset Overview
- **Number of observations**: {dataset_info.get('n_rows', 'N/A')}
- **Number of variables**: {dataset_info.get('n_columns', 'N/A')}
- **Memory usage**: {dataset_info.get('memory_usage_mb', 0):.2f} MB

### Variable Types
- **Numeric variables**: {dataset_info.get('n_numeric', 0)}
- **Categorical variables**: {dataset_info.get('n_categorical', 0)}
- **Binary variables**: {dataset_info.get('n_binary', 0)}

### Data Quality
- **Missing values**: {dataset_info.get('missing_percentage', 0):.2f}%
- **Quality score**: {dataset_info.get('quality_score', 0):.2f}
"""
    
    def _generate_causal_structure_section(self, model: CausalModelResult) -> str:
        graph = model.graph
        section = f"""
## Causal Structure Analysis

### Learned Graph Structure
- **Number of nodes**: {graph.n_nodes}
- **Number of edges**: {graph.n_edges}
- **Is DAG**: {graph.is_dag}

### Model Quality Metrics
- **Log-likelihood**: {model.log_likelihood:.4f}
- **BIC**: {model.bic:.4f}
- **AIC**: {model.aic:.4f}

### Discovered Causal Relationships

| Source | Target | Confidence | Coefficient |
|--------|--------|------------|-------------|
"""
        for edge in graph.edges:
            coef = f"{edge.coefficient:.4f}" if edge.coefficient else "N/A"
            section += f"| {edge.source} | {edge.target} | {edge.confidence:.2f} | {coef} |\n"
        
        return section
    
    def _generate_equations_section(self, model: CausalModelResult) -> str:
        section = """
## Structural Equations

### Learned Equations

"""
        for eq in model.structural_equations:
            latex = eq.to_latex()
            section += f"**{eq.variable}**:\n\n$${latex}$$\n\n"
            if eq.coefficients:
                section += "- Coefficients:\n"
                for parent, coef in eq.coefficients.items():
                    section += f"  - {parent}: {coef:.4f}\n"
            section += f"- Noise std: {eq.noise_std:.4f}\n"
            if eq.r_squared is not None:
                section += f"- R²: {eq.r_squared:.4f}\n"
            section += "\n"
        
        return section
    
    def _generate_intervention_section(self, results: List) -> str:
        section = "## Intervention Analysis\n\n"
        for i, result in enumerate(results):
            section += f"### Intervention {i+1}\n\n"
            section += f"**Formula**: {getattr(result, 'intervention_formula', 'N/A')}\n\n"
        return section
    
    def _generate_optimization_section(self, result: Dict) -> str:
        section = """
## Policy Optimization Results

### Optimal Solution

"""
        section += "**Optimal Actions**:\n\n"
        for var, value in result.get('optimal_actions', {}).items():
            section += f"- {var}: {value:.4f}\n"
        section += f"\n**Optimal Utility**: {result.get('optimal_utility', 0):.4f}\n\n"
        return section
    
    def _generate_uncertainty_section(self, model: CausalModelResult) -> str:
        return """
## Uncertainty Quantification

The structural learning procedure accounts for uncertainty through:
- Bootstrap resampling for edge confidence
- Bayesian posterior estimation
- Sensitivity analysis for parameter perturbations
"""
    
    def _generate_mathematical_appendix(self, model: CausalModelResult) -> str:
        return """
## Mathematical Appendix

### A. Structural Causal Model

A Structural Causal Model (SCM) is defined as:
- V: endogenous variables
- U: exogenous (noise) variables  
- F: structural functions
- P(U): probability distribution over exogenous variables

### B. Interventional Distribution

The interventional distribution P(Y|do(X=x)) is computed using the truncated factorization.

### C. Optimization Formulation

The policy optimization problem is:
max_a E[U(Y) | do(A=a)] subject to g(a) <= c
"""
    
    def _generate_methodology_section(self) -> str:
        return """
## Methodology

### Causal Discovery
- Constraint-based methods: PC algorithm
- Score-based methods: Greedy Equivalence Search
- Differentiable methods: NOTEARS
- Bayesian methods: MCMC sampling

### Parameter Estimation
- Ordinary Least Squares for linear models
- Ridge/Lasso regression for regularized estimation
- Bayesian inference for uncertainty quantification
"""
    
    async def _generate_output(self, report_id: str, config: ReportConfig, content: Dict[str, str]) -> tuple:
        os.makedirs(self._storage_path, exist_ok=True)
        
        if config.format == ReportFormat.JSON:
            output = json.dumps(content, indent=2)
            file_path = os.path.join(self._storage_path, f"{report_id}.json")
        else:
            output = self._to_markdown(config, content)
            file_path = os.path.join(self._storage_path, f"{report_id}.md")
        
        with open(file_path, 'w') as f:
            f.write(output)
        
        return output, file_path
    
    def _to_markdown(self, config: ReportConfig, content: Dict[str, str]) -> str:
        md = f"# {config.report_title}\n\n"
        md += f"*Generated: {datetime.utcnow().isoformat()}*\n\n"
        if config.report_author:
            md += f"*Author: {config.report_author}*\n\n"
        
        for section_name, section_content in content.items():
            md += section_content + "\n\n"
        
        return md
    
    def _count_figures(self, content: Dict[str, str]) -> int:
        return sum(c.count("![") for c in content.values())
    
    def _count_tables(self, content: Dict[str, str]) -> int:
        return sum(c.count("|---|") for c in content.values())
    
    async def create_experiment_log(
        self,
        session_id: str,
        experiment_name: str,
        experiment_type: str,
        configuration: Dict[str, Any],
        random_seed: Optional[int] = None
    ) -> ExperimentLog:
        """Create a new experiment log"""
        log_id = str(uuid.uuid4())
        
        log = ExperimentLog(
            log_id=log_id,
            session_id=session_id,
            experiment_name=experiment_name,
            experiment_type=experiment_type,
            configuration=configuration,
            random_seed=random_seed,
            start_time=datetime.utcnow(),
            status="running"
        )
        
        self._experiments[log_id] = log
        return log
    
    async def update_experiment_log(
        self,
        log_id: str,
        results_summary: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        status: str = None,
        error_message: str = None
    ):
        """Update an experiment log"""
        if log_id not in self._experiments:
            return
        
        log = self._experiments[log_id]
        
        if results_summary:
            log.results_summary = results_summary
        if metrics:
            log.metrics = metrics
        if status:
            log.status = status
        if error_message:
            log.error_message = error_message
        
        if status in ["completed", "failed"]:
            log.end_time = datetime.utcnow()
            log.duration_seconds = (log.end_time - log.start_time).total_seconds()
    
    async def get_mathematical_explanation(
        self,
        model: CausalModelResult
    ) -> MathematicalExplanation:
        """Generate mathematical explanation for a model"""
        explanation_id = str(uuid.uuid4())
        
        equations_latex = [eq.to_latex() for eq in model.structural_equations]
        
        return MathematicalExplanation(
            explanation_id=explanation_id,
            model_id=model.model_id,
            structural_equations_latex=equations_latex,
            likelihood_formulation="L(θ|D) = ∏_i P(X_i|Pa_i; θ)",
            posterior_formulation="P(θ|D) ∝ P(D|θ)P(θ)",
            estimation_method="Maximum Likelihood / Bayesian Posterior",
            intervention_formulas={},
            truncated_factorization="P(Y|do(X=x)) = ∏_{i:V_i∉X} P(V_i|Pa_i)|_{X=x}",
            optimization_objective="max_a E[U(Y)|do(A=a)]",
            constraints_latex=[],
            uncertainty_formulation="Bootstrap / Posterior Sampling",
            assumptions=[
                "Causal Markov Condition",
                "Faithfulness Assumption", 
                "No Hidden Confounders",
                "Correct Functional Form"
            ]
        )
    
    async def create_reproducibility_package(
        self,
        session_id: str,
        data_config: Dict[str, Any],
        structure_config: Dict[str, Any],
        parameter_config: Dict[str, Any],
        random_seeds: Dict[str, int]
    ) -> ReproducibilityPackage:
        """Create a reproducibility package"""
        import sys
        
        package_id = str(uuid.uuid4())
        
        library_versions = {
            "numpy": "1.26.3",
            "pandas": "2.2.0",
            "scipy": "1.12.0",
            "scikit-learn": "1.4.0",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}"
        }
        
        return ReproducibilityPackage(
            package_id=package_id,
            session_id=session_id,
            data_config=data_config,
            structure_config=structure_config,
            parameter_config=parameter_config,
            random_seeds=random_seeds,
            library_versions=library_versions
        )