"""
Report and Experiment Logging Model Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class ReportFormat(str, Enum):
    """Report output formats"""
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    LATEX = "latex"


class ReportSection(str, Enum):
    """Sections to include in report"""
    DATA_SUMMARY = "data_summary"
    CAUSAL_STRUCTURE = "causal_structure"
    STRUCTURAL_EQUATIONS = "structural_equations"
    INTERVENTION_ANALYSIS = "intervention_analysis"
    COUNTERFACTUAL_ANALYSIS = "counterfactual_analysis"
    OPTIMIZATION_RESULTS = "optimization_results"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"
    MATHEMATICAL_APPENDIX = "mathematical_appendix"
    METHODOLOGY = "methodology"


class ExperimentLog(BaseModel):
    """Log entry for experiment tracking"""
    log_id: str = Field(..., description="Unique log identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Experiment metadata
    experiment_name: str = Field(..., description="Experiment name")
    experiment_type: str = Field(..., description="Type of experiment")
    
    # Configuration
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters"
    )
    random_seed: Optional[int] = Field(default=None, description="Random seed used")
    
    # Results summary
    results_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of results"
    )
    
    # Metrics
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    
    # Timestamps
    start_time: datetime = Field(..., description="Experiment start time")
    end_time: Optional[datetime] = Field(default=None, description="Experiment end time")
    duration_seconds: Optional[float] = Field(default=None, description="Duration in seconds")
    
    # Status
    status: str = Field(default="running", description="Experiment status")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    # References
    dataset_id: Optional[str] = Field(default=None, description="Dataset used")
    model_id: Optional[str] = Field(default=None, description="Model used")


class ReportConfig(BaseModel):
    """Configuration for report generation"""
    # Report identification
    report_title: str = Field(
        default="Causal Policy Optimization Report",
        description="Report title"
    )
    report_author: Optional[str] = Field(default=None, description="Report author")
    
    # Format
    format: ReportFormat = Field(
        default=ReportFormat.PDF,
        description="Output format"
    )
    
    # Sections to include
    sections: List[ReportSection] = Field(
        default_factory=lambda: [
            ReportSection.DATA_SUMMARY,
            ReportSection.CAUSAL_STRUCTURE,
            ReportSection.STRUCTURAL_EQUATIONS,
            ReportSection.INTERVENTION_ANALYSIS,
            ReportSection.OPTIMIZATION_RESULTS,
            ReportSection.UNCERTAINTY_ANALYSIS,
            ReportSection.MATHEMATICAL_APPENDIX
        ],
        description="Sections to include"
    )
    
    # Content options
    include_code: bool = Field(default=False, description="Include code snippets")
    include_raw_data: bool = Field(default=False, description="Include raw data tables")
    include_visualizations: bool = Field(default=True, description="Include visualizations")
    include_confidence_intervals: bool = Field(default=True, description="Include confidence intervals")
    
    # Detail level
    detail_level: str = Field(
        default="standard",
        description="Detail level: minimal, standard, detailed"
    )
    
    # References
    session_id: str = Field(..., description="Session identifier")
    dataset_id: Optional[str] = Field(default=None, description="Dataset to include")
    model_id: Optional[str] = Field(default=None, description="Model to include")
    optimization_id: Optional[str] = Field(default=None, description="Optimization to include")
    
    # Custom content
    custom_sections: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Custom sections to add"
    )


class ReportResult(BaseModel):
    """Result of report generation"""
    report_id: str = Field(..., description="Unique report identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Report metadata
    title: str = Field(..., description="Report title")
    format: ReportFormat = Field(..., description="Report format")
    
    # Generated content
    file_path: Optional[str] = Field(default=None, description="Path to generated file")
    file_size_bytes: Optional[int] = Field(default=None, description="File size")
    download_url: Optional[str] = Field(default=None, description="Download URL")
    
    # Content summary
    sections_included: List[str] = Field(..., description="Sections included")
    n_pages: Optional[int] = Field(default=None, description="Number of pages")
    n_figures: int = Field(default=0, description="Number of figures")
    n_tables: int = Field(default=0, description="Number of tables")
    
    # Generation metadata
    generation_time: float = Field(..., description="Time to generate (seconds)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Status
    status: str = Field(default="completed", description="Generation status")
    error_message: Optional[str] = Field(default=None, description="Error if failed")


class MathematicalExplanation(BaseModel):
    """Mathematical explanation for display"""
    explanation_id: str = Field(..., description="Unique identifier")
    model_id: str = Field(..., description="Model identifier")
    
    # Structural equations
    structural_equations_latex: List[str] = Field(
        default_factory=list,
        description="LaTeX formatted structural equations"
    )
    
    # Likelihood formulation
    likelihood_formulation: str = Field(
        default="",
        description="Likelihood function in LaTeX"
    )
    
    # Posterior estimation
    posterior_formulation: str = Field(
        default="",
        description="Posterior distribution formulation"
    )
    estimation_method: str = Field(
        default="",
        description="Estimation method description"
    )
    
    # Intervention formulas
    intervention_formulas: Dict[str, str] = Field(
        default_factory=dict,
        description="Intervention formulas in LaTeX"
    )
    truncated_factorization: str = Field(
        default="",
        description="Truncated factorization formula"
    )
    
    # Optimization objective
    optimization_objective: str = Field(
        default="",
        description="Optimization objective in LaTeX"
    )
    constraints_latex: List[str] = Field(
        default_factory=list,
        description="Constraints in LaTeX"
    )
    
    # Uncertainty bounds
    uncertainty_formulation: str = Field(
        default="",
        description="Uncertainty quantification formulation"
    )
    
    # Derivations
    derivations: Dict[str, str] = Field(
        default_factory=dict,
        description="Step-by-step derivations"
    )
    
    # Assumptions
    assumptions: List[str] = Field(
        default_factory=list,
        description="Model assumptions"
    )


class SessionSummary(BaseModel):
    """Summary of a session's activities"""
    session_id: str = Field(..., description="Session identifier")
    
    # Session metadata
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    
    # Data uploaded
    datasets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Dataset summaries"
    )
    
    # Models created
    models: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Model summaries"
    )
    
    # Analyses performed
    interventions: int = Field(default=0, description="Number of interventions")
    counterfactuals: int = Field(default=0, description="Number of counterfactuals")
    optimizations: int = Field(default=0, description="Number of optimizations")
    
    # Reports generated
    reports: List[str] = Field(
        default_factory=list,
        description="Report IDs generated"
    )
    
    # Experiment logs
    experiment_logs: List[str] = Field(
        default_factory=list,
        description="Experiment log IDs"
    )


class ReproducibilityPackage(BaseModel):
    """Package for reproducing an analysis"""
    package_id: str = Field(..., description="Package identifier")
    session_id: str = Field(..., description="Session identifier")
    
    # Configuration snapshots
    data_config: Dict[str, Any] = Field(..., description="Data configuration")
    structure_config: Dict[str, Any] = Field(..., description="Structure learning config")
    parameter_config: Dict[str, Any] = Field(..., description="Parameter learning config")
    
    # Random seeds
    random_seeds: Dict[str, int] = Field(
        default_factory=dict,
        description="Random seeds used for each step"
    )
    
    # Version information
    library_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Library versions used"
    )
    
    # Data references
    dataset_hash: Optional[str] = Field(
        default=None,
        description="Hash of dataset for verification"
    )
    
    # Code
    analysis_code: Optional[str] = Field(
        default=None,
        description="Python code to reproduce analysis"
    )
    
    # Results to verify
    expected_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Expected results for verification"
    )
    
    created_at: datetime = Field(default_factory=datetime.utcnow)