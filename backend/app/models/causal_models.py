"""
Causal Model Schemas for Structure Learning and Parameter Estimation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime


class StructureLearningMethod(str, Enum):
    """Methods for causal structure learning"""
    PC = "pc"  # Constraint-based PC algorithm
    GES = "ges"  # Greedy Equivalence Search
    NOTEARS = "notears"  # NOTEARS differentiable approach
    HYBRID = "hybrid"  # Hybrid score + constraint
    BAYESIAN = "bayesian"  # Bayesian structure learning


class ScoreFunction(str, Enum):
    """Score functions for structure learning"""
    BIC = "bic"  # Bayesian Information Criterion
    AIC = "aic"  # Akaike Information Criterion
    BDEU = "bdeu"  # Bayesian Dirichlet equivalent uniform
    MDL = "mdl"  # Minimum Description Length


class IndependenceTest(str, Enum):
    """Statistical independence tests"""
    CHI_SQUARE = "chi_square"
    G_TEST = "g_test"
    FISHER_Z = "fisher_z"
    PEARSON = "pearson"
    KS = "ks"  # Kolmogorov-Smirnov


class FunctionalForm(str, Enum):
    """Functional forms for structural equations"""
    LINEAR = "linear"
    LINEAR_GAUSSIAN = "linear_gaussian"
    NONLINEAR_ADDITIVE = "nonlinear_additive"
    NEURAL = "neural"
    GP = "gaussian_process"


class Node(BaseModel):
    """Node in the causal graph"""
    id: str = Field(..., description="Node identifier")
    name: str = Field(..., description="Variable name")
    type: str = Field(default="variable", description="Node type")
    x: Optional[float] = Field(default=None, description="X position for visualization")
    y: Optional[float] = Field(default=None, description="Y position for visualization")
    
    # Node metadata
    is_target: bool = Field(default=False, description="Whether this is a target variable")
    is_action: bool = Field(default=False, description="Whether this is an action variable")
    is_confounder: bool = Field(default=False, description="Whether this is a confounder")


class Edge(BaseModel):
    """Edge in the causal graph"""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    
    # Edge properties
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Edge confidence/probability"
    )
    coefficient: Optional[float] = Field(default=None, description="Edge coefficient (linear models)")
    coefficient_std: Optional[float] = Field(default=None, description="Coefficient standard error")
    
    # Causal direction confidence
    direction_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in causal direction"
    )
    
    # Edge type
    is_directed: bool = Field(default=True, description="Whether edge is directed")
    edge_type: str = Field(default="causal", description="Edge type (causal, association, etc.)")


class StructuralEquation(BaseModel):
    """Structural equation for a variable"""
    variable: str = Field(..., description="Target variable name")
    parents: List[str] = Field(default_factory=list, description="Parent variables")
    functional_form: FunctionalForm = Field(
        default=FunctionalForm.LINEAR_GAUSSIAN,
        description="Functional form of the equation"
    )
    
    # Linear coefficients
    coefficients: Optional[Dict[str, float]] = Field(
        default=None,
        description="Coefficients for each parent"
    )
    intercept: Optional[float] = Field(default=None, description="Intercept term")
    
    # Noise distribution
    noise_mean: float = Field(default=0.0, description="Mean of noise term")
    noise_std: float = Field(default=1.0, description="Standard deviation of noise term")
    noise_distribution: str = Field(default="gaussian", description="Noise distribution type")
    
    # Nonlinear function representation
    nonlinear_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters for nonlinear functions"
    )
    
    # Model fit statistics
    r_squared: Optional[float] = Field(default=None, description="R-squared value")
    rmse: Optional[float] = Field(default=None, description="Root mean squared error")
    log_likelihood: Optional[float] = Field(default=None, description="Log-likelihood")
    
    def to_latex(self) -> str:
        """Generate LaTeX representation of the equation"""
        if self.functional_form in [FunctionalForm.LINEAR, FunctionalForm.LINEAR_GAUSSIAN]:
            terms = []
            if self.intercept is not None:
                terms.append(f"{self.intercept:.3f}")
            if self.coefficients:
                for parent, coef in self.coefficients.items():
                    terms.append(f"{coef:.3f} \\cdot {parent}")
            equation = " + ".join(terms) if terms else "0"
            return f"{self.variable} = {equation} + \\epsilon_{{{self.variable}}}"
        else:
            return f"{self.variable} = f_{{{self.variable}}}({', '.join(self.parents)}) + \\epsilon_{{{self.variable}}}"


class CausalGraph(BaseModel):
    """Complete causal graph structure"""
    nodes: List[Node] = Field(default_factory=list, description="Graph nodes")
    edges: List[Edge] = Field(default_factory=list, description="Graph edges")
    
    # Graph properties
    is_dag: bool = Field(default=True, description="Whether graph is a DAG")
    n_nodes: int = Field(default=0, description="Number of nodes")
    n_edges: int = Field(default=0, description="Number of edges")
    
    # Topological ordering
    topological_order: Optional[List[str]] = Field(
        default=None,
        description="Topological ordering of nodes"
    )
    
    # Adjacency representation
    adjacency_matrix: Optional[List[List[int]]] = Field(
        default=None,
        description="Adjacency matrix representation"
    )
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id or node.name == node_id:
                return node
        return None
    
    def get_parents(self, node_id: str) -> List[str]:
        """Get parent nodes of a given node"""
        parents = []
        for edge in self.edges:
            if edge.target == node_id:
                parents.append(edge.source)
        return parents
    
    def get_children(self, node_id: str) -> List[str]:
        """Get child nodes of a given node"""
        children = []
        for edge in self.edges:
            if edge.source == node_id:
                children.append(edge.target)
        return children


class StructureLearningConfig(BaseModel):
    """Configuration for structure learning"""
    method: StructureLearningMethod = Field(
        default=StructureLearningMethod.HYBRID,
        description="Structure learning method"
    )
    
    # Score-based parameters
    score_function: ScoreFunction = Field(
        default=ScoreFunction.BIC,
        description="Score function for structure evaluation"
    )
    max_parents: int = Field(default=5, description="Maximum number of parents per node")
    
    # Constraint-based parameters
    independence_test: IndependenceTest = Field(
        default=IndependenceTest.FISHER_Z,
        description="Independence test method"
    )
    alpha: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Significance level for independence tests"
    )
    
    # NOTEARS specific
    lambda_l1: float = Field(default=0.1, description="L1 regularization for NOTEARS")
    lambda_dag: float = Field(default=10.0, description="DAG penalty for NOTEARS")
    
    # Bayesian specific
    n_samples: int = Field(default=1000, description="Number of MCMC samples")
    prior_edge_prob: float = Field(default=0.3, description="Prior edge probability")
    
    # Bootstrap for confidence estimation
    use_bootstrap: bool = Field(default=True, description="Use bootstrap for edge confidence")
    n_bootstrap: int = Field(default=100, description="Number of bootstrap samples")
    
    # Random seed for reproducibility
    random_seed: Optional[int] = Field(default=42, description="Random seed")


class ParameterLearningConfig(BaseModel):
    """Configuration for parameter learning"""
    functional_form: FunctionalForm = Field(
        default=FunctionalForm.LINEAR_GAUSSIAN,
        description="Functional form for structural equations"
    )
    
    # Bayesian parameters
    use_bayesian: bool = Field(default=True, description="Use Bayesian estimation")
    n_posterior_samples: int = Field(default=1000, description="Number of posterior samples")
    
    # Prior specification
    coefficient_prior: str = Field(default="normal", description="Prior for coefficients")
    coefficient_prior_params: Dict[str, float] = Field(
        default_factory=lambda: {"mean": 0.0, "std": 1.0},
        description="Prior parameters for coefficients"
    )
    noise_prior: str = Field(default="halfnormal", description="Prior for noise std")
    noise_prior_params: Dict[str, float] = Field(
        default_factory=lambda: {"std": 1.0},
        description="Prior parameters for noise"
    )
    
    # Regularization
    l1_penalty: float = Field(default=0.0, description="L1 penalty for coefficients")
    l2_penalty: float = Field(default=0.0, description="L2 penalty for coefficients")


class CausalModelResult(BaseModel):
    """Result of causal model learning"""
    model_id: str = Field(..., description="Unique model identifier")
    session_id: str = Field(..., description="Session identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    
    # Learned structure
    graph: CausalGraph = Field(..., description="Learned causal graph")
    structural_equations: List[StructuralEquation] = Field(
        default_factory=list,
        description="Learned structural equations"
    )
    
    # Model quality metrics
    log_likelihood: float = Field(..., description="Model log-likelihood")
    bic: float = Field(..., description="Bayesian Information Criterion")
    aic: float = Field(..., description="Akaike Information Criterion")
    
    # Structure learning metrics
    structure_learning_time: float = Field(..., description="Time for structure learning (seconds)")
    parameter_learning_time: float = Field(..., description="Time for parameter learning (seconds)")
    
    # Validation metrics (if ground truth available)
    structural_hamming_distance: Optional[int] = Field(
        default=None,
        description="Structural Hamming Distance from ground truth"
    )
    edge_precision: Optional[float] = Field(default=None, description="Edge precision")
    edge_recall: Optional[float] = Field(default=None, description="Edge recall")
    edge_f1: Optional[float] = Field(default=None, description="Edge F1 score")
    
    # Configuration used
    structure_config: StructureLearningConfig = Field(..., description="Structure learning config")
    parameter_config: ParameterLearningConfig = Field(..., description="Parameter learning config")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Posterior samples (for uncertainty quantification)
    posterior_samples_stored: bool = Field(default=False, description="Whether posterior samples are stored")


class GraphComparison(BaseModel):
    """Comparison between two causal graphs"""
    graph1_id: str = Field(..., description="First graph ID")
    graph2_id: str = Field(..., description="Second graph ID")
    
    # Edge differences
    edges_only_in_g1: List[Edge] = Field(default_factory=list)
    edges_only_in_g2: List[Edge] = Field(default_factory=list)
    common_edges: List[Edge] = Field(default_factory=list)
    
    # Metrics
    structural_hamming_distance: int = Field(..., description="SHD between graphs")
    jaccard_similarity: float = Field(..., description="Jaccard similarity of edge sets")
    
    # Node differences
    nodes_only_in_g1: List[str] = Field(default_factory=list)
    nodes_only_in_g2: List[str] = Field(default_factory=list)