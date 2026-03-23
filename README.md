# Interactive Causal Policy Optimization Lab

## A Platform for Causal Discovery, Intervention Analysis, Counterfactual Reasoning, and Policy Optimization

- Guided by Dr. R.J. Mohan Sahu and Developed by Satya Narayan Sahu

---

## Table of Contents

1. [Introduction and Overview](#1-introduction-and-overview)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Structural Causal Models (SCMs)](#4-structural-causal-models-scms)
5. [Causal Discovery Algorithms](#5-causal-discovery-algorithms)
6. [Intervention Analysis](#6-intervention-analysis)
7. [Counterfactual Reasoning](#7-counterfactual-reasoning)
8. [Policy Optimization](#8-policy-optimization)
9. [Implementation Details](#9-implementation-details)
10. [API Reference](#10-api-reference)
11. [Frontend Architecture](#11-frontend-architecture)
12. [Usage Examples](#12-usage-examples)
13. [Proofs and Derivations](#13-proofs-and-derivations)
14. [References](#14-references)

---

## 1. Introduction and Overview

### 1.1 What is Causal Inference?

Causal inference is the process of drawing conclusions about causal relationships from data. Unlike traditional statistical inference, which focuses on associations and correlations, causal inference aims to answer questions about "what happens if we intervene" or "what would have happened if things were different."

The fundamental distinction between correlation and causation can be illustrated through the famous saying: "Correlation does not imply causation." For example, ice cream sales and drowning deaths are correlated, but one does not cause the other—both are influenced by a common cause: hot weather.

### 1.2 The Ladder of Causality

Judea Pearl's "Ladder of Causality" provides a framework for understanding different levels of causal reasoning:

**Level 1 - Association (Seeing):**
- Questions: "What is P(Y|X)?"
- Example: "What is the probability of rain given cloudy skies?"
- Traditional statistics and machine learning operate at this level.

**Level 2 - Intervention (Doing):**
- Questions: "What is P(Y|do(X))?"
- Example: "What would happen to sales if we double the price?"
- Requires understanding of causal mechanisms.

**Level 3 - Counterfactuals (Imagining):**
- Questions: "What would Y be if X had been different?"
- Example: "Would this patient have recovered if they had taken the medication?"
- Requires structural knowledge of the system.

### 1.3 Why Causal Inference Matters

Causal inference is essential for:

1. **Policy Decisions**: Understanding the effects of interventions before implementing them
2. **Medical Research**: Determining treatment efficacy and safety
3. **Economics**: Evaluating the impact of economic policies
4. **Machine Learning**: Building robust and interpretable models
5. **Social Sciences**: Understanding complex social phenomena

### 1.4 System Overview

This Interactive Causal Policy Optimization Lab provides a comprehensive platform for:

- **Causal Discovery**: Learning causal structure from observational data
- **Intervention Analysis**: Computing the effects of interventions
- **Counterfactual Reasoning**: Answering "what if" questions
- **Policy Optimization**: Finding optimal intervention strategies

---

## 2. Theoretical Foundations

### 2.1 Probability Theory Fundamentals

Before diving into causal inference, we establish the probabilistic foundations.

#### 2.1.1 Probability Distributions

A probability distribution P over a set of random variables X₁, X₂, ..., Xₙ is defined by:

$$P: \Omega \rightarrow [0, 1]$$

where Ω is the sample space. The joint distribution is:

$$P(X_1, X_2, ..., X_n) = P(X_1) \cdot P(X_2|X_1) \cdot ... \cdot P(X_n|X_1, ..., X_{n-1})$$

#### 2.1.2 Conditional Independence

Variables X and Y are conditionally independent given Z, denoted X ⊥ Y | Z, if:

$$P(X, Y | Z) = P(X | Z) \cdot P(Y | Z)$$

This is equivalent to:

$$P(X | Y, Z) = P(X | Z)$$

#### 2.1.3 Bayes' Theorem

Bayes' theorem provides the foundation for updating beliefs:

$$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$

In causal inference, we extend this to handle interventions.

### 2.2 Graph Theory Fundamentals

#### 2.2.1 Directed Acyclic Graphs (DAGs)

A DAG G = (V, E) consists of:
- V: A set of vertices (nodes)
- E: A set of directed edges (arrows)
- No directed cycles

**Definition**: A directed cycle exists if there is a path from a node back to itself following the direction of arrows.

#### 2.2.2 Graphical Relationships

**Parent**: X is a parent of Y if there is a directed edge X → Y
$$Pa(Y) = \{X : X \rightarrow Y \in E\}$$

**Child**: Y is a child of X if X → Y
$$Ch(X) = \{Y : X \rightarrow Y \in E\}$$

**Ancestor**: X is an ancestor of Y if there is a directed path from X to Y
$$An(Y) = \{X : X \rightarrow ... \rightarrow Y\}$$

**Descendant**: Y is a descendant of X if X is an ancestor of Y
$$De(X) = \{Y : X \text{ is an ancestor of } Y\}$$

#### 2.2.3 D-Separation

D-separation is a graphical criterion for determining conditional independence:

**Definition**: A path p is blocked by a set Z if and only if:
1. p contains a chain i → m → j or a fork i ← m → j such that m ∈ Z, or
2. p contains a collider i → m ← j such that m ∉ Z and no descendant of m is in Z.

**Theorem**: If Z d-separates X and Y, then X ⊥ Y | Z in every distribution compatible with the DAG.

### 2.3 The Causal Markov Assumption

The Causal Markov Assumption (CMA) connects causal structure to probability:

**Assumption**: Given a DAG G and joint distribution P, if G represents the causal structure, then:

$$P(X_1, ..., X_n) = \prod_{i=1}^{n} P(X_i | Pa(X_i))$$

This factorization implies that each variable is independent of its non-descendants given its parents.

### 2.4 Faithfulness Assumption

The Faithfulness Assumption ensures that all conditional independencies in P are implied by the graph structure:

**Assumption**: If X ⊥ Y | Z in P, then Z d-separates X and Y in G.

Together, CMA and Faithfulness establish a one-to-one correspondence between graphical and probabilistic independencies.

---

## 3. Mathematical Framework

### 3.1 Structural Causal Models (SCMs)

A Structural Causal Model M consists of:

1. **Endogenous Variables**: V = {V₁, V₂, ..., Vₙ} - determined within the model
2. **Exogenous Variables**: U = {U₁, U₂, ..., Uₘ} - determined outside the model
3. **Structural Equations**: f = {f₁, f₂, ..., fₙ} - deterministic functions
4. **Probability Distribution**: P(U) over exogenous variables

Each endogenous variable is determined by:

$$V_i = f_i(Pa(V_i), U_i)$$

#### 3.1.1 Example: Simple SCM

Consider a simple treatment-outcome model:

```
U_X → X → Y ← U_Y
       ↑
      U_XY
```

Structural equations:
$$X = f_X(U_X, U_{XY})$$
$$Y = f_Y(X, U_Y, U_{XY})$$

### 3.2 The do-Operator

The do-operator formalizes interventions:

**Definition**: do(X = x) represents an intervention that sets X to value x, replacing the structural equation for X.

**Effect on SCM**:
- Original: X = f_X(Pa(X), U_X)
- After do(X = x): X = x (constant)

#### 3.2.1 Interventional Distribution

The interventional distribution P(Y | do(X = x)) is computed by:

$$P(Y | do(X = x)) = \sum_{u} P(Y | X = x, u) P(u)$$

This is equivalent to the truncated factorization:

$$P(Y | do(X = x)) = \sum_{v \setminus \{X\}} \prod_{V_i \neq X} P(V_i | Pa(V_i)) \bigg|_{X=x}$$

### 3.3 do-Calculus

Pearl's do-calculus provides three rules for transforming interventional distributions:

#### Rule 1: Insertion/Deletion of Observations

$$P(Y | do(X), Z, W) = P(Y | do(X), W)$$

if (Y ⊥ Z | X, W) in the graph where all arrows into X are removed.

**Proof Sketch**: This rule applies when Z provides no additional information about Y given X and W in the manipulated graph.

#### Rule 2: Action/Observation Exchange

$$P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)$$

if (Y ⊥ Z | X, W) in the graph where arrows into X and into Z are removed.

**Proof Sketch**: When Z is not a descendant of X, observing Z is equivalent to intervening on Z.

#### Rule 3: Insertion/Deletion of Actions

$$P(Y | do(X), do(Z), W) = P(Y | do(X), W)$$

if (Y ⊥ Z | X, W) in the graph where arrows into X are removed and arrows out of Z are removed.

**Proof Sketch**: This rule applies when Z has no effect on Y through any path.

### 3.4 Back-Door Criterion

The Back-Door Criterion provides a graphical test for identifiability:

**Definition**: A set Z satisfies the back-door criterion relative to (X, Y) if:
1. No node in Z is a descendant of X
2. Z blocks every path between X and Y that contains an arrow into X

**Theorem (Back-Door Adjustment)**: If Z satisfies the back-door criterion, then:

$$P(Y | do(X = x)) = \sum_{z} P(Y | X = x, Z = z) P(Z = z)$$

**Proof**:
1. By Rule 2 of do-calculus, since Z blocks all back-door paths:
   $$P(Y | do(X)) = \sum_z P(Y | X, Z = z) P(Z = z | do(X))$$

2. By Rule 3, since Z is not a descendant of X:
   $$P(Z = z | do(X)) = P(Z = z)$$

3. Combining:
   $$P(Y | do(X)) = \sum_z P(Y | X, Z = z) P(Z = z)$$

### 3.5 Front-Door Criterion

When no back-door admissible set exists, the Front-Door Criterion may apply:

**Definition**: A set Z satisfies the front-door criterion relative to (X, Y) if:
1. Z intercepts all directed paths from X to Y
2. There is no unblocked back-door path from X to Z
3. All back-door paths from Z to Y are blocked by X

**Theorem (Front-Door Adjustment)**:

$$P(Y | do(X = x)) = \sum_{z} P(Z = z | X = x) \sum_{x'} P(Y | X = x', Z = z) P(X = x')$$

---

## 4. Structural Causal Models (SCMs)

### 4.1 Formal Definition

A Structural Causal Model M = ⟨U, V, F, P(U)⟩ where:

- **U**: Exogenous (background) variables
- **V**: Endogenous (observed) variables  
- **F**: Set of structural functions {f₁, ..., fₙ}
- **P(U)**: Probability distribution over U

### 4.2 Structural Equations

Each endogenous variable Vᵢ has a structural equation:

$$V_i = f_i(Pa_i, U_i)$$

where:
- Paᵢ are the parents of Vᵢ in the causal graph
- Uᵢ is the exogenous noise term

#### 4.2.1 Linear Gaussian SCM

The most common parametric form:

$$V_i = \sum_{j \in Pa_i} \beta_{ij} V_j + \epsilon_i$$

where εᵢ ~ N(0, σᵢ²) are independent Gaussian noise terms.

**Implementation in Code**:

```python
class LinearGaussianSCM:
    def __init__(self, adjacency_matrix, coefficients, noise_std):
        self.W = adjacency_matrix  # n x n matrix
        self.beta = coefficients    # n x n matrix
        self.sigma = noise_std      # n-dimensional vector
    
    def sample(self, n_samples):
        n = len(self.sigma)
        U = np.random.normal(0, self.sigma, (n_samples, n))
        
        # Topological order for sampling
        order = self.topological_sort()
        V = np.zeros((n_samples, n))
        
        for i in order:
            parents = np.where(self.W[:, i] == 1)[0]
            V[:, i] = V[:, parents] @ self.beta[parents, i] + U[:, i]
        
        return V
```

### 4.3 Causal Graph Construction

The causal graph G is derived from the SCM:

- Nodes: One for each endogenous variable
- Edges: X → Y if X appears in f_Y

**Graph Construction Algorithm**:

```python
def build_causal_graph(structural_equations):
    """
    Build DAG from structural equations.
    
    Args:
        structural_equations: Dict mapping variable to (parents, function)
    
    Returns:
        networkx.DiGraph: Causal DAG
    """
    import networkx as nx
    
    G = nx.DiGraph()
    
    for var, (parents, _) in structural_equations.items():
        G.add_node(var)
        for parent in parents:
            G.add_edge(parent, var)
    
    # Verify acyclicity
    assert nx.is_directed_acyclic_graph(G), "Graph contains cycles!"
    
    return G
```

### 4.4 Interventions in SCMs

An intervention do(X = x) modifies the SCM:

**Original SCM**:
$$X = f_X(Pa_X, U_X)$$

**Modified SCM**:
$$X = x$$

**Code Implementation**:

```python
def intervene(scm, intervention_dict):
    """
    Apply interventions to an SCM.
    
    Args:
        scm: Original structural causal model
        intervention_dict: Dict mapping variables to intervention values
    
    Returns:
        Modified SCM with interventions applied
    """
    modified_scm = scm.copy()
    
    for var, value in intervention_dict.items():
        # Replace structural equation with constant
        modified_scm.equations[var] = lambda u: value
        # Remove incoming edges in graph
        for parent in modified_scm.graph.predecessors(var):
            modified_scm.graph.remove_edge(parent, var)
    
    return modified_scm
```

### 4.5 Counterfactuals in SCMs

Counterfactual reasoning requires three steps:

#### Step 1: Abduction

Compute posterior distribution over exogenous variables:

$$P(U | V = v) \propto P(V = v | U) P(U)$$

For linear Gaussian models:

$$U | V = v \sim N(\mu_{U|v}, \Sigma_{U|v})$$

#### Step 2: Action

Modify the SCM according to the counterfactual intervention:

$$M_{X \leftarrow x} = M \text{ with } f_X \text{ replaced by } X = x$$

#### Step 3: Prediction

Compute the counterfactual outcome:

$$Y_{X=x}(u) = f_Y^{modified}(Pa_Y^{modified}, u)$$

**Complete Counterfactual Implementation**:

```python
def compute_counterfactual(scm, observed_data, intervention, target_var):
    """
    Compute counterfactual value.
    
    Args:
        scm: Structural causal model
        observed_data: Observed values of endogenous variables
        intervention: Dict of counterfactual interventions
        target_var: Variable to compute counterfactual for
    
    Returns:
        Counterfactual value of target_var
    """
    # Step 1: Abduction - infer exogenous variables
    exogenous = {}
    for var in scm.endogenous_vars:
        parents = scm.get_parents(var)
        if len(parents) == 0:
            exogenous[var] = observed_data[var]
        else:
            predicted = scm.equations[var]([observed_data[p] for p in parents])
            exogenous[var] = observed_data[var] - predicted
    
    # Step 2: Action - modify SCM
    modified_scm = scm.copy()
    for var, value in intervention.items():
        modified_scm.equations[var] = lambda u: value
    
    # Step 3: Prediction - compute counterfactual
    counterfactual_values = {}
    for var in modified_scm.topological_order():
        if var in intervention:
            counterfactual_values[var] = intervention[var]
        else:
            parents = modified_scm.get_parents(var)
            parent_values = [counterfactual_values[p] for p in parents]
            counterfactual_values[var] = modified_scm.equations[var](
                parent_values + [exogenous[var]]
            )
    
    return counterfactual_values[target_var]
```

---

## 5. Causal Discovery Algorithms

### 5.1 Overview

Causal discovery aims to learn the causal graph from observational data. The main approaches are:

1. **Constraint-based**: Use conditional independence tests
2. **Score-based**: Search over graph space optimizing a score
3. **Hybrid**: Combine both approaches
4. **Continuous Optimization**: Differentiable approaches

### 5.2 PC Algorithm (Constraint-Based)

The PC algorithm learns the skeleton and orients edges using conditional independence tests.

#### 5.2.1 Algorithm Description

**Phase 1: Skeleton Discovery**

```
Algorithm: PC-Skeleton
Input: Data D, significance level α
Output: Undirected skeleton S, separation sets Sep

1. Start with complete undirected graph S
2. For depth = 0, 1, 2, ...:
   For each pair (X, Y) adjacent in S:
     For each subset Z ⊂ Adj(X) \ {Y} with |Z| = depth:
       Test X ⊥ Y | Z
       If independent:
         Remove edge X - Y from S
         Sep(X, Y) = Z
         Break
   If no edges removed at this depth, stop
3. Return S, Sep
```

**Phase 2: Edge Orientation**

```
Algorithm: PC-Orient
Input: Skeleton S, separation sets Sep
Output: Partially directed DAG

1. For each unshielded triple X - Z - Y:
   If Z ∉ Sep(X, Y):
     Orient as X → Z ← Y (v-structure)

2. Apply Meek's orientation rules:
   R1: If X → Y - Z and X, Z not adjacent: Y → Z
   R2: If X → Y → Z and X - Z: X → Z
   R3: If X - Y, X - Z, Y → W, Z → W, X, W not adjacent: X → W
```

#### 5.2.2 Mathematical Foundation

**Conditional Independence Testing**:

For continuous variables, use partial correlation:

$$\rho_{XY|Z} = \frac{\rho_{XY} - \rho_{XZ}\rho_{YZ}}{\sqrt{(1-\rho_{XZ}^2)(1-\rho_{YZ}^2)}}$$

Fisher's z-transformation for testing:

$$z = \frac{1}{2} \ln\left(\frac{1+r}{1-r}\right) \approx \text{arctanh}(r)$$

Under H₀: ρ = 0:

$$\sqrt{n - |Z| - 3} \cdot z \sim N(0, 1)$$

#### 5.2.3 Implementation

```python
import numpy as np
from scipy import stats
from itertools import combinations

def pc_algorithm(data, alpha=0.05):
    """
    PC algorithm for causal discovery.
    
    Args:
        data: pandas DataFrame with n samples and p variables
        alpha: Significance level for independence tests
    
    Returns:
        networkx.DiGraph: Learned causal DAG
    """
    n_vars = data.shape[1]
    variables = list(data.columns)
    
    # Initialize complete graph
    skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)
    sep_sets = {}
    
    # Phase 1: Skeleton discovery
    depth = 0
    while depth < n_vars - 1:
        edges_removed = False
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] == 0:
                    continue
                
                # Get adjacent nodes
                adj_i = [k for k in range(n_vars) if skeleton[i, k] == 1 and k != j]
                
                # Test all conditioning sets of size 'depth'
                for z_idx in combinations(adj_i, depth):
                    z = [variables[k] for k in z_idx]
                    
                    # Partial correlation test
                    corr, p_value = partial_correlation_test(
                        data[variables[i]], 
                        data[variables[j]], 
                        data[z] if z else None
                    )
                    
                    if p_value > alpha:  # Independent
                        skeleton[i, j] = 0
                        skeleton[j, i] = 0
                        sep_sets[(i, j)] = z_idx
                        sep_sets[(j, i)] = z_idx
                        edges_removed = True
                        break
        
        if not edges_removed:
            break
        depth += 1
    
    # Phase 2: Orient edges (v-structures)
    directed = np.zeros((n_vars, n_vars))
    
    for k in range(n_vars):
        adj_k = [i for i in range(n_vars) if skeleton[i, k] == 1]
        
        for idx_i, i in enumerate(adj_k):
            for j in adj_k[idx_i + 1:]:
                if skeleton[i, j] == 0:  # Not adjacent
                    sep = sep_sets.get((i, j), set())
                    if k not in sep:
                        # Orient as v-structure
                        directed[i, k] = 1
                        directed[j, k] = 1
    
    # Apply Meek's rules
    directed = apply_meek_rules(skeleton, directed)
    
    return build_graph_from_matrices(skeleton, directed, variables)


def partial_correlation_test(x, y, z=None):
    """
    Compute partial correlation and p-value.
    
    Args:
        x, y: Variables to test
        z: Conditioning set (optional)
    
    Returns:
        correlation, p_value
    """
    if z is None or len(z.columns) == 0:
        corr, p_value = stats.pearsonr(x, y)
    else:
        # Regress x on z
        from sklearn.linear_model import LinearRegression
        lr_x = LinearRegression().fit(z, x)
        res_x = x - lr_x.predict(z)
        
        # Regress y on z
        lr_y = LinearRegression().fit(z, y)
        res_y = y - lr_y.predict(z)
        
        # Correlation of residuals
        corr, p_value = stats.pearsonr(res_x, res_y)
    
    return corr, p_value
```

### 5.3 GES Algorithm (Score-Based)

Greedy Equivalence Search optimizes a score over equivalence classes of DAGs.

#### 5.3.1 Scoring Functions

**BIC Score (Bayesian Information Criterion)**:

$$\text{BIC}(G | D) = -2 \ell(D | G) + k \log(n)$$

where:
- ℓ(D|G) is the log-likelihood
- k is the number of parameters
- n is the sample size

For linear Gaussian models:

$$\ell(D | G) = -\frac{n}{2} \sum_{i=1}^{p} \left[ \log(2\pi \sigma_i^2) + \frac{\text{RSS}_i}{\sigma_i^2} \right]$$

**BDeu Score (Bayesian Dirichlet equivalent uniform)**:

For discrete variables:

$$\text{BDeu}(G | D) = \sum_{i=1}^{p} \sum_{j=1}^{q_i} \left[ \log \frac{\Gamma(\alpha_{ij})}{\Gamma(\alpha_{ij} + n_{ij})} + \sum_{k=1}^{r_i} \log \frac{\Gamma(\alpha_{ijk} + n_{ijk})}{\Gamma(\alpha_{ijk})} \right]$$

where:
- qᵢ is the number of parent configurations for Xᵢ
- rᵢ is the number of states of Xᵢ
- α parameters are prior pseudo-counts

#### 5.3.2 Algorithm Description

```
Algorithm: GES
Input: Data D, score function S
Output: DAG G*

1. Forward Phase:
   G = empty graph
   While improving:
     For each edge addition e:
       Compute ΔS(e) = S(G + e) - S(G)
     e* = argmax ΔS(e)
     If ΔS(e*) > 0:
       G = G + e*

2. Backward Phase:
   While improving:
     For each edge deletion e:
       Compute ΔS(e) = S(G - e) - S(G)
     e* = argmax ΔS(e)
     If ΔS(e*) > 0:
       G = G - e*

3. Return G
```

#### 5.3.3 Implementation

```python
def ges_algorithm(data, score='bic'):
    """
    Greedy Equivalence Search for causal discovery.
    
    Args:
        data: pandas DataFrame
        score: Scoring function ('bic' or 'aic')
    
    Returns:
        networkx.DiGraph: Learned DAG
    """
    n_vars = data.shape[1]
    variables = list(data.columns)
    
    # Initialize empty graph
    adj_matrix = np.zeros((n_vars, n_vars))
    current_score = compute_score(data, adj_matrix, score)
    
    # Forward phase
    while True:
        best_delta = 0
        best_edge = None
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and adj_matrix[i, j] == 0:
                    # Try adding edge i -> j
                    adj_matrix[i, j] = 1
                    
                    if is_dag(adj_matrix):
                        new_score = compute_score(data, adj_matrix, score)
                        delta = new_score - current_score
                        
                        if delta > best_delta:
                            best_delta = delta
                            best_edge = (i, j, 'add')
                    
                    adj_matrix[i, j] = 0
        
        if best_edge is None:
            break
        
        i, j, _ = best_edge
        adj_matrix[i, j] = 1
        current_score += best_delta
    
    # Backward phase
    while True:
        best_delta = 0
        best_edge = None
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adj_matrix[i, j] == 1:
                    # Try removing edge i -> j
                    adj_matrix[i, j] = 0
                    new_score = compute_score(data, adj_matrix, score)
                    delta = new_score - current_score
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j, 'remove')
                    
                    adj_matrix[i, j] = 1
        
        if best_edge is None:
            break
        
        i, j, _ = best_edge
        adj_matrix[i, j] = 0
        current_score += best_delta
    
    return build_graph_from_adjacency(adj_matrix, variables)


def compute_score(data, adj_matrix, score_type='bic'):
    """
    Compute BIC or AIC score for a graph.
    """
    n_samples = len(data)
    n_vars = len(data.columns)
    
    log_likelihood = 0
    n_params = 0
    
    for j in range(n_vars):
        parents = np.where(adj_matrix[:, j] == 1)[0]
        y = data.iloc[:, j].values
        
        if len(parents) == 0:
            # No parents - just variance
            rss = np.sum((y - y.mean())**2)
            log_likelihood += -0.5 * n_samples * np.log(rss / n_samples + 1e-10)
            n_params += 1
        else:
            # Linear regression
            X = data.iloc[:, parents].values
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X, y)
            residuals = y - lr.predict(X)
            rss = np.sum(residuals**2)
            
            log_likelihood += -0.5 * n_samples * np.log(rss / n_samples + 1e-10)
            n_params += len(parents) + 1  # coefficients + intercept
    
    if score_type == 'bic':
        return -2 * log_likelihood + n_params * np.log(n_samples)
    else:  # AIC
        return -2 * log_likelihood + 2 * n_params


def is_dag(adj_matrix):
    """
    Check if adjacency matrix represents a DAG.
    
    Uses the trace of exp(A ◦ A) - n = 0 criterion.
    """
    from scipy.linalg import expm
    n = adj_matrix.shape[0]
    E = expm(adj_matrix * adj_matrix)
    return np.trace(E) - n < 1e-6
```

### 5.4 NOTEARS Algorithm (Continuous Optimization)

NOTEARS reformulates DAG learning as a continuous optimization problem.

#### 5.4.1 Mathematical Formulation

**Objective Function**:

$$\min_{W} \frac{1}{2n} \|X - XW\|_F^2 + \lambda \|W\|_1$$

subject to:

$$h(W) = \text{tr}(e^{W \circ W}) - d = 0$$

where:
- W is the weighted adjacency matrix
- ◦ denotes element-wise product
- d is the number of variables

**Acyclicity Constraint Derivation**:

The constraint h(W) = 0 ensures acyclicity because:

$$e^{A} = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

For a DAG, Aᵏ = 0 for k ≥ d (no paths of length ≥ d exist).

Therefore:

$$\text{tr}(e^{A \circ A}) = \text{tr}(I + A \circ A + \frac{(A \circ A)^2}{2!} + ...) = d$$

iff A represents a DAG.

#### 5.4.2 Augmented Lagrangian Method

The constrained problem is solved using augmented Lagrangian:

$$\mathcal{L}(W, \lambda) = \frac{1}{2n} \|X - XW\|_F^2 + \lambda \|W\|_1 + \frac{\rho}{2} h(W)^2 + \alpha h(W)$$

**Gradient Computation**:

$$\frac{\partial h}{\partial W} = (e^{W \circ W})^T \circ 2W$$

#### 5.4.3 Implementation

```python
import numpy as np
from scipy.linalg import expm

def notears_algorithm(X, lambda_l1=0.1, lambda_dag=10, max_iter=100, lr=0.01):
    """
    NOTEARS: Non-combinatorial optimization for DAG learning.
    
    Args:
        X: Data matrix (n_samples x n_features)
        lambda_l1: L1 regularization parameter
        lambda_dag: DAG penalty parameter
        max_iter: Maximum iterations
        lr: Learning rate
    
    Returns:
        W: Learned weighted adjacency matrix
    """
    n_samples, n_vars = X.shape
    
    # Standardize data
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Initialize weight matrix
    W = np.random.randn(n_vars, n_vars) * 0.01
    W = (W + W.T) / 2  # Symmetric initialization
    np.fill_diagonal(W, 0)  # No self-loops
    
    # Dual variables for augmented Lagrangian
    alpha = 0.0  # Lagrange multiplier
    rho = 1.0    # Penalty parameter
    
    for iteration in range(max_iter):
        # Compute gradients
        loss = 0.5 * np.sum((X - X @ W) ** 2) / n_samples
        
        # Gradient of loss
        grad_loss = -X.T @ (X - X @ W) / n_samples
        
        # Acyclicity constraint and gradient
        E = expm(W * W)
        h = np.trace(E)
        
        # Gradient of h(W)
        grad_h = (E.T * 2 * W)
        
        # Total gradient
        grad = grad_loss + lambda_l1 * np.sign(W) + alpha * grad_h + rho * h * grad_h
        
        # Update weights
        W = W - lr * grad
        np.fill_diagonal(W, 0)  # No self-loops
        
        # Update dual variables
        if h < 0.25:
            rho = rho * 2
        alpha = alpha + rho * h
        
        # Check convergence
        if h < 1e-8:
            break
    
    # Threshold small values
    W[np.abs(W) < 0.3] = 0
    
    return W
```

### 5.5 LiNGAM (Linear Non-Gaussian Acyclic Model)

LiNGAM exploits non-Gaussianity to uniquely identify causal direction.

#### 5.5.1 Mathematical Foundation

**Key Insight**: In linear models with non-Gaussian noise, causal direction is identifiable.

**Model**:
$$X_i = \sum_{j \in Pa_i} \beta_{ij} X_j + \epsilon_i$$

where εᵢ are non-Gaussian and independent.

**Identifiability Theorem**: Under the LiNGAM assumptions, the causal ordering and edge weights are uniquely identifiable from observational data.

#### 5.5.2 ICA-Based LiNGAM

```
Algorithm: ICA-LiNGAM
Input: Data matrix X (n x p)
Output: Adjacency matrix B, causal order

1. Center the data: X = X - mean(X)
2. Apply ICA: X = AS where A is mixing matrix
3. Find permutation P such that P·A⁻¹ is close to lower triangular
4. Estimate B from the permuted mixing matrix
5. Prune small edges
```

**Implementation**:

```python
from sklearn.decomposition import FastICA

def lingam_algorithm(X):
    """
    ICA-based LiNGAM for causal discovery.
    
    Args:
        X: Data matrix (n_samples x n_features)
    
    Returns:
        B: Causal adjacency matrix
        order: Causal ordering of variables
    """
    n_samples, n_vars = X.shape
    
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # Apply ICA
    ica = FastICA(n_components=n_vars, random_state=42)
    S = ica.fit_transform(X_centered)
    A = ica.mixing_  # Mixing matrix
    
    # Find causal ordering
    # Use row-wise ratio test
    A_inv = np.linalg.inv(A)
    
    # Find permutation to make A_inv lower triangular
    order = find_causal_order(np.abs(A_inv))
    
    # Reorder and estimate B
    B = np.zeros((n_vars, n_vars))
    for i, target_idx in enumerate(order):
        for j in range(i):
            source_idx = order[j]
            # Regress X[target] on X[sources before it]
            y = X[:, target_idx]
            X_pred = X[:, order[:i]]
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X_pred, y)
            B[source_idx, target_idx] = lr.coef_[j]
    
    return B, order


def find_causal_order(W):
    """
    Find causal ordering from ICA unmixing matrix.
    
    Uses the criterion that rows corresponding to 
    earlier variables have smaller sums.
    """
    n = W.shape[0]
    row_sums = W.sum(axis=1)
    order = np.argsort(row_sums)[::-1]
    return order
```

### 5.6 FCI Algorithm (Handling Latent Confounders)

The Fast Causal Inference (FCI) algorithm handles unmeasured confounders.

#### 5.6.1 Overview

FCI outputs a Partial Ancestral Graph (PAG) that represents:
- Directed edges (→): Direct causation
- Bidirected edges (↔): Latent confounder
- Undirected edges (—): Ancestral relationship

#### 5.6.2 Algorithm Steps

```
Algorithm: FCI
Input: Data D, significance level α
Output: PAG

1. Run PC algorithm to get initial skeleton
2. Determine possible d-separation sets
3. Test for discriminating paths
4. Orient edges using FCI orientation rules
5. Return PAG
```

### 5.7 Bootstrap Confidence

Assess uncertainty in learned structure using bootstrap:

```python
def bootstrap_confidence(data, algorithm, n_bootstrap=100):
    """
    Compute bootstrap confidence for edges.
    
    Args:
        data: pandas DataFrame
        algorithm: Causal discovery function
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Edge confidence matrix
    """
    n_vars = data.shape[1]
    edge_counts = np.zeros((n_vars, n_vars))
    
    for b in range(n_bootstrap):
        # Resample with replacement
        sample_idx = np.random.choice(len(data), size=len(data), replace=True)
        sample = data.iloc[sample_idx]
        
        # Learn structure
        graph = algorithm(sample)
        
        # Count edges
        for i in range(n_vars):
            for j in range(n_vars):
                if graph.has_edge(data.columns[i], data.columns[j]):
                    edge_counts[i, j] += 1
    
    # Normalize
    confidence = edge_counts / n_bootstrap
    
    return confidence
```

---

## 6. Intervention Analysis

### 6.1 The do-Operator

The do-operator formalizes the notion of intervention:

**Definition**: do(X = x) represents an intervention that sets X to value x by replacing its structural equation.

**Graphical Effect**: Remove all incoming edges to X in the causal graph.

### 6.2 Interventional Distribution

The interventional distribution P(Y | do(X = x)) differs from conditional P(Y | X = x):

$$P(Y | do(X = x)) \neq P(Y | X = x) \text{ in general}$$

**When are they equal?** When there are no back-door paths from X to Y.

### 6.3 Back-Door Adjustment

**Theorem**: If Z satisfies the back-door criterion relative to (X, Y):

$$P(Y | do(X = x)) = \sum_z P(Y | X = x, Z = z) P(Z = z)$$

**Implementation**:

```python
def backdoor_adjustment(data, treatment, outcome, adjustment_set):
    """
    Compute causal effect using back-door adjustment.
    
    Args:
        data: pandas DataFrame
        treatment: Treatment variable name
        outcome: Outcome variable name
        adjustment_set: List of adjustment variables
    
    Returns:
        Causal effect estimate
    """
    from sklearn.linear_model import LinearRegression
    
    # Fit outcome model: E[Y | X, Z]
    X_features = data[[treatment] + adjustment_set]
    y = data[outcome]
    
    model = LinearRegression().fit(X_features, y)
    
    # Compute E[Y | do(X = x)] for each x
    treatment_values = data[treatment].unique()
    effects = {}
    
    for x in treatment_values:
        # Create counterfactual dataset
        cf_data = data.copy()
        cf_data[treatment] = x
        
        # Predict outcomes
        X_cf = cf_data[[treatment] + adjustment_set]
        y_pred = model.predict(X_cf)
        
        # Average over adjustment set
        effects[x] = y_pred.mean()
    
    return effects
```

### 6.4 Propensity Score Methods

When the adjustment set is high-dimensional, propensity scores help:

**Propensity Score**: 
$$e(x) = P(T = 1 | X = x)$$

**Inverse Probability Weighting (IPW)**:

$$E[Y(1)] = \frac{1}{n} \sum_{i: T_i = 1} \frac{Y_i}{e(X_i)}$$

**Implementation**:

```python
def propensity_score_weighting(data, treatment, outcome, confounders):
    """
    Estimate causal effect using IPW.
    
    Args:
        data: pandas DataFrame
        treatment: Binary treatment variable
        outcome: Outcome variable
        confounders: List of confounding variables
    
    Returns:
        Average treatment effect (ATE)
    """
    from sklearn.linear_model import LogisticRegression
    
    # Estimate propensity scores
    X = data[confounders]
    T = data[treatment]
    
    ps_model = LogisticRegression().fit(X, T)
    propensity_scores = ps_model.predict_proba(X)[:, 1]
    
    # Clip extreme propensity scores
    propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
    
    # Compute weighted outcomes
    treated = data[treatment] == 1
    
    # IPW estimator
    y1_weighted = (data.loc[treated, outcome] / propensity_scores[treated]).mean()
    y0_weighted = (data.loc[~treated, outcome] / (1 - propensity_scores[~treated])).mean()
    
    ate = y1_weighted - y0_weighted
    
    return ate
```

### 6.5 Doubly Robust Estimation

Combines outcome modeling and propensity scores:

$$\hat{\tau}_{DR} = \frac{1}{n} \sum_{i=1}^{n} \left[ \hat{\mu}_1(X_i) + \frac{T_i(Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \hat{\mu}_0(X_i) - \frac{(1-T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)} \right]$$

**Implementation**:

```python
def doubly_robust_estimation(data, treatment, outcome, confounders):
    """
    Doubly robust ATE estimation.
    
    Advantages:
    - Consistent if either outcome model OR propensity model is correct
    - Lower variance than IPW
    """
    from sklearn.linear_model import LogisticRegression, LinearRegression
    
    X = data[confounders]
    T = data[treatment]
    Y = data[outcome]
    
    # Fit propensity score model
    ps_model = LogisticRegression().fit(X, T)
    e = ps_model.predict_proba(X)[:, 1]
    e = np.clip(e, 0.01, 0.99)
    
    # Fit outcome models
    treated_idx = T == 1
    
    mu1_model = LinearRegression().fit(X[treated_idx], Y[treated_idx])
    mu0_model = LinearRegression().fit(X[~treated_idx], Y[~treated_idx])
    
    mu1 = mu1_model.predict(X)
    mu0 = mu0_model.predict(X)
    
    # Doubly robust estimator
    dr1 = mu1 + T * (Y - mu1) / e
    dr0 = mu0 + (1 - T) * (Y - mu0) / (1 - e)
    
    ate = (dr1 - dr0).mean()
    
    return ate
```

### 6.6 Front-Door Adjustment

When back-door adjustment is not possible but a front-door exists:

**Front-Door Criterion**: A set M satisfies the front-door criterion relative to (X, Y) if:
1. M intercepts all directed paths from X to Y
2. There is no unblocked back-door path from X to M
3. All back-door paths from M to Y are blocked by X

**Front-Door Adjustment Formula**:

$$P(Y | do(X = x)) = \sum_m P(M = m | X = x) \sum_{x'} P(Y | X = x', M = m) P(X = x')$$

### 6.7 Instrumental Variables

When there are unmeasured confounders, instrumental variables can help:

**Instrument Conditions**:
1. Relevance: Z affects X
2. Exclusion: Z affects Y only through X
3. Independence: Z is independent of unmeasured confounders

**Two-Stage Least Squares (2SLS)**:

```python
def two_stage_least_squares(data, outcome, treatment, instrument, controls=None):
    """
    2SLS estimation for instrumental variable analysis.
    
    Args:
        data: pandas DataFrame
        outcome: Outcome variable
        treatment: Endogenous treatment
        instrument: Instrumental variable
        controls: Optional control variables
    
    Returns:
        IV estimate of treatment effect
    """
    from sklearn.linear_model import LinearRegression
    
    # First stage: Regress treatment on instrument
    if controls:
        X_first = data[[instrument] + controls]
    else:
        X_first = data[[instrument]]
    
    first_stage = LinearRegression().fit(X_first, data[treatment])
    treatment_hat = first_stage.predict(X_first)
    
    # Second stage: Regress outcome on predicted treatment
    if controls:
        X_second = np.column_stack([treatment_hat, data[controls]])
    else:
        X_second = treatment_hat.reshape(-1, 1)
    
    second_stage = LinearRegression().fit(X_second, data[outcome])
    
    return second_stage.coef_[0]
```

---

## 7. Counterfactual Reasoning

### 7.1 Three-Step Process

Counterfactual reasoning answers: "What would Y have been if X had been x?"

**Step 1: Abduction** - Infer exogenous variables from observations
**Step 2: Action** - Modify the SCM with the counterfactual intervention
**Step 3: Prediction** - Compute the counterfactual outcome

### 7.2 Linear Gaussian Counterfactuals

For linear Gaussian SCMs, counterfactuals have closed-form solutions.

**Model**:
$$Y = \beta X + \epsilon_Y$$
$$X = \gamma Z + \epsilon_X$$

**Counterfactual Computation**:

Given observed (X = x, Y = y), compute Y if X had been x':

1. **Abduction**: 
   $$\epsilon_Y = y - \beta x$$

2. **Action**: Set X = x'

3. **Prediction**:
   $$Y_{X=x'} = \beta x' + \epsilon_Y = \beta x' + (y - \beta x) = y + \beta(x' - x)$$

**Implementation**:

```python
class LinearGaussianCounterfactual:
    """
    Counterfactual reasoning for linear Gaussian SCMs.
    """
    
    def __init__(self, coefficients, noise_std):
        """
        Args:
            coefficients: Dict mapping (parent, child) to coefficient
            noise_std: Dict mapping variable to noise std
        """
        self.coef = coefficients
        self.noise_std = noise_std
    
    def abduction(self, observed_values, topological_order, parents):
        """
        Infer exogenous noise from observations.
        
        Args:
            observed_values: Dict of observed variable values
            topological_order: List of variables in topological order
            parents: Dict mapping variable to list of parents
        
        Returns:
            Dict of inferred exogenous values
        """
        exogenous = {}
        
        for var in topological_order:
            if len(parents.get(var, [])) == 0:
                # Root node: exogenous = observed
                exogenous[var] = observed_values[var]
            else:
                # Compute predicted value
                predicted = sum(
                    self.coef.get((p, var), 0) * observed_values[p]
                    for p in parents[var]
                )
                # Exogenous = observed - predicted
                exogenous[var] = observed_values[var] - predicted
        
        return exogenous
    
    def predict_counterfactual(self, intervention, exogenous, topological_order, parents):
        """
        Compute counterfactual values.
        
        Args:
            intervention: Dict of counterfactual interventions
            exogenous: Dict of inferred exogenous values
            topological_order: List of variables in topological order
            parents: Dict mapping variable to list of parents
        
        Returns:
            Dict of counterfactual values
        """
        cf_values = {}
        
        for var in topological_order:
            if var in intervention:
                # Intervened variable
                cf_values[var] = intervention[var]
            elif len(parents.get(var, [])) == 0:
                # Root node
                cf_values[var] = exogenous[var]
            else:
                # Compute from parents
                cf_values[var] = sum(
                    self.coef.get((p, var), 0) * cf_values[p]
                    for p in parents[var]
                ) + exogenous[var]
        
        return cf_values
```

### 7.3 Non-Linear Counterfactuals

For non-linear models, counterfactuals require more sophisticated methods:

#### 7.3.1 Neural Network SCMs

```python
import torch
import torch.nn as nn

class NeuralSCM(nn.Module):
    """
    Neural network-based structural causal model.
    """
    
    def __init__(self, n_vars, hidden_dim=64):
        super().__init__()
        
        # Each variable has its own neural network
        self.networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_vars, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(n_vars)
        ])
        
        # Adjacency mask
        self.adj_mask = nn.Parameter(torch.zeros(n_vars, n_vars), requires_grad=False)
    
    def forward(self, U, intervention=None):
        """
        Generate samples from the SCM.
        
        Args:
            U: Exogenous noise (batch_size x n_vars)
            intervention: Dict mapping variable index to intervention value
        
        Returns:
            Endogenous variable values
        """
        batch_size = U.shape[0]
        n_vars = U.shape[1]
        
        V = torch.zeros(batch_size, n_vars)
        
        # Process in topological order
        for i in range(n_vars):
            if intervention and i in intervention:
                V[:, i] = intervention[i]
            else:
                # Get parent values
                parent_input = V.clone()
                # Apply adjacency mask
                masked_input = parent_input * self.adj_mask[i, :]
                # Add noise
                network_input = torch.cat([masked_input, U[:, i:i+1]], dim=1)
                V[:, i] = self.networks[i](network_input).squeeze()
        
        return V
    
    def counterfactual(self, observed, intervention, target_var):
        """
        Compute counterfactual using neural SCM.
        
        Args:
            observed: Observed values
            intervention: Counterfactual intervention
            target_var: Target variable index
        
        Returns:
            Counterfactual value
        """
        # Step 1: Abduction - infer U from observed
        U = self.infer_exogenous(observed)
        
        # Step 2 & 3: Action and Prediction
        cf_value = self.forward(U, intervention=intervention)
        
        return cf_value[:, target_var]
    
    def infer_exogenous(self, observed):
        """
        Infer exogenous noise given observations.
        
        Uses gradient descent to find U that generates observed V.
        """
        U = torch.zeros_like(observed, requires_grad=True)
        optimizer = torch.optim.Adam([U], lr=0.01)
        
        for _ in range(1000):
            V_pred = self.forward(U)
            loss = ((V_pred - observed) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss < 1e-6:
                break
        
        return U.detach()
```

### 7.4 Necessary and Sufficient Causes

**Necessary Cause**: X is a necessary cause of Y if Y would not have occurred without X.

$$P(Y_{X=0} = 0 | X = 1, Y = 1)$$

**Sufficient Cause**: X is a sufficient cause of Y if X alone can cause Y.

$$P(Y_{X=1} = 1 | X = 0, Y = 0)$$

**Probability of Necessity (PN)**:

$$PN = P(Y_{X=0} = 0 | X = 1, Y = 1)$$

**Probability of Sufficiency (PS)**:

$$PS = P(Y_{X=1} = 1 | X = 0, Y = 0)$$

**Probability of Necessity and Sufficiency (PNS)**:

$$PNS = P(Y_{X=1} = 1, Y_{X=0} = 0)$$

### 7.5 Mediation Analysis

Decompose total effect into direct and indirect effects:

**Total Effect (TE)**:
$$TE = E[Y_{X=1} - Y_{X=0}]$$

**Natural Direct Effect (NDE)**:
$$NDE = E[Y_{X=1, M_{X=0}} - Y_{X=0}]$$

**Natural Indirect Effect (NIE)**:
$$NIE = E[Y_{X=0, M_{X=1}} - Y_{X=0}]$$

**Mediation Formula**:

$$NIE = \sum_m [E[Y | X=0, M=m] - E[Y | X=1, M=m]] P(M=m | X=1)$$

**Implementation**:

```python
def mediation_analysis(data, treatment, mediator, outcome, confounders):
    """
    Perform mediation analysis.
    
    Args:
        data: pandas DataFrame
        treatment: Treatment variable
        mediator: Mediator variable
        outcome: Outcome variable
        confounders: List of confounders
    
    Returns:
        Dict with total, direct, and indirect effects
    """
    from sklearn.linear_model import LinearRegression
    
    # Model 1: Mediator model
    X_m = data[[treatment] + confounders]
    mediator_model = LinearRegression().fit(X_m, data[mediator])
    
    # Model 2: Outcome model
    X_y = data[[treatment, mediator] + confounders]
    outcome_model = LinearRegression().fit(X_y, data[outcome])
    
    # Total effect
    X_te = data[[treatment] + confounders]
    te_model = LinearRegression().fit(X_te, data[outcome])
    total_effect = te_model.coef_[0]
    
    # Direct effect (effect of X on Y controlling for M)
    direct_effect = outcome_model.coef_[0]
    
    # Indirect effect (effect through M)
    # = effect of X on M * effect of M on Y
    indirect_effect = mediator_model.coef_[0] * outcome_model.coef_[1]
    
    # Proportion mediated
    prop_mediated = indirect_effect / total_effect if total_effect != 0 else 0
    
    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'proportion_mediated': prop_mediated
    }
```

---

## 8. Policy Optimization

### 8.1 Problem Formulation

Given a causal model, find optimal interventions to maximize expected utility:

$$\max_{a \in \mathcal{A}} E[U(Y) | do(a)]$$

subject to constraints:
$$g(a) \leq 0$$
$$h(a) = 0$$

### 8.2 Expected Utility Under Intervention

For a linear SCM:

$$E[Y | do(X = x)] = \beta_0 + \sum_{i \in I} \beta_i x_i$$

where I is the set of intervention variables.

### 8.3 Constrained Optimization

**Budget Constraint**:
$$\sum_{i} c_i |x_i - x_i^{baseline}| \leq B$$

**Implementation**:

```python
from scipy.optimize import minimize
import numpy as np

def optimize_policy(scm, target_variable, intervention_variables, 
                    objective='maximize', budget=None, costs=None,
                    bounds=None, constraints=None):
    """
    Find optimal intervention values.
    
    Args:
        scm: Structural causal model
        target_variable: Variable to optimize
        intervention_variables: Variables that can be intervened on
        objective: 'maximize' or 'minimize'
        budget: Total budget constraint
        costs: Cost per unit change for each intervention variable
        bounds: Dict mapping variable to (min, max) bounds
        constraints: Additional constraints
    
    Returns:
        Optimal intervention values and expected outcome
    """
    n_interventions = len(intervention_variables)
    
    # Objective function
    def objective_fn(x):
        # Compute expected outcome under intervention
        intervention_dict = dict(zip(intervention_variables, x))
        expected_y = scm.compute_interventional_expectation(
            intervention_dict, target_variable
        )
        return -expected_y if objective == 'maximize' else expected_y
    
    # Budget constraint
    constraint_list = []
    
    if budget is not None and costs is not None:
        def budget_constraint(x):
            total_cost = sum(costs[i] * abs(x[i]) for i in range(n_interventions))
            return budget - total_cost
        
        constraint_list.append({
            'type': 'ineq',
            'fun': budget_constraint
        })
    
    # Add custom constraints
    if constraints:
        constraint_list.extend(constraints)
    
    # Bounds
    if bounds:
        var_bounds = [bounds.get(v, (None, None)) for v in intervention_variables]
    else:
        var_bounds = [(None, None)] * n_interventions
    
    # Initial guess (baseline values)
    x0 = np.zeros(n_interventions)
    
    # Optimize
    result = minimize(
        objective_fn,
        x0,
        method='SLSQP',
        bounds=var_bounds,
        constraints=constraint_list,
        options={'maxiter': 1000}
    )
    
    if result.success:
        optimal_interventions = dict(zip(intervention_variables, result.x))
        optimal_value = -result.fun if objective == 'maximize' else result.fun
        
        return {
            'optimal_interventions': optimal_interventions,
            'optimal_value': optimal_value,
            'success': True
        }
    else:
        return {
            'success': False,
            'message': result.message
        }
```

### 8.4 Multi-Objective Optimization

When there are multiple objectives:

**Pareto Frontier**: Set of solutions where no objective can be improved without worsening another.

```python
def pareto_optimization(scm, target_variables, intervention_variables, 
                        weights=None, n_points=50):
    """
    Compute Pareto frontier for multi-objective optimization.
    
    Args:
        scm: Structural causal model
        target_variables: List of target variables
        intervention_variables: Variables that can be intervened on
        weights: If provided, scalarize objectives
        n_points: Number of Pareto points to compute
    
    Returns:
        List of Pareto optimal solutions
    """
    pareto_frontier = []
    
    if weights is None:
        # Generate different weight combinations
        weight_combinations = []
        for w1 in np.linspace(0, 1, n_points):
            w2 = 1 - w1
            weight_combinations.append([w1, w2])
    else:
        weight_combinations = [weights]
    
    for w in weight_combinations:
        def scalarized_objective(x):
            intervention_dict = dict(zip(intervention_variables, x))
            values = [
                scm.compute_interventional_expectation(intervention_dict, t)
                for t in target_variables
            ]
            # Weighted sum
            return -sum(wi * vi for wi, vi in zip(w, values))
        
        result = minimize(
            scalarized_objective,
            x0=np.zeros(len(intervention_variables)),
            method='SLSQP'
        )
        
        if result.success:
            pareto_frontier.append({
                'interventions': dict(zip(intervention_variables, result.x)),
                'objectives': {
                    t: scm.compute_interventional_expectation(
                        dict(zip(intervention_variables, result.x)), t
                    ) for t in target_variables
                }
            })
    
    return pareto_frontier
```

### 8.5 Robust Policy Optimization

Account for uncertainty in causal parameters:

```python
def robust_optimization(scm, target_variable, intervention_variables,
                        uncertainty_set='ellipsoidal', radius=1.0):
    """
    Robust optimization under model uncertainty.
    
    Args:
        scm: Structural causal model with parameter uncertainty
        target_variable: Variable to optimize
        intervention_variables: Variables that can be intervened on
        uncertainty_set: Type of uncertainty set
        radius: Size of uncertainty set
    
    Returns:
        Robust optimal policy
    """
    # Nominal parameters
    beta_nominal = scm.get_coefficients(target_variable)
    
    def worst_case_objective(x):
        """
        Compute worst-case expected outcome over uncertainty set.
        """
        # For ellipsoidal uncertainty, worst case is on boundary
        intervention_dict = dict(zip(intervention_variables, x))
        
        # Worst-case perturbation
        delta = radius * np.sign(x) / np.sqrt(len(x))
        
        # Perturbed coefficients
        beta_worst = beta_nominal.copy()
        for i, var in enumerate(intervention_variables):
            beta_worst[var] -= delta[i]
        
        # Compute expected outcome with worst-case parameters
        expected_y = sum(beta_worst.get(v, 0) * x[i] 
                        for i, v in enumerate(intervention_variables))
        
        return -expected_y  # Maximize worst case
    
    result = minimize(
        worst_case_objective,
        x0=np.zeros(len(intervention_variables)),
        method='SLSQP'
    )
    
    return {
        'interventions': dict(zip(intervention_variables, result.x)),
        'worst_case_value': -result.fun,
        'success': result.success
    }
```

---

## 9. Implementation Details

### 9.1 System Architecture

The Interactive Causal Policy Optimization Lab follows a modern client-server architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + TypeScript)            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ DataUpload  │  │ CausalGraph │  │ InterventionPanel       │ │
│  │             │  │ View        │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │Counterfactual│ │ Optimization│  │ ReportsView             │ │
│  │ Panel       │  │ Dashboard   │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                           │                                     │
│                    Zustand Store                                │
│                    (Session State)                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI + Python)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    API Routers                             │ │
│  │  /api/data  │ /api/causal │ /api/intervention │ /api/opt   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Services Layer                          │ │
│  │  CausalService │ DataService │ InterventionService │ ...   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Models Layer                            │ │
│  │  CausalModels │ DataModels │ InterventionModels │ ...      │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Storage                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Session Data │  │ Model Cache  │  │ Report Storage       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Backend Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── models/              # Pydantic data models
│   │   ├── __init__.py
│   │   ├── causal_models.py     # Causal graph models
│   │   ├── data_models.py       # Dataset models
│   │   ├── intervention_models.py
│   │   ├── optimization_models.py
│   │   └── report_models.py
│   ├── routers/             # API endpoints
│   │   ├── __init__.py
│   │   ├── causal.py            # Causal discovery endpoints
│   │   ├── data.py              # Data upload/validation
│   │   ├── intervention.py      # Intervention analysis
│   │   ├── optimization.py      # Policy optimization
│   │   └── reports.py           # Report generation
│   └── services/            # Business logic
│       ├── __init__.py
│       ├── causal_service.py
│       ├── data_service.py
│       ├── intervention_service.py
│       ├── optimization_service.py
│       └── report_service.py
├── data/                    # Session data storage
├── models/                  # Saved model files
├── reports/                 # Generated reports
└── requirements.txt         # Python dependencies
```

### 9.3 Frontend Structure

```
frontend/
├── src/
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # React entry point
│   ├── index.css            # Global styles (Tailwind)
│   ├── types.ts             # TypeScript type definitions
│   ├── vite-env.d.ts        # Vite environment types
│   ├── components/          # React components
│   │   ├── CausalGraphView.tsx    # Interactive graph visualization
│   │   ├── CounterfactualPanel.tsx
│   │   ├── DataUpload.tsx         # File upload component
│   │   ├── InterventionPanel.tsx
│   │   ├── MathExplanation.tsx    # Mathematical explanations
│   │   ├── OptimizationDashboard.tsx
│   │   └── ReportsView.tsx
│   └── store/               # State management
│       └── sessionStore.ts  # Zustand session store
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

### 9.4 Key Technologies

**Backend**:
- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **NumPy/SciPy**: Numerical computing
- **Pandas**: Data manipulation
- **NetworkX**: Graph algorithms
- **Scikit-learn**: Machine learning utilities

**Frontend**:
- **React 18**: UI library with hooks
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool
- **Tailwind CSS**: Utility-first CSS framework
- **React Flow**: Interactive graph visualization
- **Dagre**: Graph layout algorithms
- **TanStack Query**: Data fetching and caching
- **Zustand**: Lightweight state management
- **Lucide React**: Icon library

### 9.5 Session Management

Sessions are managed through unique session IDs:

```python
# Backend: Session middleware
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid.uuid4())
    request.state.session_id = session_id
    response = await call_next(request)
    response.headers["X-Session-ID"] = session_id
    return response
```

```typescript
// Frontend: Session store
interface SessionState {
  sessionId: string | null;
  datasetId: string | null;
  modelId: string | null;
  setSessionId: (id: string) => void;
  setDatasetId: (id: string) => void;
  setModelId: (id: string) => void;
}
```

### 9.6 Data Flow

1. **Data Upload**: User uploads CSV/JSON → Validated → Stored with session ID
2. **Causal Discovery**: Dataset ID → Structure learning → Model stored
3. **Intervention**: Model ID + intervention spec → Effect computation
4. **Optimization**: Model ID + constraints → Optimal policy
5. **Reports**: Session data → PDF/HTML report generation

---

## 10. API Reference

### 10.1 Data Management Endpoints

#### POST /api/data/upload
Upload and validate a dataset.

**Request**:
- Content-Type: multipart/form-data
- Body: File (CSV or JSON)

**Response**:
```json
{
  "dataset_id": "uuid",
  "filename": "data.csv",
  "n_rows": 1000,
  "n_columns": 10,
  "columns": [
    {"name": "age", "type": "numeric", "missing": 0},
    {"name": "income", "type": "numeric", "missing": 5}
  ],
  "validation_errors": []
}
```

#### GET /api/data/{dataset_id}
Retrieve dataset metadata.

#### DELETE /api/data/{dataset_id}
Delete a dataset.

### 10.2 Causal Analysis Endpoints

#### POST /api/causal/learn/{dataset_id}
Learn causal structure from data.

**Request Body**:
```json
{
  "method": "hybrid",
  "use_bootstrap": true,
  "n_bootstrap": 50
}
```

**Response**:
```json
{
  "model_id": "uuid",
  "graph": {
    "nodes": [
      {"id": "X1", "name": "X1", "type": "continuous"}
    ],
    "edges": [
      {"source": "X1", "target": "X2", "coefficient": 0.75, "confidence": 0.9}
    ],
    "n_nodes": 5,
    "n_edges": 4
  },
  "statistics": {
    "bic": -1234.56,
    "log_likelihood": -1000.0,
    "n_samples": 1000
  }
}
```

#### GET /api/causal/model/{model_id}
Retrieve a learned causal model.

#### GET /api/causal/model/{model_id}/graph
Get the causal graph structure.

#### GET /api/causal/model/{model_id}/metrics
Get model quality metrics.

### 10.3 Intervention Endpoints

#### POST /api/intervention/intervene
Compute interventional distribution.

**Request Body**:
```json
{
  "model_id": "uuid",
  "intervention_variables": {"X1": 1.5, "X2": 2.0},
  "target_variables": ["Y"]
}
```

**Response**:
```json
{
  "model_id": "uuid",
  "intervention_formula": "P(Y | do(X1=1.5, X2=2.0))",
  "distribution_stats": {
    "Y": {
      "mean": 3.2,
      "std": 0.5,
      "median": 3.1,
      "q1": 2.8,
      "q3": 3.5
    }
  },
  "causal_effects": {
    "Y": 0.75
  }
}
```

#### POST /api/intervention/counterfactual
Compute counterfactual query.

**Request Body**:
```json
{
  "model_id": "uuid",
  "evidence": {"X1": 1.0, "X2": 2.0, "Y": 3.0},
  "intervention": {"X1": 2.0},
  "target_variable": "Y"
}
```

**Response**:
```json
{
  "model_id": "uuid",
  "factual_value": 3.0,
  "counterfactual_value": 3.75,
  "explanation": "Under the counterfactual intervention, Y would have changed from 3.000 to 3.750."
}
```

#### POST /api/intervention/identify
Identify causal effect using do-calculus.

**Request Body**:
```json
{
  "model_id": "uuid",
  "treatment": "X1",
  "outcome": "Y"
}
```

**Response**:
```json
{
  "model_id": "uuid",
  "treatment": "X1",
  "outcome": "Y",
  "is_identifiable": true,
  "adjustment_set": ["X2", "X3"],
  "identification_formula": "P(Y | do(X1)) = Σ_Z P(Y | X1, Z) P(Z)"
}
```

### 10.4 Optimization Endpoints

#### POST /api/optimization/optimize
Find optimal intervention values.

**Request Body**:
```json
{
  "model_id": "uuid",
  "target_variable": "Y",
  "objective": "maximize",
  "intervention_variables": ["X1", "X2"],
  "constraints": {
    "budget": 1000,
    "costs": {"X1": 10, "X2": 20}
  }
}
```

**Response**:
```json
{
  "optimal_value": 5.5,
  "baseline_value": 3.0,
  "optimal_interventions": {
    "X1": 2.5,
    "X2": 1.8
  },
  "sensitivity": {
    "X1": 0.3,
    "X2": 0.2
  }
}
```

#### POST /api/optimization/evaluate
Evaluate a specific policy.

**Request Body**:
```json
{
  "model_id": "uuid",
  "policy_actions": {"X1": 2.0, "X2": 1.5}
}
```

**Response**:
```json
{
  "evaluation_id": "uuid",
  "model_id": "uuid",
  "expected_utility": 4.5,
  "outcome_distribution": {
    "Y": {"mean": 4.5, "std": 0.8}
  }
}
```

### 10.5 Report Endpoints

#### POST /api/reports/generate
Generate a comprehensive report.

**Request Body**:
```json
{
  "model_id": "uuid",
  "report_type": "full",
  "include_graphs": true,
  "include_interventions": true
}
```

**Response**:
```json
{
  "report_id": "uuid",
  "download_url": "/api/reports/download/uuid",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### GET /api/reports/download/{report_id}
Download a generated report.

---

## 11. Frontend Architecture

### 11.1 Component Hierarchy

```
App
├── Sidebar (Navigation)
│   └── NavLinks
└── Main Content
    ├── DataUpload
    │   ├── FileDropZone
    │   ├── ValidationResults
    │   └── DataPreview
    ├── CausalGraphView
    │   ├── MethodSelector
    │   ├── ReactFlow Graph
    │   │   ├── Nodes
    │   │   └── Edges
    │   └── ModelStatistics
    ├── InterventionPanel
    │   ├── InterventionForm
    │   ├── EffectVisualization
    │   └── ResultsTable
    ├── CounterfactualPanel
    │   ├── EvidenceForm
    │   ├── CounterfactualForm
    │   └── ComparisonView
    ├── OptimizationDashboard
    │   ├── ObjectiveSelector
    │   ├── ConstraintBuilder
    │   ├── OptimizationResults
    │   └── SensitivityChart
    ├── MathExplanation
    │   ├── ConceptCard
    │   ├── FormulaDisplay
    │   └── InteractiveExample
    └── ReportsView
        ├── ReportGenerator
        └── ReportPreview
```

### 11.2 State Management

Using Zustand for global state:

```typescript
// store/sessionStore.ts
import { create } from 'zustand';

interface SessionState {
  // Session
  sessionId: string | null;
  
  // Data
  datasetId: string | null;
  datasetInfo: DatasetInfo | null;
  
  // Model
  modelId: string | null;
  modelInfo: CausalModel | null;
  
  // Actions
  setSessionId: (id: string) => void;
  setDatasetId: (id: string) => void;
  setDatasetInfo: (info: DatasetInfo) => void;
  setModelId: (id: string) => void;
  setModelInfo: (model: CausalModel) => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: null,
  datasetId: null,
  datasetInfo: null,
  modelId: null,
  modelInfo: null,
  
  setSessionId: (id) => set({ sessionId: id }),
  setDatasetId: (id) => set({ datasetId: id }),
  setDatasetInfo: (info) => set({ datasetInfo: info }),
  setModelId: (id) => set({ modelId: id }),
  setModelInfo: (model) => set({ modelInfo: model }),
  reset: () => set({
    datasetId: null,
    datasetInfo: null,
    modelId: null,
    modelInfo: null
  }),
}));
```

### 11.3 Data Fetching

Using TanStack Query for API calls:

```typescript
// Example: Fetching causal model
const { data: model, isLoading, error } = useQuery({
  queryKey: ['model', modelId],
  queryFn: async () => {
    const response = await fetch(`/api/causal/model/${modelId}`, {
      headers: { 'X-Session-ID': sessionId || '' },
    });
    return response.json();
  },
  enabled: !!modelId,
});
```

### 11.4 Graph Visualization

The causal graph is rendered using React Flow with Dagre layout:

```typescript
// Layout algorithm
function getLayoutedElements(nodes: Node[], edges: Edge[], direction = 'TB') {
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 150, height: 50 });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.position = {
      x: nodeWithPosition.x - 75,
      y: nodeWithPosition.y - 25,
    };
  });

  return { nodes, edges };
}
```

### 11.5 Styling

Using Tailwind CSS with custom configuration:

```javascript
// tailwind.config.js
module.exports = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eef2ff',
          500: '#6366f1',
          900: '#312e81',
        },
      },
    },
  },
  plugins: [],
};
```

---

## 12. Usage Examples

### 12.1 Complete Workflow Example

```python
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
SESSION_ID = "test_session_001"

headers = {"X-Session-ID": SESSION_ID}

# Step 1: Upload dataset
with open("data.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/data/upload",
        headers=headers,
        files={"file": f}
    )
dataset_id = response.json()["dataset_id"]
print(f"Dataset uploaded: {dataset_id}")

# Step 2: Learn causal structure
response = requests.post(
    f"{BASE_URL}/api/causal/learn/{dataset_id}",
    headers=headers,
    json={"method": "hybrid"}
)
model_id = response.json()["model_id"]
print(f"Model learned: {model_id}")

# Step 3: Compute intervention effect
response = requests.post(
    f"{BASE_URL}/api/intervention/intervene",
    headers=headers,
    json={
        "model_id": model_id,
        "intervention_variables": {"treatment": 1.0},
        "target_variables": ["outcome"]
    }
)
print(f"Intervention effect: {response.json()}")

# Step 4: Compute counterfactual
response = requests.post(
    f"{BASE_URL}/api/intervention/counterfactual",
    headers=headers,
    json={
        "model_id": model_id,
        "evidence": {"treatment": 0, "outcome": 2.5},
        "intervention": {"treatment": 1},
        "target_variable": "outcome"
    }
)
print(f"Counterfactual: {response.json()}")

# Step 5: Optimize policy
response = requests.post(
    f"{BASE_URL}/api/optimization/optimize",
    headers=headers,
    json={
        "model_id": model_id,
        "target_variable": "outcome",
        "objective": "maximize",
        "intervention_variables": ["treatment"],
        "constraints": {"budget": 100}
    }
)
print(f"Optimal policy: {response.json()}")

# Step 6: Generate report
response = requests.post(
    f"{BASE_URL}/api/reports/generate",
    headers=headers,
    json={
        "model_id": model_id,
        "report_type": "full"
    }
)
print(f"Report: {response.json()}")
```

### 12.2 Python Client Library

```python
class CausalLabClient:
    """
    Python client for the Causal Policy Optimization Lab API.
    """
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.dataset_id = None
        self.model_id = None
    
    def create_session(self):
        """Create a new session."""
        import uuid
        self.session_id = str(uuid.uuid4())
        return self.session_id
    
    def upload_data(self, file_path):
        """Upload a dataset."""
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/api/data/upload",
                headers={"X-Session-ID": self.session_id},
                files={"file": f}
            )
        self.dataset_id = response.json()["dataset_id"]
        return response.json()
    
    def learn_causal_structure(self, method="hybrid"):
        """Learn causal structure from data."""
        response = requests.post(
            f"{self.base_url}/api/causal/learn/{self.dataset_id}",
            headers={"X-Session-ID": self.session_id},
            json={"method": method}
        )
        self.model_id = response.json()["model_id"]
        return response.json()
    
    def intervene(self, interventions, targets):
        """Compute intervention effects."""
        response = requests.post(
            f"{self.base_url}/api/intervention/intervene",
            headers={"X-Session-ID": self.session_id},
            json={
                "model_id": self.model_id,
                "intervention_variables": interventions,
                "target_variables": targets
            }
        )
        return response.json()
    
    def counterfactual(self, evidence, intervention, target):
        """Compute counterfactual."""
        response = requests.post(
            f"{self.base_url}/api/intervention/counterfactual",
            headers={"X-Session-ID": self.session_id},
            json={
                "model_id": self.model_id,
                "evidence": evidence,
                "intervention": intervention,
                "target_variable": target
            }
        )
        return response.json()
    
    def optimize(self, target, variables, objective="maximize", constraints=None):
        """Find optimal policy."""
        response = requests.post(
            f"{self.base_url}/api/optimization/optimize",
            headers={"X-Session-ID": self.session_id},
            json={
                "model_id": self.model_id,
                "target_variable": target,
                "objective": objective,
                "intervention_variables": variables,
                "constraints": constraints or {}
            }
        )
        return response.json()


# Usage
client = CausalLabClient()
client.create_session()
client.upload_data("my_data.csv")
client.learn_causal_structure()

# Compute intervention
result = client.intervene(
    interventions={"X1": 1.5},
    targets=["Y"]
)

# Compute counterfactual
cf = client.counterfactual(
    evidence={"X1": 1.0, "Y": 2.0},
    intervention={"X1": 2.0},
    target="Y"
)

# Optimize
optimal = client.optimize(
    target="Y",
    variables=["X1", "X2"],
    objective="maximize",
    constraints={"budget": 1000}
)
```

### 12.3 JavaScript/TypeScript Client

```typescript
// causalLabClient.ts
interface CausalLabConfig {
  baseUrl?: string;
}

class CausalLabClient {
  private baseUrl: string;
  private sessionId: string | null = null;
  private datasetId: string | null = null;
  private modelId: string | null = null;

  constructor(config: CausalLabConfig = {}) {
    this.baseUrl = config.baseUrl || 'http://localhost:8000';
  }

  async createSession(): Promise<string> {
    this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    return this.sessionId;
  }

  private async fetch(endpoint: string, options: RequestInit = {}): Promise<any> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': this.sessionId || '',
        ...options.headers,
      },
    });
    return response.json();
  }

  async uploadData(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/data/upload`, {
      method: 'POST',
      headers: { 'X-Session-ID': this.sessionId || '' },
      body: formData,
    });

    const data = await response.json();
    this.datasetId = data.dataset_id;
    return data;
  }

  async learnCausalStructure(method = 'hybrid'): Promise<any> {
    const data = await this.fetch(`/api/causal/learn/${this.datasetId}`, {
      method: 'POST',
      body: JSON.stringify({ method }),
    });
    this.modelId = data.model_id;
    return data;
  }

  async intervene(
    interventions: Record<string, number>,
    targets: string[]
  ): Promise<any> {
    return this.fetch('/api/intervention/intervene', {
      method: 'POST',
      body: JSON.stringify({
        model_id: this.modelId,
        intervention_variables: interventions,
        target_variables: targets,
      }),
    });
  }

  async counterfactual(
    evidence: Record<string, number>,
    intervention: Record<string, number>,
    target: string
  ): Promise<any> {
    return this.fetch('/api/intervention/counterfactual', {
      method: 'POST',
      body: JSON.stringify({
        model_id: this.modelId,
        evidence,
        intervention,
        target_variable: target,
      }),
    });
  }

  async optimize(
    target: string,
    variables: string[],
    objective: 'maximize' | 'minimize' = 'maximize',
    constraints: Record<string, any> = {}
  ): Promise<any> {
    return this.fetch('/api/optimization/optimize', {
      method: 'POST',
      body: JSON.stringify({
        model_id: this.modelId,
        target_variable: target,
        objective,
        intervention_variables: variables,
        constraints,
      }),
    });
  }
}

export { CausalLabClient };
```

---

## 13. Proofs and Derivations

### 13.1 Proof of Back-Door Adjustment

**Theorem**: If Z satisfies the back-door criterion relative to (X, Y), then:

$$P(Y | do(X = x)) = \sum_z P(Y | X = x, Z = z) P(Z = z)$$

**Proof**:

1. By the truncated factorization formula:
   $$P(Y | do(X = x)) = \sum_{v \setminus \{X, Y\}} \prod_{V_i \notin \{X, Y\}} P(V_i | Pa(V_i)) \cdot P(Y | Pa(Y)) \bigg|_{X=x}$$

2. Since Z blocks all back-door paths, Y is independent of X given Z in the graph where arrows into X are removed.

3. By Rule 2 of do-calculus:
   $$P(Y | do(X), Z) = P(Y | X, Z)$$

4. By Rule 3 of do-calculus, since Z is not a descendant of X:
   $$P(Z | do(X)) = P(Z)$$

5. Therefore:
   $$P(Y | do(X)) = \sum_z P(Y | do(X), Z = z) P(Z = z | do(X))$$
   $$= \sum_z P(Y | X, Z = z) P(Z = z)$$

Q.E.D.

### 13.2 Proof of Front-Door Adjustment

**Theorem**: If M satisfies the front-door criterion relative to (X, Y), then:

$$P(Y | do(X = x)) = \sum_m P(M = m | X = x) \sum_{x'} P(Y | X = x', M = m) P(X = x')$$

**Proof**:

1. First, note that P(M | do(X)) = P(M | X) because:
   - All back-door paths from X to M are blocked (condition 2)
   - By Rule 2: P(M | do(X)) = P(M | X)

2. Second, P(Y | do(M)) can be computed using back-door adjustment with X:
   $$P(Y | do(M = m)) = \sum_{x'} P(Y | M = m, X = x') P(X = x')$$
   
   This is because X blocks all back-door paths from M to Y (condition 3).

3. By the chain rule of interventions:
   $$P(Y | do(X = x)) = \sum_m P(Y | do(M = m)) P(M = m | do(X = x))$$

4. Substituting from steps 1 and 2:
   $$P(Y | do(X = x)) = \sum_m \left[ \sum_{x'} P(Y | M = m, X = x') P(X = x') \right] P(M = m | X = x)$$

Q.E.D.

### 13.3 Proof of do-Calculus Rules

**Rule 1**: P(Y | do(X), Z, W) = P(Y | do(X), W) if (Y ⊥ Z | X, W) in G̅ₓ

**Proof**:
- G̅ₓ is the graph with all arrows into X removed
- The condition states that Z is conditionally independent of Y given X, W in this modified graph
- Since the intervention do(X) removes all arrows into X, the distribution P(V | do(X)) factorizes according to G̅ₓ
- Therefore, the conditional independence in the graph implies probabilistic independence

**Rule 2**: P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if (Y ⊥ Z | X, W) in G̅ₓz̅

**Proof**:
- G̅ₓz̅ is the graph with arrows into X and into Z removed
- The intervention do(Z) removes the structural equation for Z
- If Z is not a descendant of X, then observing Z provides the same information as intervening on Z
- The graphical condition ensures this equivalence

**Rule 3**: P(Y | do(X), do(Z), W) = P(Y | do(X), W) if (Y ⊥ Z | X, W) in G̅ₓz

**Proof**:
- G̅ₓz is the graph with arrows into X and arrows out of Z removed
- This represents the causal structure when we intervene on X but remove Z's effects
- If Y is independent of Z in this graph, then Z has no causal effect on Y
- Therefore, the intervention do(Z) has no effect on Y

### 13.4 Derivation of Acyclicity Constraint

**Theorem**: A matrix W represents a DAG if and only if:

$$h(W) = \text{tr}(e^{W \circ W}) - d = 0$$

**Proof**:

1. For any matrix A, the matrix exponential is:
   $$e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

2. For a DAG with d nodes, there are no paths of length ≥ d (would imply a cycle).

3. Therefore, for adjacency matrix A of a DAG:
   $$A^k = 0 \text{ for all } k \geq d$$

4. The element-wise square A ◦ A has the same zero pattern as A.

5. For (A ◦ A)ᵏ, the (i,j) entry counts weighted paths of length k from i to j.

6. For a DAG:
   $$(A \circ A)^k = 0 \text{ for } k \geq d$$

7. Therefore:
   $$e^{A \circ A} = I + A \circ A + \frac{(A \circ A)^2}{2!} + ... + \frac{(A \circ A)^{d-1}}{(d-1)!}$$

8. Taking the trace:
   $$\text{tr}(e^{A \circ A}) = \text{tr}(I) + 0 + 0 + ... = d$$

9. Conversely, if tr(e^{A◦A}) = d, then all off-diagonal terms in the sum must be zero, implying no cycles.

Q.E.D.

### 13.5 Derivation of BIC Score

**Bayesian Information Criterion**:

$$\text{BIC} = -2 \ell(\hat{\theta}) + k \log(n)$$

**Derivation**:

1. Consider the Bayesian model evidence:
   $$P(D) = \int P(D|\theta) P(\theta) d\theta$$

2. Using Laplace approximation around the MLE θ̂:
   $$\log P(D) \approx \log P(D|\hat{\theta}) - \frac{k}{2} \log(n)$$

3. For large n, this gives:
   $$\text{BIC} \approx -2 \log P(D|M)$$

4. For linear Gaussian models:
   $$\ell(\hat{\theta}) = -\frac{n}{2} \log(2\pi\hat{\sigma}^2) - \frac{n}{2}$$

5. Therefore:
   $$\text{BIC} = n \log(\hat{\sigma}^2) + k \log(n) + \text{const}$$

### 13.6 Proof of Identifiability in LiNGAM

**Theorem**: Under LiNGAM assumptions, the causal ordering is uniquely identifiable.

**Proof Sketch**:

1. The model is:
   $$X = BX + \epsilon$$
   where B is lower triangular with zeros on diagonal.

2. This can be written as:
   $$(I - B)X = \epsilon$$
   $$X = (I - B)^{-1}\epsilon = A\epsilon$$

3. ICA recovers A up to permutation and scaling.

4. The key insight: due to non-Gaussianity, there is a unique permutation that makes A⁻¹ lower triangular.

5. This permutation gives the causal ordering.

6. The lower triangular structure ensures acyclicity.

---

## 14. References

### 14.1 Foundational Works

1. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
   - The definitive text on causal inference, introducing SCMs, do-calculus, and the ladder of causality.

2. **Spirtes, P., Glymour, C., & Scheines, R.** (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
   - Foundational work on constraint-based causal discovery algorithms.

3. **Imbens, G. W., & Rubin, D. B.** (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
   - Comprehensive treatment of the potential outcomes framework.

### 14.2 Causal Discovery

4. **Colombo, D., & Maathuis, M. H.** (2014). Order-independent constraint-based causal structure learning. *Journal of Machine Learning Research*, 15, 3741-3782.
   - Improvements to the PC algorithm for order independence.

5. **Chickering, D. M.** (2002). Optimal structure identification with greedy search. *Journal of Machine Learning Research*, 3, 507-554.
   - Theoretical foundations of GES algorithm.

6. **Shimizu, S., Hoyer, P. O., Hyvärinen, A., & Kerminen, A.** (2006). A linear non-Gaussian acyclic model for causal discovery. *Journal of Machine Learning Research*, 7, 2003-2030.
   - Introduction of LiNGAM algorithm.

7. **Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P.** (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.
   - NOTEARS algorithm for continuous optimization-based causal discovery.

### 14.3 Intervention and Counterfactuals

8. **Tian, J., & Pearl, J.** (2002). A general identification condition for causal effects. *Proceedings of AAAI-02*, 567-573.
   - General conditions for causal effect identification.

9. **Shpitser, I., & Pearl, J.** (2006). Identification of joint interventional distributions in recursive semi-Markovian causal models. *Proceedings of AAAI-06*, 1219-1226.
   - Complete algorithm for causal effect identification.

10. **Balke, A., & Pearl, J.** (1994). Counterfactual probabilities: Computational methods, bounds and applications. *Proceedings of UAI-94*, 46-54.
    - Foundations of counterfactual computation.

### 14.4 Policy Optimization

11. **Manski, C. F.** (2004). Statistical treatment rules for heterogeneous populations. *Econometrica*, 72(4), 1221-1246.
    - Optimal treatment assignment under uncertainty.

12. **Hirano, K., & Porter, J. R.** (2009). Asymptotics for statistical treatment rules. *Econometrica*, 77(5), 1683-1701.
    - Asymptotic properties of optimal policies.

### 14.5 Software and Tools

13. **Kalainathan, E., Goudet, O., Guyon, I., Lopez-Paz, D., & Sebag, M.** (2022). Causal discovery toolbox. *Journal of Machine Learning Research*, 23(227), 1-12.
    - Comprehensive Python library for causal discovery.

14. **Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E.** (2020). Learning sparse nonparametric DAGs. *Proceedings of AISTATS-20*.
    - Extensions of NOTEARS to nonparametric models.

### 14.6 Applications

15. **Pearl, J., & Mackenzie, D.** (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
    - Accessible introduction to causal thinking for general audiences.

16. **Hernán, M. A., & Robins, J. M.** (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
    - Applied causal inference with focus on epidemiology.

17. **Angrist, J. D., & Pischke, J.-S.** (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.
    - Practical guide to causal inference in economics.

### 14.7 Online Resources

18. **Pearl, J.** Causal Inference in Statistics: An Overview. *Statistics Surveys*, 3, 96-146.
    - Accessible overview of causal inference for statisticians.

19. **Peters, J., Janzing, D., & Schölkopf, B.** (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT Press.
    - Machine learning perspective on causal inference.

20. **Rosenbaum, P. R.** (2010). *Design of Observational Studies*. Springer.
    - Design-based approach to causal inference.

---

## License

This project is srictly under the guidance of PhD. Dr. RJ Mohan Sahu and developed by Satya Narayan Sahu of Class 12 License. And without any prior intemation or a written permission, no one is allowed to use for any developmental, upgradation, execution or commercialization of this project anywhere. If noticed, there will be a serious legal consequences.

---

## Acknowledgments

This project builds upon decades of research in causal inference, structural equation modeling, and machine learning. We gratefully acknowledge the foundational contributions of Judea Pearl, whose work established the theoretical framework for modern causal inference, as well as the countless researchers who have advanced the field.

Special thanks to the open-source community for developing the tools and libraries that make this project possible, including FastAPI, React, React Flow, and the scientific Python ecosystem.

---

*Last updated: February 2026*
*Version: 1.0*
+++++++ REPLACE</task_progress>
