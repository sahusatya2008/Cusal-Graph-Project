"""
Causal Service for Structure Learning and Parameter Estimation
Implements hybrid score-based and constraint-based causal discovery
"""
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import os
import logging
import networkx as nx
from scipy import stats
from scipy.linalg import expm
import warnings

from app.config import settings
from app.models.causal_models import (
    CausalGraph, Node, Edge, StructuralEquation,
    CausalModelResult, StructureLearningConfig, ParameterLearningConfig,
    StructureLearningMethod, ScoreFunction, IndependenceTest, FunctionalForm
)

logger = logging.getLogger(__name__)


class CausalService:
    """
    Service for causal structure learning and parameter estimation
    Implements hybrid methods combining score-based and constraint-based approaches
    """
    
    def __init__(self):
        self._storage_path = settings.MODELS_DIR
        
    async def learn_structure(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig,
        session_id: str
    ) -> Tuple[CausalGraph, Dict[str, Any]]:
        """
        Learn causal structure from data
        
        Args:
            df: Preprocessed dataframe
            config: Structure learning configuration
            session_id: Session identifier
            
        Returns:
            Tuple of (CausalGraph, learning_metadata)
        """
        logger.info(f"Starting structure learning with method: {config.method}")
        
        # Set random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Choose method
        if config.method == StructureLearningMethod.PC:
            graph, metadata = await self._pc_algorithm(df, config)
        elif config.method == StructureLearningMethod.GES:
            graph, metadata = await self._ges_algorithm(df, config)
        elif config.method == StructureLearningMethod.NOTEARS:
            graph, metadata = await self._notears_algorithm(df, config)
        elif config.method == StructureLearningMethod.BAYESIAN:
            graph, metadata = await self._bayesian_structure_learning(df, config)
        else:  # HYBRID
            graph, metadata = await self._hybrid_algorithm(df, config)
        
        # Compute edge confidences via bootstrap if enabled
        if config.use_bootstrap:
            edge_confidences = await self._bootstrap_confidence(
                df, config, graph
            )
            # Update edge confidences
            for edge in graph.edges:
                key = (edge.source, edge.target)
                if key in edge_confidences:
                    edge.confidence = edge_confidences[key]
        
        # Compute topological order
        graph.topological_order = self._topological_sort(graph)
        
        # Build adjacency matrix
        graph.adjacency_matrix = self._build_adjacency_matrix(graph, df.columns.tolist())
        
        graph.n_nodes = len(graph.nodes)
        graph.n_edges = len(graph.edges)
        
        logger.info(f"Structure learning complete: {graph.n_nodes} nodes, {graph.n_edges} edges")
        
        return graph, metadata
    
    async def _pc_algorithm(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig
    ) -> Tuple[CausalGraph, Dict[str, Any]]:
        """
        PC (Peter-Clark) algorithm - constraint-based approach
        Uses conditional independence tests to discover structure
        """
        variables = df.columns.tolist()
        n_vars = len(variables)
        
        # Initialize complete undirected graph
        adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        sep_sets = {}  # Separation sets
        
        # Phase 1: Edge removal based on independence tests
        depth = 0
        while depth < n_vars - 1:
            edges_to_remove = []
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj_matrix[i, j] == 0:
                        continue
                    
                    # Get adjacent nodes (excluding j)
                    adj_i = [k for k in range(n_vars) 
                            if adj_matrix[i, k] == 1 and k != j]
                    
                    # Test independence with conditioning sets of size 'depth'
                    if len(adj_i) >= depth:
                        from itertools import combinations
                        for cond_set in combinations(adj_i, depth):
                            cond_vars = [variables[k] for k in cond_set]
                            
                            # Perform independence test
                            is_independent, p_value = await self._independence_test(
                                df[variables[i]], df[variables[j]],
                                df[cond_vars] if cond_vars else None,
                                config.independence_test,
                                config.alpha
                            )
                            
                            if is_independent:
                                edges_to_remove.append((i, j))
                                sep_sets[(i, j)] = cond_set
                                sep_sets[(j, i)] = cond_set
                                break
            
            # Remove edges
            for i, j in edges_to_remove:
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0
            
            depth += 1
            
            # Stop if no edges were removed
            if not edges_to_remove:
                break
        
        # Phase 2: Orient edges (v-structures)
        # Find v-structures: i -> k <- j where i and j are not adjacent
        directed = np.zeros((n_vars, n_vars))
        
        for k in range(n_vars):
            adj_k = [i for i in range(n_vars) if adj_matrix[i, k] == 1]
            
            for i_idx, i in enumerate(adj_k):
                for j in adj_k[i_idx + 1:]:
                    # Check if i and j are not adjacent
                    if adj_matrix[i, j] == 0:
                        # Check if k is not in separation set of i and j
                        sep = sep_sets.get((i, j), set())
                        if k not in (sep if sep else set()):
                            # Orient as v-structure
                            directed[i, k] = 1
                            directed[j, k] = 1
        
        # Phase 3: Apply orientation rules (Meek rules)
        # Rule 1: If i -> k -> j and i - j, orient i -> j
        # (Simplified implementation)
        changed = True
        while changed:
            changed = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if adj_matrix[i, j] == 1 and directed[i, j] == 0 and directed[j, i] == 0:
                        for k in range(n_vars):
                            if directed[i, k] == 1 and directed[k, j] == 1:
                                directed[i, j] = 1
                                changed = True
        
        # Build graph
        nodes = [Node(id=var, name=var) for var in variables]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if directed[i, j] == 1:
                    edges.append(Edge(
                        source=variables[i],
                        target=variables[j],
                        is_directed=True
                    ))
                elif adj_matrix[i, j] == 1 and i < j and directed[i, j] == 0 and directed[j, i] == 0:
                    # Undirected edge (could not be oriented)
                    pass
        
        graph = CausalGraph(nodes=nodes, edges=edges, is_dag=True)
        
        metadata = {
            "method": "PC",
            "depth_reached": depth,
            "n_independence_tests": len(sep_sets) * 2,
            "alpha": config.alpha
        }
        
        return graph, metadata
    
    async def _ges_algorithm(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig
    ) -> Tuple[CausalGraph, Dict[str, Any]]:
        """
        Greedy Equivalence Search - score-based approach
        Searches over equivalence classes of DAGs
        """
        variables = df.columns.tolist()
        n_vars = len(variables)
        
        # Initialize empty graph
        adj_matrix = np.zeros((n_vars, n_vars))
        
        # Forward phase: Add edges to improve score
        current_score = await self._compute_score(df, adj_matrix, config.score_function)
        
        improved = True
        forward_steps = 0
        while improved:
            improved = False
            best_delta = 0
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adj_matrix[i, j] == 0:
                        # Try adding edge i -> j
                        adj_matrix[i, j] = 1
                        
                        # Check acyclicity
                        if self._is_dag(adj_matrix):
                            new_score = await self._compute_score(df, adj_matrix, config.score_function)
                            delta = new_score - current_score
                            
                            if delta > best_delta:
                                best_delta = delta
                                best_edge = (i, j, 'add')
                        
                        adj_matrix[i, j] = 0
            
            if best_edge is not None:
                i, j, _ = best_edge
                adj_matrix[i, j] = 1
                current_score += best_delta
                improved = True
                forward_steps += 1
        
        # Backward phase: Remove edges to improve score
        improved = True
        backward_steps = 0
        while improved:
            improved = False
            best_delta = 0
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if adj_matrix[i, j] == 1:
                        # Try removing edge i -> j
                        adj_matrix[i, j] = 0
                        new_score = await self._compute_score(df, adj_matrix, config.score_function)
                        delta = new_score - current_score
                        
                        if delta > best_delta:
                            best_delta = delta
                            best_edge = (i, j, 'remove')
                        
                        adj_matrix[i, j] = 1
            
            if best_edge is not None:
                i, j, _ = best_edge
                adj_matrix[i, j] = 0
                current_score += best_delta
                improved = True
                backward_steps += 1
        
        # Build graph
        nodes = [Node(id=var, name=var) for var in variables]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adj_matrix[i, j] == 1:
                    edges.append(Edge(
                        source=variables[i],
                        target=variables[j],
                        is_directed=True
                    ))
        
        graph = CausalGraph(nodes=nodes, edges=edges, is_dag=True)
        
        metadata = {
            "method": "GES",
            "forward_steps": forward_steps,
            "backward_steps": backward_steps,
            "final_score": current_score,
            "score_function": config.score_function
        }
        
        return graph, metadata
    
    async def _notears_algorithm(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig
    ) -> Tuple[CausalGraph, Dict[str, Any]]:
        """
        NOTEARS: Non-combinatorial optimization via trace EXponential
        Differentiable approach to structure learning
        """
        variables = df.columns.tolist()
        n_vars = len(variables)
        n_samples = len(df)
        
        # Standardize data
        X = df.values.astype(float)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Initialize weight matrix
        W = np.zeros((n_vars, n_vars))
        
        # Optimization parameters
        lambda_l1 = config.lambda_l1
        lambda_dag = config.lambda_dag
        lr = 0.01
        max_iter = config.n_samples if config.n_samples > 0 else 1000
        
        # Gradient descent with acyclicity constraint
        for iteration in range(max_iter):
            # Compute gradient
            grad = np.zeros((n_vars, n_vars))
            
            # Likelihood gradient (least squares)
            for i in range(n_vars):
                parents = np.where(W[:, i] != 0)[0]
                if len(parents) > 0:
                    pred = X[:, parents] @ W[parents, i]
                    residual = X[:, i] - pred
                    for p in parents:
                        grad[p, i] = -2 * np.mean(residual * X[:, p])
            
            # L1 gradient (subgradient)
            grad += lambda_l1 * np.sign(W)
            
            # Acyclicity gradient
            # h(W) = tr(e^{W ◦ W}) - n
            E = expm(W * W)
            h = np.trace(E) - n_vars
            dag_grad = E.T * (2 * W)
            grad += lambda_dag * dag_grad
            
            # Update weights
            W = W - lr * grad
            
            # Project to enforce acyclicity
            if h > 0.1:
                lambda_dag *= 1.1
            
            # Check convergence
            if iteration % 100 == 0:
                h = np.trace(expm(W * W)) - n_vars
                if h < 1e-4:
                    break
        
        # Threshold small weights
        W[np.abs(W) < 0.1] = 0
        
        # Build graph
        nodes = [Node(id=var, name=var) for var in variables]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if np.abs(W[i, j]) > 0.1:
                    edges.append(Edge(
                        source=variables[i],
                        target=variables[j],
                        coefficient=float(W[i, j]),
                        is_directed=True
                    ))
        
        graph = CausalGraph(nodes=nodes, edges=edges, is_dag=True)
        
        metadata = {
            "method": "NOTEARS",
            "iterations": iteration + 1,
            "final_h": float(h),
            "lambda_l1": lambda_l1,
            "lambda_dag": lambda_dag
        }
        
        return graph, metadata
    
    async def _bayesian_structure_learning(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig
    ) -> Tuple[CausalGraph, Dict[str, Any]]:
        """
        Bayesian structure learning with MCMC sampling
        Samples from posterior P(G|D)
        """
        variables = df.columns.tolist()
        n_vars = len(variables)
        
        # Initialize with empty graph
        current_adj = np.zeros((n_vars, n_vars))
        current_score = await self._compute_bayesian_score(df, current_adj, config)
        
        # MCMC sampling
        n_samples = config.n_samples
        samples = []
        scores = [current_score]
        
        for sample_idx in range(n_samples):
            # Propose a change (add, delete, or reverse edge)
            proposal_type = np.random.choice(['add', 'delete', 'reverse'])
            
            i, j = np.random.randint(0, n_vars, size=2)
            if i == j:
                continue
            
            proposed_adj = current_adj.copy()
            
            if proposal_type == 'add' and current_adj[i, j] == 0:
                proposed_adj[i, j] = 1
            elif proposal_type == 'delete' and current_adj[i, j] == 1:
                proposed_adj[i, j] = 0
            elif proposal_type == 'reverse' and current_adj[i, j] == 1:
                proposed_adj[i, j] = 0
                proposed_adj[j, i] = 1
            
            # Check acyclicity
            if not self._is_dag(proposed_adj):
                continue
            
            # Compute acceptance probability
            proposed_score = await self._compute_bayesian_score(df, proposed_adj, config)
            log_accept_ratio = proposed_score - current_score
            
            if np.log(np.random.random()) < log_accept_ratio:
                current_adj = proposed_adj
                current_score = proposed_score
            
            samples.append(current_adj.copy())
            scores.append(current_score)
        
        # Compute edge posterior probabilities
        edge_probs = np.zeros((n_vars, n_vars))
        for adj in samples[int(n_samples * 0.5):]:  # Use second half of samples
            edge_probs += adj
        edge_probs /= len(samples[int(n_samples * 0.5):])
        
        # Build graph with edges above threshold
        threshold = 0.5
        nodes = [Node(id=var, name=var) for var in variables]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if edge_probs[i, j] > threshold:
                    edges.append(Edge(
                        source=variables[i],
                        target=variables[j],
                        confidence=float(edge_probs[i, j]),
                        is_directed=True
                    ))
        
        graph = CausalGraph(nodes=nodes, edges=edges, is_dag=True)
        
        metadata = {
            "method": "Bayesian",
            "n_samples": n_samples,
            "final_score": current_score,
            "acceptance_rate": len(set(map(lambda x: x.tobytes(), samples))) / n_samples
        }
        
        return graph, metadata
    
    async def _hybrid_algorithm(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig
    ) -> Tuple[CausalGraph, Dict[str, Any]]:
        """
        Hybrid approach: Combine constraint-based and score-based methods
        1. Use PC to get skeleton
        2. Use GES to orient edges and refine
        """
        # Step 1: Get skeleton using constraint-based approach
        skeleton, sep_sets = await self._learn_skeleton(df, config)
        
        # Step 2: Orient edges using score-based approach
        variables = df.columns.tolist()
        n_vars = len(variables)
        
        # Initialize with skeleton
        adj_matrix = skeleton.copy()
        
        # Orient v-structures
        directed = np.zeros((n_vars, n_vars))
        
        for k in range(n_vars):
            adj_k = [i for i in range(n_vars) if skeleton[i, k] == 1]
            
            for i_idx, i in enumerate(adj_k):
                for j in adj_k[i_idx + 1:]:
                    if skeleton[i, j] == 0:
                        sep = sep_sets.get((i, j), set())
                        if k not in (sep if sep else set()):
                            directed[i, k] = 1
                            directed[j, k] = 1
        
        # Refine with score-based search
        current_score = await self._compute_score(df, directed, config.score_function)
        
        # Local search for improvements
        for _ in range(100):
            improved = False
            for i in range(n_vars):
                for j in range(n_vars):
                    if skeleton[i, j] == 1 and directed[i, j] == 0 and directed[j, i] == 0:
                        # Try both orientations
                        for orient in [(i, j), (j, i)]:
                            test_directed = directed.copy()
                            test_directed[orient[0], orient[1]] = 1
                            
                            if self._is_dag(test_directed):
                                test_score = await self._compute_score(
                                    df, test_directed, config.score_function
                                )
                                if test_score > current_score:
                                    directed = test_directed
                                    current_score = test_score
                                    improved = True
                                    break
                if improved:
                    break
            if not improved:
                break
        
        # Build graph
        nodes = [Node(id=var, name=var) for var in variables]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if directed[i, j] == 1:
                    edges.append(Edge(
                        source=variables[i],
                        target=variables[j],
                        is_directed=True
                    ))
        
        graph = CausalGraph(nodes=nodes, edges=edges, is_dag=True)
        
        metadata = {
            "method": "Hybrid",
            "skeleton_edges": int(skeleton.sum() / 2),
            "final_score": current_score
        }
        
        return graph, metadata
    
    async def _learn_skeleton(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig
    ) -> Tuple[np.ndarray, Dict]:
        """Learn undirected skeleton using independence tests"""
        variables = df.columns.tolist()
        n_vars = len(variables)
        
        skeleton = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        sep_sets = {}
        
        depth = 0
        while depth < n_vars - 1:
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if skeleton[i, j] == 0:
                        continue
                    
                    adj_i = [k for k in range(n_vars) 
                            if skeleton[i, k] == 1 and k != j]
                    
                    from itertools import combinations
                    for cond_set in combinations(adj_i, depth):
                        cond_vars = [variables[k] for k in cond_set]
                        
                        is_independent, _ = await self._independence_test(
                            df[variables[i]], df[variables[j]],
                            df[cond_vars] if cond_vars else None,
                            config.independence_test,
                            config.alpha
                        )
                        
                        if is_independent:
                            skeleton[i, j] = 0
                            skeleton[j, i] = 0
                            sep_sets[(i, j)] = cond_set
                            break
            
            depth += 1
            if depth > config.max_parents:
                break
        
        return skeleton, sep_sets
    
    async def _independence_test(
        self,
        x: pd.Series,
        y: pd.Series,
        z: Optional[pd.DataFrame],
        test_type: IndependenceTest,
        alpha: float
    ) -> Tuple[bool, float]:
        """
        Perform conditional independence test
        Returns (is_independent, p_value)
        """
        # Remove missing values
        if z is not None and len(z.columns) > 0:
            valid_idx = x.notna() & y.notna() & z.notna().all(axis=1)
            x = x[valid_idx]
            y = y[valid_idx]
            z = z[valid_idx]
        else:
            valid_idx = x.notna() & y.notna()
            x = x[valid_idx]
            y = y[valid_idx]
        
        if len(x) < 10:
            return True, 1.0
        
        if test_type == IndependenceTest.PEARSON:
            if z is not None and len(z.columns) > 0:
                # Partial correlation
                from sklearn.linear_model import LinearRegression
                
                # Regress x on z
                lr_x = LinearRegression()
                lr_x.fit(z, x)
                res_x = x - lr_x.predict(z)
                
                # Regress y on z
                lr_y = LinearRegression()
                lr_y.fit(z, y)
                res_y = y - lr_y.predict(z)
                
                corr, p_value = stats.pearsonr(res_x, res_y)
            else:
                corr, p_value = stats.pearsonr(x, y)
            
            return p_value > alpha, p_value
        
        elif test_type == IndependenceTest.FISHER_Z:
            if z is not None and len(z.columns) > 0:
                from sklearn.linear_model import LinearRegression
                lr_x = LinearRegression().fit(z, x)
                lr_y = LinearRegression().fit(z, y)
                res_x = x - lr_x.predict(z)
                res_y = y - lr_y.predict(z)
                corr, _ = stats.pearsonr(res_x, res_y)
            else:
                corr, _ = stats.pearsonr(x, y)
            
            # Fisher z-transformation
            n = len(x)
            z_score = 0.5 * np.log((1 + corr) / (1 - corr))
            se = 1 / np.sqrt(n - 3)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score) / se))
            
            return p_value > alpha, p_value
        
        else:  # Default to Pearson
            corr, p_value = stats.pearsonr(x, y)
            return p_value > alpha, p_value
    
    async def _compute_score(
        self,
        df: pd.DataFrame,
        adj_matrix: np.ndarray,
        score_func: ScoreFunction
    ) -> float:
        """Compute score for a graph structure"""
        n_samples = len(df)
        n_vars = len(df.columns)
        
        if score_func == ScoreFunction.BIC:
            # BIC = -2 * log_likelihood + k * log(n)
            log_lik = 0
            k_params = 0
            
            for j in range(n_vars):
                parents = np.where(adj_matrix[:, j] == 1)[0]
                y = df.iloc[:, j].values
                
                if len(parents) == 0:
                    # No parents - just variance
                    log_lik += -0.5 * n_samples * np.log(y.var() + 1e-8)
                    k_params += 1
                else:
                    # Linear regression
                    X = df.iloc[:, parents].values
                    try:
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression()
                        lr.fit(X, y)
                        residuals = y - lr.predict(X)
                        log_lik += -0.5 * n_samples * np.log(residuals.var() + 1e-8)
                        k_params += len(parents) + 1
                    except:
                        log_lik += -0.5 * n_samples * np.log(y.var() + 1e-8)
                        k_params += 1
            
            bic = -2 * log_lik + k_params * np.log(n_samples)
            return -bic  # Higher is better
        
        elif score_func == ScoreFunction.AIC:
            log_lik = 0
            k_params = 0
            
            for j in range(n_vars):
                parents = np.where(adj_matrix[:, j] == 1)[0]
                y = df.iloc[:, j].values
                
                if len(parents) == 0:
                    log_lik += -0.5 * len(y) * np.log(y.var() + 1e-8)
                    k_params += 1
                else:
                    X = df.iloc[:, parents].values
                    try:
                        from sklearn.linear_model import LinearRegression
                        lr = LinearRegression().fit(X, y)
                        residuals = y - lr.predict(X)
                        log_lik += -0.5 * len(y) * np.log(residuals.var() + 1e-8)
                        k_params += len(parents) + 1
                    except:
                        log_lik += -0.5 * len(y) * np.log(y.var() + 1e-8)
                        k_params += 1
            
            aic = -2 * log_lik + 2 * k_params
            return -aic
        
        return 0.0
    
    async def _compute_bayesian_score(
        self,
        df: pd.DataFrame,
        adj_matrix: np.ndarray,
        config: StructureLearningConfig
    ) -> float:
        """Compute Bayesian score (BDeu)"""
        n_samples = len(df)
        n_vars = len(df.columns)
        
        # BDeu score
        score = 0
        prior_edge = config.prior_edge_prob
        
        for j in range(n_vars):
            parents = np.where(adj_matrix[:, j] == 1)[0]
            
            # Prior on number of parents
            n_parents = len(parents)
            score += np.log(prior_edge ** n_parents * (1 - prior_edge) ** (n_vars - 1 - n_parents))
            
            # Marginal likelihood (simplified)
            y = df.iloc[:, j].values
            if len(parents) == 0:
                score += -0.5 * n_samples * np.log(y.var() + 1e-8)
            else:
                X = df.iloc[:, parents].values
                try:
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression().fit(X, y)
                    residuals = y - lr.predict(X)
                    score += -0.5 * n_samples * np.log(residuals.var() + 1e-8)
                except:
                    score += -0.5 * n_samples * np.log(y.var() + 1e-8)
        
        return score
    
    def _is_dag(self, adj_matrix: np.ndarray) -> bool:
        """Check if adjacency matrix represents a DAG"""
        n = adj_matrix.shape[0]
        
        # Check for cycles using matrix exponential
        # A graph is a DAG iff trace(e^{A ◦ A}) = n
        E = expm(adj_matrix * adj_matrix)
        return np.trace(E) - n < 1e-6
    
    async def _bootstrap_confidence(
        self,
        df: pd.DataFrame,
        config: StructureLearningConfig,
        base_graph: CausalGraph
    ) -> Dict[Tuple[str, str], float]:
        """Compute edge confidence via bootstrap"""
        n_bootstrap = config.n_bootstrap
        edge_counts = {}
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            boot_df = df.sample(n=len(df), replace=True)
            
            # Learn structure on bootstrap sample
            try:
                boot_graph, _ = await self._ges_algorithm(boot_df, config)
                
                for edge in boot_graph.edges:
                    key = (edge.source, edge.target)
                    edge_counts[key] = edge_counts.get(key, 0) + 1
            except:
                continue
        
        # Normalize counts to probabilities
        edge_confidences = {
            key: count / n_bootstrap 
            for key, count in edge_counts.items()
        }
        
        return edge_confidences
    
    def _topological_sort(self, graph: CausalGraph) -> List[str]:
        """Compute topological ordering of nodes"""
        # Build adjacency dict
        adj = {node.name: [] for node in graph.nodes}
        in_degree = {node.name: 0 for node in graph.nodes}
        
        for edge in graph.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Kahn's algorithm
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
    
    def _build_adjacency_matrix(
        self, 
        graph: CausalGraph, 
        variables: List[str]
    ) -> List[List[int]]:
        """Build adjacency matrix from graph"""
        var_idx = {var: i for i, var in enumerate(variables)}
        n = len(variables)
        adj = [[0] * n for _ in range(n)]
        
        for edge in graph.edges:
            if edge.source in var_idx and edge.target in var_idx:
                adj[var_idx[edge.source]][var_idx[edge.target]] = 1
        
        return adj
    
    async def learn_parameters(
        self,
        df: pd.DataFrame,
        graph: CausalGraph,
        config: ParameterLearningConfig
    ) -> List[StructuralEquation]:
        """
        Learn structural equation parameters
        
        For each variable, estimate:
        - Coefficients for parent variables
        - Noise distribution parameters
        """
        equations = []
        
        for node in graph.nodes:
            var_name = node.name
            parents = graph.get_parents(var_name)
            
            equation = StructuralEquation(
                variable=var_name,
                parents=parents,
                functional_form=config.functional_form
            )
            
            if len(parents) == 0:
                # No parents - just estimate marginal distribution
                y = df[var_name].values
                equation.intercept = float(y.mean())
                equation.noise_mean = 0.0
                equation.noise_std = float(y.std())
                equation.r_squared = 0.0
                equation.rmse = float(y.std())
                
            else:
                # Estimate structural equation
                y = df[var_name].values
                X = df[parents].values
                
                if config.functional_form in [FunctionalForm.LINEAR, FunctionalForm.LINEAR_GAUSSIAN]:
                    # Linear regression
                    from sklearn.linear_model import LinearRegression, Ridge, Lasso
                    
                    if config.l2_penalty > 0:
                        lr = Ridge(alpha=config.l2_penalty)
                    elif config.l1_penalty > 0:
                        lr = Lasso(alpha=config.l1_penalty)
                    else:
                        lr = LinearRegression()
                    
                    lr.fit(X, y)
                    
                    equation.intercept = float(lr.intercept_)
                    equation.coefficients = {
                        parent: float(coef) 
                        for parent, coef in zip(parents, lr.coef_)
                    }
                    
                    residuals = y - lr.predict(X)
                    equation.noise_mean = float(residuals.mean())
                    equation.noise_std = float(residuals.std())
                    
                    # R-squared
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    equation.r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                    equation.rmse = float(np.sqrt(np.mean(residuals ** 2)))
                    
                elif config.functional_form == FunctionalForm.NONLINEAR_ADDITIVE:
                    # Use polynomial features or splines (simplified)
                    from sklearn.preprocessing import PolynomialFeatures
                    from sklearn.linear_model import Ridge
                    
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_poly = poly.fit_transform(X)
                    
                    lr = Ridge(alpha=0.1)
                    lr.fit(X_poly, y)
                    
                    equation.intercept = float(lr.intercept_)
                    equation.nonlinear_params = {
                        "type": "polynomial",
                        "degree": 2,
                        "coefficients": lr.coef_.tolist()
                    }
                    
                    residuals = y - lr.predict(X_poly)
                    equation.noise_std = float(residuals.std())
                    equation.r_squared = float(lr.score(X_poly, y))
            
            equations.append(equation)
        
        return equations
    
    async def compute_model_metrics(
        self,
        df: pd.DataFrame,
        graph: CausalGraph,
        equations: List[StructuralEquation]
    ) -> Dict[str, float]:
        """Compute model quality metrics"""
        # Log-likelihood
        log_lik = 0
        for eq in equations:
            y = df[eq.variable].values
            if eq.noise_std > 0:
                log_lik += -0.5 * len(y) * np.log(2 * np.pi * eq.noise_std ** 2)
                log_lik += -0.5 * np.sum((y - y.mean()) ** 2) / (eq.noise_std ** 2)
        
        # BIC and AIC
        n_samples = len(df)
        n_params = sum(
            len(eq.coefficients) + 2 if eq.coefficients else 2
            for eq in equations
        )
        
        bic = -2 * log_lik + n_params * np.log(n_samples)
        aic = -2 * log_lik + 2 * n_params
        
        return {
            "log_likelihood": log_lik,
            "bic": bic,
            "aic": aic,
            "n_parameters": n_params
        }
    
    async def save_model(
        self,
        session_id: str,
        model_id: str,
        result: CausalModelResult
    ):
        """Save model to storage"""
        import joblib
        
        session_dir = os.path.join(self._storage_path, session_id, "models")
        os.makedirs(session_dir, exist_ok=True)
        
        file_path = os.path.join(session_dir, f"{model_id}.joblib")
        joblib.dump(result.model_dump(), file_path)
    
    async def load_model(
        self,
        session_id: str,
        model_id: str
    ) -> Optional[CausalModelResult]:
        """Load model from storage"""
        import joblib
        
        file_path = os.path.join(self._storage_path, session_id, "models", f"{model_id}.joblib")
        
        if not os.path.exists(file_path):
            return None
        
        data = joblib.load(file_path)
        return CausalModelResult(**data)