import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg, stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
from hdbscan import HDBSCAN
from tqdm import tqdm
import os
from typing import Tuple, List, Dict, Callable
#from sklearn.decomposition import PCA
from scipy import linalg
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.feature_selection import f_regression
import datetime
import glob
from matplotlib.colors import LinearSegmentedColormap
import time

# Constants
BASE_NUCLEOTIDES = ['A', 'T', 'C', 'G']
DEFAULT_SNV_RATE = 0.01
DEFAULT_MUTATION_RATE = 0.01
KEY_PATTERNS = ['ATCG', 'GCAT', 'TGCA', 'CGTA', 'TACG', 'GATC', 'CTAG', 'AGCT']


class CustomPCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = linalg.svd(X_centered, full_matrices=False)
        
        explained_variance = (S ** 2) / (n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var

        if self.n_components is None:
            n_components = min(n_samples, n_features) - 1
        elif 0 < self.n_components < 1:
            n_components = next(i for i, ev in enumerate(explained_variance_ratio.cumsum())
                                if ev >= self.n_components) + 1
        else:
            n_components = min(self.n_components, n_features)

        self.n_components_ = n_components
        self.components_ = Vt[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = S[:n_components]

        # Normalize explained_variance_ratio_ to sum to 1
        self.explained_variance_ratio_ /= self.explained_variance_ratio_.sum()

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        X_transformed = np.dot(X_centered, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        return X_transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.dot(X, self.components_) + self.mean_

def set_random_seed(seed: int = 6527612) -> None:
    """Set the random seed"""
    np.random.seed(seed)

def create_sequence(length: int) -> str:
    """Create a random DNA sequence of given length."""
    return ''.join(np.random.choice(BASE_NUCLEOTIDES, size=length))

def mutate_sequence(sequence: str, mutation_rate: float = DEFAULT_MUTATION_RATE, window_size: int = 5, ld_strength: float = 0.7) -> str:
    """Mutate a given DNA sequence with a specified mutation rate and linkage disequilibrium."""
    mutated_seq = list(sequence)
    for i in range(len(sequence)):
        if np.random.random() < mutation_rate:
            # Mutate the current base
            mutated_seq[i] = np.random.choice([base for base in BASE_NUCLEOTIDES if base != sequence[i]])
            
            # Apply LD: Correlated mutations within the window
            for j in range(max(0, i - window_size), min(len(sequence), i + window_size + 1)):
                if i != j:
                    ld_probability = ld_strength * (1 - abs(i - j) / window_size)
                    if np.random.random() < ld_probability:
                        mutated_seq[j] = np.random.choice([base for base in BASE_NUCLEOTIDES if base != sequence[j]])
    
    return ''.join(mutated_seq)





def create_pangenome_graph(n_base_seqs: int, seq_length: int, n_variants: int, n_snps: int) -> Tuple[nx.DiGraph, Dict[str, str], np.ndarray, np.ndarray]:
    """Create a pangenome graph with linkage disequilibrium."""
    G = nx.DiGraph()
    sequences = {}
    snp_positions = np.random.choice(seq_length, size=n_snps, replace=False)
    snp_effects = np.random.normal(0, 1, n_snps)
    
    node_id_counter = 0  # Initialize node ID counter

    # Create base sequences
    for i in range(n_base_seqs):
        seq = create_sequence(seq_length)
        node_name = f"base_{i}"
        G.add_node(node_name, sequence=seq, position=i, id=node_id_counter)  # Add unique ID to node
        sequences[node_name] = seq
        node_id_counter += 1  # Increment ID counter
        if i > 0:
            G.add_edge(f"base_{i-1}", node_name)

    variant_counter = 0
    shared_alt_prob = 0.2
    max_branches_per_base = 3

    def mutate_with_linkage(sequence: str, mutation_rate: float = DEFAULT_MUTATION_RATE, linkage_distance: int = 5) -> str:
        """Mutate a sequence with linkage disequilibrium."""
        mutated_seq = list(sequence)
        for i in range(len(sequence)):
            if np.random.random() < mutation_rate:
                mutated_seq[i] = np.random.choice([base for base in BASE_NUCLEOTIDES if base != sequence[i]])
                # Introduce correlated mutations in nearby positions
                for j in range(max(0, i - linkage_distance), min(len(sequence), i + linkage_distance + 1)):
                    if i != j and np.random.random() < mutation_rate * 0.1:  # Chance of correlated mutation
                        mutated_seq[j] = np.random.choice([base for base in BASE_NUCLEOTIDES if base != sequence[j]])
        return ''.join(mutated_seq)

    for base_index in range(n_base_seqs - 1):
        branches = 0
        while branches < max_branches_per_base and np.random.random() < 0.7:
            branch_type = np.random.choice(['shared_alt', 'variant', 'alt'], p=[0.2, 0.5, 0.3])
            branch_length = np.random.randint(3, 6) if branch_type == 'shared_alt' else np.random.randint(1, 4)
            
            prev_node = f"base_{base_index}"
            start_position = base_index
            for j in range(branch_length):
                var_seq = mutate_with_linkage(sequences[prev_node])
                node_name = f"{branch_type}_{variant_counter}"
                variant_counter += 1
                position = start_position + j
                G.add_node(node_name, sequence=var_seq, position=position, id=node_id_counter) # Add unique ID to variant nodes 
                sequences[node_name] = var_seq
                node_id_counter += 1  # Increment ID counter
                G.add_edge(prev_node, node_name)
                prev_node = node_name

            # Connect back to the main path
            end_position = min(start_position + branch_length, n_base_seqs - 1)
            reconnect_range = range(max(end_position - 1, start_position + 1), min(end_position + 2, n_base_seqs))
            reconnect_point = np.random.choice(list(reconnect_range))
            G.add_edge(prev_node, f"base_{reconnect_point}")
            branches += 1

    return G, sequences, snp_positions, snp_effects

def generate_individual_path(G: nx.DiGraph, target_length: int) -> List[str]:
    path = ['base_0']
    current_node = 'base_0'
    current_length = len(G.nodes[current_node]['sequence'])
    in_alternative = False
    alternative_decay = 0.9

    while current_length < target_length:
        neighbors = list(G.successors(current_node))
        if not neighbors:
            break

        base_nodes = [n for n in neighbors if n.startswith('base_')]
        alt_nodes = [n for n in neighbors if not n.startswith('base_')]

        if in_alternative:
            if alt_nodes and np.random.random() < alternative_decay:
                next_node = np.random.choice(alt_nodes)
                alternative_decay *= 0.9
            elif base_nodes:
                next_node = min(base_nodes, key=lambda x: abs(int(x.split('_')[1]) - G.nodes[current_node]['position']))
                in_alternative = False
            else:
                next_node = np.random.choice(neighbors)
        else:
            if base_nodes and alt_nodes:
                if np.random.random() < 0.2:
                    next_node = np.random.choice(alt_nodes)
                    in_alternative = True
                    alternative_decay = 0.9
                else:
                    next_node = min(base_nodes, key=lambda x: abs(int(x.split('_')[1]) - G.nodes[current_node]['position']))
            else:
                next_node = np.random.choice(neighbors)

        path.append(next_node)
        current_node = next_node
        current_length += len(G.nodes[current_node]['sequence'])

    if len(path) == 1:
        path.extend(list(nx.dfs_preorder_nodes(G, 'base_0'))[1:])


    if len(path) <= 1:
        path.extend(list(nx.dfs_preorder_nodes(G, 'base_0'))[1:])

    return path



def compute_laplacian(G: nx.Graph) -> np.ndarray:
    """Compute the Laplacian matrix of the graph."""
    return nx.laplacian_matrix(G).toarray()

def compute_eigenvectors(L: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the first k eigenvectors of the Laplacian matrix.
    
    Args:
    L (np.ndarray): The Laplacian matrix of the graph.
    k (int): The number of eigenvectors to return.
    
    Returns:
    np.ndarray: The k most important eigenvectors, sorted by importance (descending order of eigenvalues).
    """
    eigenvalues, eigenvectors = linalg.eigh(L)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Try keeping the smallest

    # Return the k most important eigenvectors
    return eigenvectors[:, :k]



def simulate_individuals(G: nx.DiGraph, n_individuals: int, node_embeddings: np.ndarray, 
                         sequences: Dict[str, str], snp_positions: np.ndarray, 
                         random_snp_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
    """Simulate individuals based on the pangenome graph."""
    node_list = list(G.nodes())
    individuals, genotypes, snp_genotypes, individual_paths = [], [], [], []
    
    # Calculate target sequence length
    target_length = sum(len(sequences[node]) for node in node_list if node.startswith('base_'))
    
    # Determine which SNPs will be random vs. sequence-based
    n_snps = len(snp_positions)
    is_random_snp = np.random.random(n_snps) < random_snp_ratio
    
    # Generate random SNPs
    random_snps = np.random.choice([0, 1, 2], size=n_snps)
    
    for _ in range(n_individuals):
        # Generate individual path
        individual_path = generate_individual_path(G, target_length)
        individual_paths.append(individual_path)
        
        # Generate genotype (using node IDs)
        genotype = np.array([1 if any(G.nodes[node]['id'] == G.nodes[path_node]['id'] for path_node in individual_path) else 0 for node in node_list])

        # Calculate individual embedding
        path_indices = [node_list.index(node) for node in individual_path]
        individual_embedding = node_embeddings[path_indices].sum(axis=0)  # Changed from mean to sum
        individual_embedding /= np.linalg.norm(individual_embedding)  # Normalize the embedding
        
        # Generate SNP genotypes
        snp_genotype = []
        for i, pos in enumerate(snp_positions):
            if is_random_snp[i]:
                # Use pre-generated random SNP
                snp = random_snps[i]
            else:
                # Based on sequences
                sequence_genotype = [sequences[node][pos] for node in individual_path]
                reference_allele = sequences[individual_path[0]][pos]
                snp = sum(1 for allele in sequence_genotype if allele != reference_allele)
            snp_genotype.append(snp)
        
        individuals.append(individual_embedding)
        genotypes.append(genotype)
        snp_genotypes.append(snp_genotype)
    
    return np.array(individuals), np.array(genotypes), np.array(snp_genotypes), individual_paths



def create_phenotype_function(sequences: Dict[str, str], snp_positions: np.ndarray, 
                              snp_effects: np.ndarray, snp_weight: float = 1.0) -> Callable:
    
    def phenotype_function(genotypes: np.ndarray, node_list: List[str], snp_genotypes: np.ndarray) -> np.ndarray:
        phenotypes = np.zeros(len(genotypes))
        pattern_weights = np.random.normal(1, 0.2, len(KEY_PATTERNS))
        for i, (genotype, snp_genotype) in enumerate(zip(genotypes, snp_genotypes)):
            pattern_score = sum(sequences[node].count(pattern) * weight 
                                for node, present in zip(node_list, genotype) if present
                                for pattern, weight in zip(KEY_PATTERNS, pattern_weights))
            context_score = sum(sequences[node].count('GC') * 0.1 for node, present in zip(node_list, genotype) if present)
            snp_score = np.dot(snp_genotype, snp_effects)
            phenotypes[i] = pattern_score + context_score + snp_weight * snp_score
            phenotypes[i] += np.random.normal(0, 0.1)
        
        return stats.zscore(phenotypes)
    
    return phenotype_function

def perform_gwas(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = X.shape[1]
    betas, p_values, intercepts = np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)

    X = np.nan_to_num(X, nan=0.0)

    for i in range(n_features):
        if np.all(X[:, i] == X[0, i]):  # Check if all values in the column are the same
            betas[i], p_values[i], intercepts[i] = 0, 1, 0
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X[:, i], y)
            betas[i] = slope
            p_values[i] = max(p_value, np.finfo(float).tiny)
            intercepts[i] = intercept

    return betas, p_values, intercepts

def perform_lasso_gwas(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = X.shape[1]
    X = np.nan_to_num(X, nan=0.0)

    # Perform LASSO
    lasso = Lasso(alpha=0.01, random_state=2024)
    lasso.fit(X, y)

    # Extract coefficients
    betas = lasso.coef_

    # Calculate p-values using f_regression
    _, p_values = f_regression(X, y)

    # Set very small p-values to the smallest possible float
    p_values = np.maximum(p_values, np.finfo(float).tiny)

    # Calculate intercepts (all the same for LASSO)
    intercepts = np.full(n_features, lasso.intercept_)

    return betas, p_values, intercepts

def plot_results(G: nx.Graph, X_test_eigen: np.ndarray, X_test_pca: np.ndarray, y_test: np.ndarray, 
                 y_pred_eigen: np.ndarray, y_pred_normal: np.ndarray, y_pred_pca: np.ndarray, y_pred_lasso: np.ndarray,
                 coefficients_eigen: np.ndarray, coefficients_normal: np.ndarray, coefficients_pca: np.ndarray, coefficients_lasso: np.ndarray,
                 r_squared_eigen: float, r_squared_normal: float, r_squared_pca: float, r_squared_lasso: float,
                 p_values_eigen: np.ndarray, p_values_normal: np.ndarray, p_values_pca: np.ndarray, p_values_lasso: np.ndarray,
                 overall_p_eigen: float, overall_p_normal: float, overall_p_pca: float, overall_p_lasso: float,
                 genotypes: np.ndarray, node_list: List[str], individual_paths: List[List[str]],
                 filename: str, min_cluster_size: int, sequences: Dict[str, str], L: np.ndarray, eigenvectors: np.ndarray) -> None:

    """Plot the results of the GWAS analysis."""
    fig, axs = plt.subplots(3, 3, figsize=(24, 24))

    def format_p_value(p_value):
        if np.isnan(p_value):
            return "NaN"
        elif p_value < 1e-4:
            return f"{p_value:.2e}"
        else:
            return f"{p_value:.4f}"

    fig.suptitle(f"Eigenvector GWAS R² = {r_squared_eigen:.4f} (p = {format_p_value(overall_p_eigen)})\n"
                 f"Normal GWAS R² = {r_squared_normal:.4f} (p = {format_p_value(overall_p_normal)})\n"
                 f"Lasso GWAS R² = {r_squared_lasso:.4f} (p = {format_p_value(overall_p_lasso)})\n"
                 f"PCA GWAS R² = {r_squared_pca:.4f} (p = {format_p_value(overall_p_pca)})")

    # Create a custom layout
    pos = {}
    base_nodes = sorted([node for node in G.nodes() if node.startswith('base_')], key=lambda x: int(x.split('_')[1]))
    variant_nodes = [node for node in G.nodes() if node.startswith('variant_')]
    alt_nodes = [node for node in G.nodes() if node.startswith('alt_')]

    # Position base nodes along a curved line
    t = np.linspace(0, np.pi, len(base_nodes))
    x = t
    y = 0.2 * np.sin(t)
    for i, node in enumerate(base_nodes):
        pos[node] = (x[i], y[i])

    # Position variant and alt nodes
    for node in variant_nodes + alt_nodes:
        base_connections = [n for n in G.neighbors(node) if n.startswith('base_')]
        if base_connections:
            base_x, base_y = pos[base_connections[0]]
            angle = np.random.uniform(-np.pi/3, np.pi/3)
            distance = 0.1 + 0.1 * np.random.random()
            pos[node] = (base_x + distance * np.cos(angle), base_y + distance * np.sin(angle))
        else:
            pos[node] = (np.random.random(), np.random.random())

    # Ensure all nodes have a position
    for node in G.nodes():
        if node not in pos:
            pos[node] = (np.random.random(), np.random.random())
    
    # Main pangenome plot
    axs[0, 0].set_title("Pangenome Graph Structure")
    node_colors = ['blue' if node.startswith('base_') else 'red' if node.startswith('variant_') else 'green' for node in G.nodes()]
    node_sizes = [50 if node.startswith('base_') else 30 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=axs[0, 0], node_color=node_colors, node_size=node_sizes)
    
    edge_colors = ['black' if 'base_' in u and 'base_' in v else 'red' if 'variant_' in u or 'variant_' in v else 'green' for u, v in G.edges()]
    edge_widths = [2 if 'base_' in u and 'base_' in v else 1 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=axs[0, 0], edge_color=edge_colors, width=edge_widths, 
                           connectionstyle="arc3,rad=0.1", arrowsize=10)
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Base', markerfacecolor='blue', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Variant', markerfacecolor='red', markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='Alternative', markerfacecolor='green', markersize=8)]
    axs[0, 0].legend(handles=legend_elements, loc='upper right')

    axs[0, 0].axis('off')

    # Individual genomes
    axs[0, 1].set_title("Individual Genomes")
    num_individuals = min(3, len(individual_paths))
    random_individuals = np.random.choice(len(individual_paths), num_individuals, replace=False)
    for i, idx in enumerate(random_individuals):
        individual_path = individual_paths[idx]
        
        ax_inset = axs[0, 1].inset_axes([0.05, 0.7 - i*0.3, 0.9, 0.25])
        
        # Create a subgraph for this individual
        individual_graph = G.subgraph(individual_path)
        
        # Use the same layout as the main graph, but only for the nodes in this path
        individual_pos = {node: pos[node] for node in individual_path if node in pos}
        
        nx.draw_networkx_nodes(individual_graph, individual_pos, ax=ax_inset,
                               node_color=['blue' if node.startswith('base_') else 'red' if node.startswith('variant_') 
                                           else 'green' if node.startswith('alt_') else 'purple' for node in individual_path],
                               node_size=5)
        
        nx.draw_networkx_edges(individual_graph, individual_pos, ax=ax_inset,
                               edge_color='gray', arrows=True, arrowsize=5,
                               connectionstyle="arc3,rad=0.1")
        
        ax_inset.set_title(f"Individual {idx}", fontsize=8)
        ax_inset.axis('off')

    print("\nNormal GWAS Effect Sizes:")
    print(coefficients_normal)
    print("\nEigenvector GWAS Effect Sizes:")
    print(coefficients_eigen)
    print("\nPCA GWAS Effect Sizes:")
    print(coefficients_pca)
    print("\nLasso GWAS Effect Sizes:")
    print(coefficients_lasso)

    # Plot effect sizes for all GWAS methods
    max_features = max(len(coefficients_normal), len(coefficients_eigen), len(coefficients_pca), len(coefficients_lasso))
    x = np.arange(max_features)
    width = 0.2

    # Pad the shorter coefficient arrays with small values
    small_value = 1e-10
    coefficients_normal_padded = np.pad(coefficients_normal, (0, max_features - len(coefficients_normal)), constant_values=small_value)
    coefficients_eigen_padded = np.pad(coefficients_eigen, (0, max_features - len(coefficients_eigen)), constant_values=small_value)
    coefficients_pca_padded = np.pad(coefficients_pca, (0, max_features - len(coefficients_pca)), constant_values=small_value)
    coefficients_lasso_padded = np.pad(coefficients_lasso, (0, max_features - len(coefficients_lasso)), constant_values=small_value)

    # Plot the effect sizes on a symlog scale
    axs[0, 2].bar(x - 1.5*width, coefficients_normal_padded, width, label='Normal GWAS', alpha=0.7, color='red')
    axs[0, 2].bar(x - 0.5*width, coefficients_eigen_padded, width, label='Eigenvector GWAS', alpha=0.7, color='blue')
    axs[0, 2].bar(x + 0.5*width, coefficients_pca_padded, width, label='PCA GWAS', alpha=0.7, color='green')
    axs[0, 2].bar(x + 1.5*width, coefficients_lasso_padded, width, label='Lasso GWAS', alpha=0.7, color='purple')

    axs[0, 2].set_title("Effect Sizes for All GWAS Methods (Symlog Scale)")
    axs[0, 2].set_xlabel("Feature Index")
    axs[0, 2].set_ylabel("Effect Size")
    axs[0, 2].set_yscale('symlog', linthresh=1e-10)  # Use symlog scale
    axs[0, 2].legend()
    axs[0, 2].set_xticks(x[::5])  # Show every 5th tick to avoid crowding
    axs[0, 2].set_xticklabels(range(1, max_features + 1, 5))

    # Plot phenotype distribution
    sns.histplot(y_test, kde=True, ax=axs[1, 1])
    axs[1, 1].set_title("Phenotype Distribution (Test Set)")
    axs[1, 1].set_xlabel("Phenotype Value")
    axs[1, 1].set_ylabel("Frequency")

    # Plot predicted vs actual phenotype for all GWAS methods
    axs[1, 2].scatter(y_test, y_pred_eigen, alpha=0.6, label='Eigenvector GWAS', color='blue', s=30)
    axs[1, 2].scatter(y_test, y_pred_normal, alpha=0.6, label='Normal GWAS', color='red', s=30)
    axs[1, 2].scatter(y_test, y_pred_pca, alpha=0.6, label='PCA GWAS', color='green', s=30)
    axs[1, 2].scatter(y_test, y_pred_lasso, alpha=0.6, label='Lasso GWAS', color='purple', s=30)

    axs[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Prediction')
    axs[1, 2].set_title("Predicted vs Actual Phenotype")
    axs[1, 2].set_xlabel("Actual Phenotype")
    axs[1, 2].set_ylabel("Predicted Phenotype")
    axs[1, 2].legend()


    # Plot HDBSCAN clustering and scatterplot
    ax3d = fig.add_subplot(3, 3, 7, projection='3d')
    scatter3d = ax3d.scatter(X_test_eigen[:, 0], X_test_eigen[:, 1], X_test_eigen[:, 2], c=y_test, cmap='viridis', s=30)
    ax3d.set_title("Individual Embeddings (with HDBSCAN Clustering)")
    ax3d.set_xlabel("Dimension 1")
    ax3d.set_ylabel("Dimension 2")
    ax3d.set_zlabel("Dimension 3")
    if X_test_eigen.shape[0] > 5:
        clusterer = HDBSCAN(min_cluster_size=min(5, X_test_eigen.shape[0]))
        cluster_labels = clusterer.fit_predict(X_test_eigen)
        unique_labels = set(cluster_labels)
        colors = plt.colormaps['rainbow'](np.linspace(0, 1, len(unique_labels) - 1))
        for k, col in zip(sorted(list(unique_labels - {-1})), colors):
            class_member_mask = (cluster_labels == k)
            xyz = X_test_eigen[class_member_mask]
            ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=50, facecolors='none', edgecolors=col, linewidth=0.6, alpha=0.5)
    ax3d.view_init(elev=20, azim=40)
    ax3d.set_box_aspect((1, 1, 0.5))  # Adjust the box aspect ratio
    
    ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax3d.set_axis_off()
    
    plt.colorbar(scatter3d, ax=ax3d, label='Phenotype')

    def create_ld_matrix(sequences):
        base_seqs = {k: v for k, v in sequences.items() if k.startswith('base_')}
        n = len(next(iter(base_seqs.values())))  # Length of each sequence
        ld_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                column_i = [seq[i] for seq in base_seqs.values()]
                column_j = [seq[j] for seq in base_seqs.values()]
                # Calculate LD based on pairwise letter similarities
                similarities = sum(a == b for a, b in zip(column_i, column_j))
                total_count = len(column_i)
                ld = similarities / total_count
                ld_matrix[i, j] = ld_matrix[j, i] = ld
        return ld_matrix

    def visualize_laplacian(L):
        return sns.heatmap(L, cmap='coolwarm', center=0)

    def visualize_pangenome_eigenvectors(G, eigenvectors, sequences, ax=None):
        if ax is None:
            ax = plt.gca()
        
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        node_colors_1 = eigenvectors[:, 0]
        node_colors_2 = eigenvectors[:, 1]
        
        node_colors_1 = 2 * (node_colors_1 - np.min(node_colors_1)) / (np.max(node_colors_1) - np.min(node_colors_1)) - 1
        node_colors_2 = 2 * (node_colors_2 - np.min(node_colors_2)) / (np.max(node_colors_2) - np.min(node_colors_2)) - 1
        
        def custom_cmap(v1, v2):
            hue = (v2 + 1) / 2
            opacity = 0.5 + 0.5 * (1 - abs(v1))
            return plt.cm.hsv(hue)[:3] + (opacity,)
        
        node_colors = [custom_cmap(v1, v2) for v1, v2 in zip(node_colors_1, node_colors_2)]
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.6, ax=ax)
        
        base_nodes = [node for node in G.nodes() if node.startswith('base_')]
        variant_nodes = [node for node in G.nodes() if node.startswith('variant_')]
        alt_nodes = [node for node in G.nodes() if node.startswith('alt_')]
        
        nx.draw_networkx_nodes(G, pos, nodelist=base_nodes, node_size=100, 
                               node_color=[node_colors[list(G.nodes()).index(node)] for node in base_nodes], 
                               node_shape='o', ax=ax)
        
        nx.draw_networkx_nodes(G, pos, nodelist=variant_nodes, node_size=50, 
                               node_color=[node_colors[list(G.nodes()).index(node)] for node in variant_nodes], 
                               node_shape='s', ax=ax)
        
        nx.draw_networkx_nodes(G, pos, nodelist=alt_nodes, node_size=50, 
                               node_color=[node_colors[list(G.nodes()).index(node)] for node in alt_nodes], 
                               node_shape='^', ax=ax)
        
        ax.set_title("Pangenome Graph with First Two Eigenvectors")
        ax.axis('off')
        
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        cax = ax.inset_axes([1.05, 0.2, 0.05, 0.6])
        custom_cmap = LinearSegmentedColormap.from_list("custom", ["blue", "green", "red"])
        cax.imshow(gradient.T, aspect='auto', cmap=custom_cmap)
        cax.set_title('2nd Eigenvector', fontsize=8)
        cax.set_xticks([])
        cax.set_yticks([0, 255])
        cax.set_yticklabels(['Low', 'High'], fontsize=8)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Base', markerfacecolor='gray', markersize=6),
            Line2D([0], [0], marker='s', color='w', label='Variant', markerfacecolor='gray', markersize=5),
            Line2D([0], [0], marker='^', color='w', label='Alternative', markerfacecolor='gray', markersize=5),
            Line2D([0], [0], marker='o', color='w', label='1st eigenvector: Opacity', markerfacecolor='none', markersize=0),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title='Node Types', fontsize=7, title_fontsize=8, markerscale=0.8)
        
        return ax

    # Create a 1x2 grid of subplots within the [1, 0] subplot
    gs = axs[1, 0].get_gridspec()
    axs[1, 0].remove()
    subaxs = gs[1, 0].subgridspec(1, 2)

    # LD Matrix
    ax_ld = fig.add_subplot(subaxs[0])
    ld_matrix = create_ld_matrix(sequences)
    sns.heatmap(ld_matrix, ax=ax_ld, cmap='coolwarm', vmin=0, vmax=1, cbar_kws={'aspect': 30})
    ax_ld.set_title("LD Matrix")
    ax_ld.set_xlabel("Base Sequence Index")
    ax_ld.set_ylabel("Base Sequence Index")

    # Pangenome with Eigenvectors
    ax_pangenome = fig.add_subplot(subaxs[1])
    visualize_pangenome_eigenvectors(G, eigenvectors, sequences, ax=ax_pangenome)
    ax_pangenome.set_title("Pangenome Graph with First Eigenvector")
    # Plot 3D embedding in the [2, 0] position (bottom left)
    axs[2, 0].remove()  # Remove the existing 2D subplot
    ax_3d = fig.add_subplot(gs[2, 0], projection='3d')
    scatter = ax_3d.scatter(X_test_eigen[:, 0], X_test_eigen[:, 1], X_test_eigen[:, 2], c=y_test, cmap='viridis')
    ax_3d.set_xlabel("Eigenvector 1")
    ax_3d.set_ylabel("Eigenvector 2")
    ax_3d.set_zlabel("Eigenvector 3")
    
    # Add colorbar for phenotype
    cbar = plt.colorbar(scatter, ax=ax_3d, pad=0.1)
    cbar.set_label("Phenotype")
    
    # Adjust 3D plot appearance
    ax_3d.set_box_aspect((1, 1, 0.5))
    ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.xaxis.line.set_color((0.0, 0.0, 0.0, 1.0))
    ax_3d.yaxis.line.set_color((0.0, 0.0, 0.0, 1.0))
    ax_3d.zaxis.line.set_color((0.0, 0.0, 0.0, 1.0))
    
    # Adjust spacing
    plt.tight_layout()

    # Plot p-values for all GWAS methods in the [2, 1] position (bottom middle)
    max_length = max(len(p_values_eigen), len(p_values_normal), len(p_values_pca), len(p_values_lasso))
    x_eigen = range(1, len(p_values_eigen)+1)
    x_normal = range(1, len(p_values_normal)+1)
    x_pca = range(1, len(p_values_pca)+1)
    x_lasso = range(1, len(p_values_lasso)+1)

    axs[2, 1].bar(x_eigen, -np.log10(p_values_eigen), alpha=0.3, label='Eigenvector GWAS', color='blue')
    axs[2, 1].bar(x_normal, -np.log10(p_values_normal), alpha=0.3, label='Normal GWAS', color='red')
    axs[2, 1].bar(x_pca, -np.log10(p_values_pca), alpha=0.3, label='PCA GWAS', color='green')
    axs[2, 1].bar(x_lasso, -np.log10(p_values_lasso), alpha=0.3, label='Lasso GWAS', color='purple')
    axs[2, 1].set_title("GWAS -log10(p-values)")
    axs[2, 1].set_xlabel("Feature Index")
    axs[2, 1].set_ylabel("-log10(p-value)")
    axs[2, 1].axhline(-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
    axs[2, 1].set_xlim(0, max_length)
    axs[2, 1].legend()

    corr_matrix = np.corrcoef([y_pred_normal, y_pred_eigen, y_pred_pca, y_pred_lasso, y_test])
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=['Normal', 'Eigen', 'PCA', 'Lasso', 'True'], 
                yticklabels=['Normal', 'Eigen', 'PCA', 'Lasso', 'True'], ax=axs[2, 2])
    axs[2, 2].set_title("Correlation Heatmap of Predictions and True Values")
        
    fig.suptitle(f"Eigenvector GWAS R² = {r_squared_eigen:.4f} (p = {format_p_value(overall_p_eigen)})\n"
                 f"Normal GWAS R² = {r_squared_normal:.4f} (p = {format_p_value(overall_p_normal)})\n"
                 f"Lasso GWAS R² = {r_squared_lasso:.4f} (p = {format_p_value(overall_p_lasso)})\n"
                 f"PCA GWAS R² = {r_squared_pca:.4f} (p = {format_p_value(overall_p_pca)})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()



def run_simulation(n_base_seqs: int, seq_length: int, n_variants: int, n_individuals: int, 
                   n_dimensions: int, n_components: int, n_snps: int, snp_weight: float, random_snp_ratio: float,
                   test_size: float, min_cluster_size: int, seed: int, suppress_plot: bool = False) -> Tuple[float, float, float, float, float, float, float, float]:

    """Run a single simulation with the given parameters."""
    G, sequences, snp_positions, snp_effects = create_pangenome_graph(n_base_seqs, seq_length, n_variants, n_snps)
    node_list = list(G.nodes())

    L = compute_laplacian(G)
    eigenvectors = compute_eigenvectors(L, n_dimensions)

    individuals, genotypes, snp_genotypes, individual_paths = simulate_individuals(G, n_individuals, eigenvectors, sequences, snp_positions, random_snp_ratio)

    # Remove NaN values
    mask = ~np.isnan(individuals).any(axis=1)
    individuals = individuals[mask]
    genotypes = genotypes[mask]
    snp_genotypes = snp_genotypes[mask]
    individual_paths = [individual_paths[i] for i in range(len(individual_paths)) if mask[i]]

    # Apply normalization after NaN removal
    individuals = stats.zscore(individuals, axis=0)
    phenotype_func = create_phenotype_function(sequences, snp_positions, snp_effects, snp_weight)

    raw_phenotypes = phenotype_func(genotypes, node_list, snp_genotypes)
    print(f"Raw phenotype stats: mean={np.mean(raw_phenotypes)}, var={np.var(raw_phenotypes)}")
    phenotypes = stats.zscore(raw_phenotypes)

    # Perform PCA
    print(f"Number of PCA components: {n_components}")

    pca = CustomPCA(n_components=n_components)

    # Replace NaN with zeros in snp_genotypes, individuals, and phenotypes
    snp_genotypes = np.nan_to_num(snp_genotypes, nan=0.0)
    individuals = np.nan_to_num(individuals, nan=0.0)
    phenotypes = np.nan_to_num(phenotypes, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(snp_genotypes, phenotypes, test_size=test_size, random_state=seed)
    X_train_eigen, X_test_eigen, y_train_eigen, y_test_eigen = train_test_split(individuals, phenotypes, test_size=test_size, random_state=seed)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"X_train shape: {X_train.shape}")

    coefficients_normal, p_values_normal, intercepts_normal = perform_gwas(X_train, y_train)
    coefficients_eigen, p_values_eigen, intercepts_eigen = perform_gwas(X_train_eigen, y_train_eigen)
    coefficients_pca, p_values_pca, intercepts_pca = perform_gwas(X_train_pca, y_train)
    coefficients_lasso, p_values_lasso, intercepts_lasso = perform_lasso_gwas(X_train, y_train)

    print("Normal GWAS coefficients:", coefficients_normal)
    print("Eigenvector GWAS coefficients:", coefficients_eigen)
    print("PCA GWAS coefficients:", coefficients_pca)
    print("Lasso GWAS coefficients:", coefficients_lasso)

    def z_norm(x):
        return (x - np.mean(x)) / np.std(x)

    y_pred_normal = np.dot(X_test, coefficients_normal) + np.mean(intercepts_normal)
    y_pred_eigen = np.dot(X_test_eigen, coefficients_eigen) + np.mean(intercepts_eigen)
    y_pred_pca = np.dot(X_test_pca, coefficients_pca) + np.mean(intercepts_pca)
    y_pred_lasso = np.dot(X_test, coefficients_lasso) + np.mean(intercepts_lasso)

    # Z-normalize y_test and all predictions
    y_test = z_norm(y_test)
    y_pred_normal = z_norm(y_pred_normal)
    y_pred_eigen = z_norm(y_pred_eigen)
    y_pred_pca = z_norm(y_pred_pca)
    y_pred_lasso = z_norm(y_pred_lasso)

    def unadjusted_r2(y_true, y_pred):
        return np.corrcoef(y_true, y_pred)[0, 1]**2

    if len(y_test) > 1:
        r_squared_normal_full = unadjusted_r2(y_test, y_pred_normal)
        r_squared_eigen_full = unadjusted_r2(y_test, y_pred_eigen)
        r_squared_pca_full = unadjusted_r2(y_test, y_pred_pca)
        r_squared_lasso_full = unadjusted_r2(y_test, y_pred_lasso)

        print("Y_test range:", np.min(y_test), np.max(y_test))
        print("Y_pred_normal range:", np.min(y_pred_normal), np.max(y_pred_normal))
        print("Y_pred_eigen range:", np.min(y_pred_eigen), np.max(y_pred_eigen))
        print("Y_pred_pca range:", np.min(y_pred_pca), np.max(y_pred_pca))
        print("Y_pred_lasso range:", np.min(y_pred_lasso), np.max(y_pred_lasso))
        
        corr_normal = np.corrcoef(y_test, y_pred_normal)[0, 1]
        corr_eigen = np.corrcoef(y_test, y_pred_eigen)[0, 1]
        corr_pca = np.corrcoef(y_test, y_pred_pca)[0, 1]
        corr_lasso = np.corrcoef(y_test, y_pred_lasso)[0, 1]
        
        print(f"Normal GWAS correlation: {corr_normal:.4f}")
        print(f"Eigenvector GWAS correlation: {corr_eigen:.4f}")
        print(f"PCA GWAS correlation: {corr_pca:.4f}")
        print(f"Lasso GWAS correlation: {corr_lasso:.4f}")
    else:
        r_squared_normal_full = r_squared_eigen_full = r_squared_pca_full = r_squared_lasso_full = np.nan

    n_test = X_test.shape[0]
    n_features = X_test.shape[1]
    
    def calculate_p_value(r_squared, n_test, n_features):
        if r_squared >= 1 or r_squared <= 0:
            return 1.0
        f_stat = (r_squared / (1 - r_squared)) * ((n_test - n_features - 1) / n_features)
        return max(1 - stats.f.cdf(f_stat, n_features, n_test - n_features - 1), np.finfo(float).tiny)

    overall_p_normal = calculate_p_value(r_squared_normal_full, n_test, X_test.shape[1])
    overall_p_eigen = calculate_p_value(r_squared_eigen_full, n_test, X_test_eigen.shape[1])
    overall_p_pca = calculate_p_value(r_squared_pca_full, n_test, X_test_pca.shape[1])
    overall_p_lasso = calculate_p_value(r_squared_lasso_full, n_test, X_test.shape[1])
    print(f"Sample size: {n_individuals}")
    
    print(f"Number of SNPs: {n_snps}")
    print(f"Number of dimensions: {n_dimensions}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Normal GWAS R²: {r_squared_normal_full:.4f}")
    print(f"Eigenvector GWAS R²: {r_squared_eigen_full:.4f}")
    print(f"PCA GWAS R²: {r_squared_pca_full:.4f}")
    print(f"Lasso GWAS R²: {r_squared_lasso_full:.4f}")
    print(f"Normal GWAS p-value: {overall_p_normal:.4e}")
    print(f"Eigenvector GWAS p-value: {overall_p_eigen:.4e}")
    print(f"PCA GWAS p-value: {overall_p_pca:.4e}")
    print(f"Lasso GWAS p-value: {overall_p_lasso:.4e}")

    filename = f"results_{n_individuals}.png"
    plot_results(G, X_test_eigen, X_test_pca, y_test, 
                 y_pred_eigen, y_pred_normal, y_pred_pca, y_pred_lasso,
                 coefficients_eigen, coefficients_normal, coefficients_pca, coefficients_lasso,
                 r_squared_eigen_full, r_squared_normal_full, r_squared_pca_full, r_squared_lasso_full,
                 p_values_eigen, p_values_normal, p_values_pca, p_values_lasso,
                 overall_p_eigen, overall_p_normal, overall_p_pca, overall_p_lasso,
                 genotypes, node_list, individual_paths, filename, min_cluster_size,
                 sequences, L, eigenvectors)

    if not suppress_plot:
        os.system(f"open {filename}")
    return r_squared_eigen_full, r_squared_normal_full, r_squared_pca_full, r_squared_lasso_full, overall_p_eigen, overall_p_normal, overall_p_pca, overall_p_lasso

def run_varied_parameter_simulation(**params):
    vary_param = params.pop('vary_param')
    csv_filename = params.pop('csv_filename')
    suppress_plot = params.pop('suppress_plot')
    max_individuals = params.pop('max_individuals')
    min_individuals = params.pop('min_individuals')
    num_steps = params.pop('num_steps')

    if vary_param == 'n_individuals':
        param_values = np.logspace(np.log10(min_individuals), np.log10(max_individuals), num_steps).astype(int)
    else:
        param_values = np.linspace(params[vary_param] * 0.125, params[vary_param] * 8, num_steps)

    results = []
    for value in param_values:
        current_params = params.copy()
        current_params[vary_param] = value
        r2_eigen, r2_normal, r2_pca, r2_lasso, p_eigen, p_normal, p_pca, p_lasso = run_simulation(**current_params, suppress_plot=suppress_plot)
        
        result = {
            vary_param: value,
            'r2_eigen': r2_eigen,
            'r2_normal': r2_normal,
            'r2_pca': r2_pca,
            'r2_lasso': r2_lasso,
            'p_eigen': p_eigen,
            'p_normal': p_normal,
            'p_pca': p_pca,
            'p_lasso': p_lasso
        }
        results.append(result)
        
        # Save results to CSV in real-time
        save_results_to_csv(results, csv_filename)

    plot_varied_parameter_results(results, vary_param, suppress_plot)
    return results

def save_results_to_csv(results, filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(timestamped_filename, index=False)
    print(f"Results saved to {timestamped_filename}")

def plot_varied_parameter_results(results, vary_param, suppress_plot):
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
    ax1.semilogx(df[vary_param], df['r2_eigen'], label='Eigenvector GWAS', marker='o')
    ax1.semilogx(df[vary_param], df['r2_normal'], label='Normal GWAS', marker='s')
    ax1.semilogx(df[vary_param], df['r2_pca'], label='PCA GWAS', marker='^')
    ax1.semilogx(df[vary_param], df['r2_lasso'], label='Lasso GWAS', marker='D')
    ax1.set_xlabel(f'{vary_param} (log scale)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score vs ' + vary_param)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
        
    ax2.semilogx(df[vary_param], [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in df['p_eigen']], label='Eigenvector GWAS')
    ax2.semilogx(df[vary_param], [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in df['p_normal']], label='Normal GWAS')
    ax2.semilogx(df[vary_param], [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in df['p_pca']], label='PCA GWAS')
    ax2.semilogx(df[vary_param], [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in df['p_lasso']], label='Lasso GWAS')
    ax2.set_xlabel(f'{vary_param} (log scale)')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Significance vs ' + vary_param)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{vary_param}_comparison.png")
    if not suppress_plot:
        plt.show()
    plt.close()
    print(f"{vary_param} comparison plot saved as '{vary_param}_comparison.png'")

def run_multiple_simulations(params, num_repeats=1):
    """Run multiple simulations and aggregate results."""
    all_results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(num_repeats):
        print(f"Running simulation {i+1}/{num_repeats}")
        csv_filename = f"simulation_results_{timestamp}_{i+1}.csv"
        params['csv_filename'] = csv_filename
        results = run_varied_parameter_simulation(**params)
        all_results.extend(results)
        
    # Aggregate results
    df = pd.DataFrame(all_results)
    
    # Calculate median R² and p-values
    median_results = df.groupby(params['vary_param']).agg({
        'r2_eigen': lambda x: np.median([r for r in x if r > -1 and not np.isnan(r)]),
        'r2_normal': lambda x: np.median([r for r in x if r > -1 and not np.isnan(r)]),
        'r2_pca': lambda x: np.median([r for r in x if r > -1 and not np.isnan(r)]),
        'r2_lasso': lambda x: np.median([r for r in x if r > -1 and not np.isnan(r)]),
        'p_eigen': 'median',
        'p_normal': 'median',
        'p_pca': 'median',
        'p_lasso': 'median'
    }).reset_index()
    
    # Plot median R² and p-values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    ax1.semilogx(median_results[params['vary_param']], median_results['r2_eigen'], label='Eigenvector GWAS', marker='o')
    ax1.semilogx(median_results[params['vary_param']], median_results['r2_normal'], label='Normal GWAS', marker='s')
    ax1.semilogx(median_results[params['vary_param']], median_results['r2_pca'], label='PCA GWAS', marker='^')
    ax1.semilogx(median_results[params['vary_param']], median_results['r2_lasso'], label='Lasso GWAS', marker='D')
    ax1.set_xlabel(f'{params["vary_param"]} (log scale)')
    ax1.set_ylabel('Median R² Score')
    ax1.set_title(f'Median R² Score vs {params["vary_param"]} ({num_repeats} repeats)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
    
    ax2.semilogx(median_results[params['vary_param']], -np.log10(median_results['p_eigen']), label='Eigenvector GWAS', marker='o')
    ax2.semilogx(median_results[params['vary_param']], -np.log10(median_results['p_normal']), label='Normal GWAS', marker='s')
    ax2.semilogx(median_results[params['vary_param']], -np.log10(median_results['p_pca']), label='PCA GWAS', marker='^')
    ax2.semilogx(median_results[params['vary_param']], -np.log10(median_results['p_lasso']), label='Lasso GWAS', marker='D')
    ax2.set_xlabel(f'{params["vary_param"]} (log scale)')
    ax2.set_ylabel('Median -log10(p-value)')
    ax2.set_title(f'Median Significance vs {params["vary_param"]} ({num_repeats} repeats)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{params['vary_param']}_comparison_median_{num_repeats}_repeats.png")
    if not params['suppress_plot']:
        plt.show()
    plt.close()
    print(f"Median comparison plot saved as '{params['vary_param']}_comparison_median_{num_repeats}_repeats.png'")

def main():
    """Main execution function."""
    #set_random_seed()
    
    # Simulation parameters
    params = {
        'n_base_seqs': 20,        # Number of base sequences forming the backbone of the pangenome graph
        'seq_length': 200,        # Length of each base sequence
        'n_variants': 100,        # Number of variant sequences to generate (randomly mutated from base sequences)
        'n_individuals': 1000,    # Initial number of individuals to simulate
        'max_individuals': 100000, # Maximum number of individuals
        'n_dimensions': 50,       # Number of dimensions for eigenvector decomposition (max: number of graph nodes)
        'n_components': 50,       # Number of principal components to use in PCA GWAS
        'n_snps': 100,            # Number of single nucleotide polymorphisms (SNPs) that may influence phenotype
        'snp_weight': 1.0,        # Weight of SNP effects on phenotype
        'random_snp_ratio': 0.95, # Ratio of random SNPs to sequence-based SNPs
        'min_individuals': 15,    # Minimum number of individuals
        'num_steps': 20,          # Number of steps between min and max individuals for scaling
        'test_size': 0.5,         # Proportion of data to use as test set in train-test split
        'min_cluster_size': 5,    # Minimum cluster size for HDBSCAN clustering algorithm
        'seed': int(time.time()),  # Random seed based on current time
        'suppress_plot': True,    # Whether to suppress plot display
        'vary_param': 'n_individuals', # Parameter to vary in scaling experiments
        'csv_filename': 'simulation_results.csv' # Filename for saving simulation results
    }

    print("Simulation parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    print("Starting pangenome GWAS simulation...")
    num_repeats = 5
    run_multiple_simulations(params, num_repeats)
    print("Simulation completed. Check the generated plots and CSV files for results.")

    # Print example sequences
    G, sequences, _, _ = create_pangenome_graph(params['n_base_seqs'], params['seq_length'], params['n_variants'], params['n_snps'])
    print("\nExample sequences:")
    for i, (node, seq) in enumerate(sequences.items()):
        if i < 5:  # Print first 5 sequences
            print(f"{node}: {seq[:50]}...")  # Print first 50 characters
        else:
            break
    
    print(f"\nTotal number of sequences: {len(sequences)}")
    print(f"Total number of nodes in the graph: {len(G.nodes())}")
    print(f"Total number of edges in the graph: {len(G.edges())}")

if __name__ == "__main__":
    main()
