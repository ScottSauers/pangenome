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

def mutate_sequence(sequence: str, mutation_rate: float = DEFAULT_MUTATION_RATE) -> str:
    """Mutate a given DNA sequence with a specified mutation rate."""
    return ''.join(np.random.choice(BASE_NUCLEOTIDES) if np.random.random() < mutation_rate else base
                   for base in sequence)







def create_pangenome_graph(n_base_seqs: int, seq_length: int, n_variants: int, n_snps: int) -> Tuple[nx.DiGraph, Dict[str, str], np.ndarray, np.ndarray]:
    """Create a pangenome graph."""
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

    for base_index in range(n_base_seqs - 1):
        branches = 0
        while branches < max_branches_per_base and np.random.random() < 0.7:
            branch_type = np.random.choice(['shared_alt', 'variant', 'alt'], p=[0.2, 0.5, 0.3])
            branch_length = np.random.randint(3, 6) if branch_type == 'shared_alt' else np.random.randint(1, 4)
            
            prev_node = f"base_{base_index}"
            start_position = base_index
            for j in range(branch_length):
                var_seq = mutate_sequence(sequences[prev_node])
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
                         random_snp_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[str]]]:
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
    """Create a function to generate phenotypes based on genotypes and SNPs."""
    pattern_weights = np.random.normal(1, 0.2, len(KEY_PATTERNS))
    
    def phenotype_function(genotypes: np.ndarray, node_list: List[str], snp_genotypes: np.ndarray) -> np.ndarray:
        phenotypes = np.zeros(len(genotypes))
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


def perform_gwas(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, np.ndarray, np.ndarray]:
    n_features = X.shape[1]
    betas, p_values = np.zeros(n_features), np.zeros(n_features)

    # Replace NaN values with zero in X
    X = np.nan_to_num(X, nan=0.0)

    # Check for infinite values in X
    if np.any(np.isinf(X)):
        print("Warning: Input X contains infinite values.")
        inf_cols = np.where(np.any(np.isinf(X), axis=0))[0]
        print(f"Indices of columns with infinite values: {inf_cols}")
        for col in inf_cols:
            print(f"Column {col}: {X[:, col][np.isinf(X[:, col])]}")

    # Check for NaN values in X (after replacement)
    if np.any(np.isnan(X)):
        print("Warning: Input X still contains NaN values after replacement.")
        nan_cols = np.where(np.any(np.isnan(X), axis=0))[0]
        print(f"Indices of columns with NaN values: {nan_cols}")
        for col in nan_cols:
            nan_values = X[:, col][np.isnan(X[:, col])]
            print(f"Column {col}: {nan_values}")
            print(f"Number of NaN values in column {col}: {len(nan_values)}")

    # Check for infinite values in y
    if np.any(np.isinf(y)):
        print("Warning: Input y contains infinite values.")
        print(f"Infinite values in y: {y[np.isinf(y)]}")

    # Check for NaN values in y
    if np.any(np.isnan(y)):
        print("Warning: Input y contains NaN values.")
        print(f"NaN values in y: {y[np.isnan(y)]}")
        print(f"Number of NaN values in y: {np.sum(np.isnan(y))}")

    for i in range(n_features):
        if np.all(X[:, i] == X[0, i]):  # Check if all values in the column are the same
            betas[i], p_values[i] = 0, 1
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X[:, i], y)
            betas[i], p_values[i] = slope, max(p_value, np.finfo(float).tiny)

    model = LinearRegression().fit(X, y)

    return model, betas, p_values



def plot_results(G: nx.Graph, X_test_eigen: np.ndarray, X_test_pca: np.ndarray, y_test: np.ndarray, 
                 y_pred_eigen: np.ndarray, y_pred_normal: np.ndarray, y_pred_pca: np.ndarray,
                 coefficients_eigen: np.ndarray, coefficients_normal: np.ndarray, coefficients_pca: np.ndarray,
                 r_squared_eigen: float, r_squared_normal: float, r_squared_pca: float,
                 p_values_eigen: np.ndarray, p_values_normal: np.ndarray, p_values_pca: np.ndarray,
                 overall_p_eigen: float, overall_p_normal: float, overall_p_pca: float,
                 genotypes: np.ndarray, node_list: List[str], individual_paths: List[List[str]],
                 filename: str, min_cluster_size: int) -> None:

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

    # Plot effect sizes for all GWAS methods
    max_features = max(len(coefficients_normal), len(coefficients_eigen), len(coefficients_pca))
    x = np.arange(max_features)
    width = 0.25


    # Pad the shorter coefficient arrays with small values
    small_value = 1e-10
    coefficients_normal_padded = np.pad(coefficients_normal, (0, max_features - len(coefficients_normal)), constant_values=small_value)
    coefficients_eigen_padded = np.pad(coefficients_eigen, (0, max_features - len(coefficients_eigen)), constant_values=small_value)
    coefficients_pca_padded = np.pad(coefficients_pca, (0, max_features - len(coefficients_pca)), constant_values=small_value)


    # Plot the effect sizes on a symlog scale
    axs[0, 2].bar(x - width, coefficients_normal_padded, width, label='Normal GWAS', alpha=0.7, color='red')
    axs[0, 2].bar(x, coefficients_eigen_padded, width, label='Eigenvector GWAS', alpha=0.7, color='blue')
    axs[0, 2].bar(x + width, coefficients_pca_padded, width, label='PCA GWAS', alpha=0.7, color='green')

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

    axs[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Prediction')
    axs[1, 2].set_title("Predicted vs Actual Phenotype")
    axs[1, 2].set_xlabel("Actual Phenotype")
    axs[1, 2].set_ylabel("Predicted Phenotype")
    axs[1, 2].legend()

    # Add PCA embedding plot
    axs[2, 2].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis')
    axs[2, 2].set_title("PCA Embeddings")
    axs[2, 2].set_xlabel("PC1")
    axs[2, 2].set_ylabel("PC2")

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
    plt.colorbar(scatter3d, ax=ax3d, label='Phenotype')
    fig.delaxes(axs[2, 0])
    fig.delaxes(axs[2, 2])
    plt.tight_layout() 

    # Plot p-values for all GWAS methods
    max_length = max(len(p_values_eigen), len(p_values_normal), len(p_values_pca))
    x_eigen = range(1, len(p_values_eigen)+1)
    x_normal = range(1, len(p_values_normal)+1)
    x_pca = range(1, len(p_values_pca)+1)

    axs[2, 1].bar(x_eigen, -np.log10(p_values_eigen), alpha=0.3, label='Eigenvector GWAS', color='blue')
    axs[2, 1].bar(x_normal, -np.log10(p_values_normal), alpha=0.3, label='Normal GWAS', color='red')
    axs[2, 1].bar(x_pca, -np.log10(p_values_pca), alpha=0.3, label='PCA GWAS', color='green')
    axs[2, 1].set_title("GWAS -log10(p-values)")
    axs[2, 1].set_xlabel("Feature Index")
    axs[2, 1].set_ylabel("-log10(p-value)")
    axs[2, 1].axhline(-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
    axs[2, 1].set_xlim(0, max_length)
    axs[2, 1].legend()

    corr_matrix = np.corrcoef([y_pred_normal, y_pred_eigen, y_pred_pca, y_test])
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=['Normal', 'Eigen', 'PCA', 'True'], 
                yticklabels=['Normal', 'Eigen', 'PCA', 'True'], ax=axs[2, 2])
    axs[2, 2].set_title("Correlation Heatmap of Predictions and True Values")
        
    fig.suptitle(f"Eigenvector GWAS R² = {r_squared_eigen:.4f} (p = {format_p_value(overall_p_eigen)})\n"
                 f"Normal GWAS R² = {r_squared_normal:.4f} (p = {format_p_value(overall_p_normal)})\n"
                 f"PCA GWAS R² = {r_squared_pca:.4f} (p = {format_p_value(overall_p_pca)})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()




def run_simulation(n_base_seqs: int, seq_length: int, n_variants: int, n_individuals: int, 
                   n_dimensions: int, n_components: int, n_snps: int, snp_weight: float, random_snp_ratio: float,
                   test_size: float, min_cluster_size: int, seed: int) -> Tuple[float, float, float, float, float, float]:

    """Run a single simulation with the given parameters."""
    G, sequences, snp_positions, snp_effects = create_pangenome_graph(n_base_seqs, seq_length, n_variants, n_snps)
    node_list = list(G.nodes())

    L = compute_laplacian(G)
    eigenvectors = compute_eigenvectors(L, n_dimensions)

    individuals, genotypes, snp_genotypes, individual_paths = simulate_individuals(G, n_individuals, eigenvectors, sequences, snp_positions)


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


    # Replace NaN with zeros in snp_genotypes
    snp_genotypes = np.nan_to_num(snp_genotypes, nan=0.0)


    # Replace NaN with zeros in individuals
    individuals = np.nan_to_num(individuals, nan=0.0)


    # Replace NaN with zeros in phenotypes
    phenotypes = np.nan_to_num(phenotypes, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(snp_genotypes, phenotypes, test_size=test_size, random_state=seed)
    X_train_eigen, X_test_eigen, y_train_eigen, y_test_eigen = train_test_split(individuals, phenotypes, test_size=test_size, random_state=seed)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)



    print(f"X_train shape: {X_train.shape}")

    model_normal, coefficients_normal, p_values_normal = perform_gwas(X_train, y_train)
    model_eigen, coefficients_eigen, p_values_eigen = perform_gwas(X_train_eigen, y_train_eigen)
    model_pca, coefficients_pca, p_values_pca = perform_gwas(X_train_pca, y_train)

    y_pred_normal = model_normal.predict(X_test)
    y_pred_eigen = model_eigen.predict(X_test_eigen)
    y_pred_pca = model_pca.predict(X_test_pca)

    if len(y_test) > 1:
        r_squared_normal_full = r2_score(y_test, y_pred_normal)
        r_squared_eigen_full = r2_score(y_test_eigen, y_pred_eigen)
        r_squared_pca_full = r2_score(y_test, y_pred_pca)
    else:
        r_squared_normal_full = r_squared_eigen_full = r_squared_pca_full = np.nan

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
    print(f"Normal GWAS p-value: {overall_p_normal:.4e}")
    print(f"Eigenvector GWAS p-value: {overall_p_eigen:.4e}")
    print(f"PCA GWAS p-value: {overall_p_pca:.4e}")

    filename = f"results_{n_individuals}.png"
    plot_results(G, X_test_eigen, X_test_pca, y_test, 
                 y_pred_eigen, y_pred_normal, y_pred_pca,
                 coefficients_eigen, coefficients_normal, coefficients_pca,
                 r_squared_eigen_full, r_squared_normal_full, r_squared_pca_full,
                 p_values_eigen, p_values_normal, p_values_pca,
                 overall_p_eigen, overall_p_normal, overall_p_pca,
                 genotypes, node_list, individual_paths, filename, min_cluster_size)

    os.system(f"open {filename}")
    return r_squared_eigen_full, r_squared_normal_full, r_squared_pca_full, overall_p_eigen, overall_p_normal, overall_p_pca



def compare_sample_sizes(n_base_seqs: int, seq_length: int, n_variants: int, max_individuals: int, 
                         n_dimensions: int, n_components: int, n_snps: int, snp_weight: float, random_snp_ratio: float,
                         min_individuals: int, num_steps: int, test_size: float, min_cluster_size: int,
                         seed: int) -> None:
    """Compare GWAS performance across different sample sizes."""
    
    # Create a log space between min_individuals and max_individuals
    sample_sizes = np.logspace(np.log10(min_individuals), np.log10(max_individuals), num=num_steps).astype(int)
    sample_sizes = np.maximum(sample_sizes, n_dimensions + 1)  # Sample size is greater than n_dimensions
    
    sample_sizes = np.unique(sample_sizes)

    eigen_r2, normal_r2, pca_r2, eigen_p, normal_p, pca_p = [], [], [], [], [], []
    
    for n_individuals in tqdm(sample_sizes, desc="Simulating different sample sizes"):
        r2_eigen, r2_normal, r2_pca, p_eigen, p_normal, p_pca = run_simulation(
            n_base_seqs, seq_length, n_variants, n_individuals, n_dimensions, n_components, n_snps, snp_weight,
            random_snp_ratio, test_size, min_cluster_size, seed
        )
        eigen_r2.append(r2_eigen)
        normal_r2.append(r2_normal)
        pca_r2.append(r2_pca)
        eigen_p.append(p_eigen)
        normal_p.append(p_normal)
        pca_p.append(p_pca)

    print("Simulation results:")
    for i, n in enumerate(sample_sizes):
        print(f"Sample size: {n}")
        print(f"  Eigenvector GWAS R²: {eigen_r2[i]:.4f}, p-value: {eigen_p[i]:.4e}")
        print(f"  Normal GWAS R²: {normal_r2[i]:.4f}, p-value: {normal_p[i]:.4e}")
        print(f"  PCA GWAS R²: {pca_r2[i]:.4f}, p-value: {pca_p[i]:.4e}")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
    ax1.semilogx(sample_sizes, eigen_r2, label='Eigenvector GWAS', marker='o')
    ax1.semilogx(sample_sizes, normal_r2, label='Normal GWAS', marker='s')
    ax1.semilogx(sample_sizes, pca_r2, label='PCA GWAS', marker='^')
    ax1.set_xlabel('Number of Individuals (log scale)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score vs Sample Size')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(bottom=0)
        
    ax2.semilogx(sample_sizes, [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in eigen_p], label='Eigenvector GWAS')
    ax2.semilogx(sample_sizes, [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in normal_p], label='Normal GWAS')
    ax2.semilogx(sample_sizes, [-np.log10(p) if not np.isnan(p) and p > 0 else 0 for p in pca_p], label='PCA GWAS')
    ax2.set_xlabel('Number of Individuals (log scale)')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Significance vs Sample Size')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    print("R² scores:")
    print("Eigenvector GWAS:", eigen_r2)
    print("Normal GWAS:", normal_r2)
    print("PCA GWAS:", pca_r2)
    print("\np-values:")
    print("Eigenvector GWAS:", eigen_p)
    print("Normal GWAS:", normal_p)
    print("PCA GWAS:", pca_p)
    
    plt.tight_layout()
    plt.savefig("sample_size_comparison.png")
    plt.close()
    print("Sample size comparison plot saved as 'sample_size_comparison.png'")
    os.system(f"open sample_size_comparison.png")

def main():
    """Main execution function."""
    set_random_seed()
    
    # Simulation parameters
    params = {
        'n_base_seqs': 20,  # Backbone of pangenome graph
        'seq_length': 200,  # How long each base sequence is
        'n_variants': 100,  # Randomly mutate letters. Creates variants which individuals may or may not have
        'max_individuals': 110000,
        'n_dimensions': 50,  # Maximum number of dimensions equals the number of nodes in the graph
        'n_components': 50,  # Number of PCA components
        'n_snps': 100,  # How many letters may end up influencing phenotype
        'snp_weight': 1.0,
        'random_snp_ratio': 0.95,  # Ratio of random SNPs to sequence-based SNPs
        'min_individuals': 15,  # Minimum number of individuals
        'num_steps': 20,  # Number of steps between min and max individuals
        'test_size': 0.5,  # Test set size for train-test split
        'min_cluster_size': 5,  # Minimum cluster size for HDBSCAN
        'seed': 2024,
    }

    
    print("Simulation parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    print("Starting pangenome GWAS simulation...")
    compare_sample_sizes(**params)
    print("Simulation completed. Check the generated plots for results.")

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
