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

# Constants
BASE_NUCLEOTIDES = ['A', 'T', 'C', 'G']
DEFAULT_SNV_RATE = 0.01
DEFAULT_MUTATION_RATE = 0.01
KEY_PATTERNS = ['ATCG', 'GCAT', 'TGCA', 'CGTA', 'TACG', 'GATC', 'CTAG', 'AGCT']

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

def create_pangenome_graph(n_base_seqs: int, seq_length: int, n_variants: int, n_snps: int) -> Tuple[nx.Graph, Dict[str, str], np.ndarray, np.ndarray]:
    """Create a pangenome graph with specified parameters."""
    G = nx.Graph()
    sequences = {}
    snp_positions = np.random.choice(seq_length, size=n_snps, replace=False)
    snp_effects = np.random.normal(0, 1, n_snps)
    
    # Create base sequences
    for i in range(n_base_seqs):
        seq = create_sequence(seq_length)
        node_name = f"base_{i}"
        G.add_node(node_name)
        sequences[node_name] = seq
        if i > 0:
            G.add_edge(f"base_{i-1}", node_name)
    
    # Create variants
    for i in range(n_variants):
        base_index = np.random.randint(n_base_seqs)
        base_node = f"base_{base_index}"
        var_seq = mutate_sequence(sequences[base_node])
        node_name = f"variant_{i}"
        G.add_node(node_name)
        sequences[node_name] = var_seq
        G.add_edge(base_node, node_name)
        
        if np.random.random() < 0.5 and base_index < n_base_seqs - 1:
            reconnect_index = np.random.randint(base_index + 1, n_base_seqs)
            G.add_edge(node_name, f"base_{reconnect_index}")

    return G, sequences, snp_positions, snp_effects

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

def simulate_individuals(G: nx.Graph, n_individuals: int, node_embeddings: np.ndarray, 
                         sequences: Dict[str, str], snp_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate individuals based on the pangenome graph."""
    node_list = list(G.nodes())
    individuals, genotypes, snp_genotypes = [], [], []
    
    for _ in range(n_individuals):
        individual_nodes = [node for node in node_list if node.startswith("base_")]
        individual_nodes += [node for node in node_list if node.startswith("variant_") and np.random.random() < 0.5]
        genotype = [1 if node in individual_nodes else 0 for node in node_list]
        individual_embedding = node_embeddings[[node_list.index(node) for node in individual_nodes]].mean(axis=0)
        
        snp_genotype = [sum(1 for allele in [sequences[node][pos] for node in individual_nodes] if allele != sequences[individual_nodes[0]][pos])
                        for pos in snp_positions]
        
        individuals.append(individual_embedding)
        genotypes.append(genotype)
        snp_genotypes.append(snp_genotype)
    
    return np.array(individuals), np.array(genotypes), np.array(snp_genotypes)

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
        
        return stats.zscore(phenotypes)
    
    return phenotype_function

def perform_gwas(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, np.ndarray, np.ndarray]:
    n_features = X.shape[1]
    betas, p_values = np.zeros(n_features), np.zeros(n_features)
    
    for i in range(n_features):
        slope, _, _, p_value, _ = stats.linregress(X[:, i], y)
        betas[i], p_values[i] = slope, p_value
    
    model = LinearRegression().fit(X, y)
    
    return model, betas, p_values







def plot_results(G: nx.Graph, X_test_eigen: np.ndarray, y_test: np.ndarray, y_pred_eigen: np.ndarray, 
                 y_pred_normal: np.ndarray, coefficients_eigen: np.ndarray, coefficients_normal: np.ndarray, 
                 r_squared_eigen: float, r_squared_normal: float, p_values_eigen: np.ndarray, 
                 p_values_normal: np.ndarray, overall_p_eigen: float, overall_p_normal: float, 
                 genotypes: np.ndarray, node_list: List[str], filename: str) -> None:

    """Plot the results of the GWAS analysis."""
    fig, axs = plt.subplots(3, 3, figsize=(24, 24))

    fig.suptitle(f"Eigenvector GWAS R² = {r_squared_eigen:.4f} (p = {overall_p_eigen:.2e})\n"
                 f"Normal GWAS R² = {r_squared_normal:.4f} (p = {overall_p_normal:.2e})")
    
    pos = nx.spring_layout(G)
    
    # Main pangenome plot
    axs[0, 0].set_title("Pangenome Graph Structure")
    nx.draw(G, pos, ax=axs[0, 0], node_color=['blue' if node.startswith('base') else 'red' for node in G.nodes()], 
            node_size=20, with_labels=False)
    
    # Individual genomes
    axs[0, 1].set_title("Individual Genomes")
    num_individuals = min(3, len(genotypes))
    random_individuals = np.random.choice(len(genotypes), num_individuals, replace=False)
    for i, idx in enumerate(random_individuals):
        individual_genome = genotypes[idx]
        individual_graph = nx.Graph()
        for j, present in enumerate(individual_genome):
            if present:
                individual_graph.add_node(node_list[j])
                if j > 0 and individual_genome[j-1]:
                    individual_graph.add_edge(node_list[j-1], node_list[j])
        
        ax_inset = axs[0, 1].inset_axes([0.05, 0.7 - i*0.3, 0.9, 0.25])
        nx.draw(individual_graph, pos, ax=ax_inset, 
            node_color=['blue' if node.startswith('base') else 'red' for node in individual_graph.nodes()],
            node_size=5, with_labels=False)
        ax_inset.set_title(f"Individual {idx}", fontsize=8)
        ax_inset.axis('off')
    axs[0, 1].axis('off')

    # Plot effect sizes for both GWAS methods
    x = range(1, len(coefficients_eigen)+1)
    axs[0, 2].bar(x, coefficients_eigen, alpha=0.5, label='Eigenvector GWAS', color='blue')
    axs[0, 2].bar(x, coefficients_normal, alpha=0.5, label='Normal GWAS', color='red')
    axs[0, 2].set_title("GWAS Effect Sizes")
    axs[0, 2].set_xlabel("SNP / Eigenvector Index")
    axs[0, 2].set_ylabel("Effect Size")
    axs[0, 2].legend()
    axs[0, 2].axhline(y=0, color='k', linestyle='--')



    # Plot phenotype distribution
    sns.histplot(y_test, kde=True, ax=axs[1, 1])
    axs[1, 1].set_title("Phenotype Distribution (Test Set)")
    axs[1, 1].set_xlabel("Phenotype Value")
    axs[1, 1].set_ylabel("Frequency")

    # Plot predicted vs actual phenotype for both GWAS methods
    axs[1, 2].scatter(y_test, y_pred_eigen, alpha=0.5, label='Eigenvector GWAS', color='blue')
    axs[1, 2].scatter(y_test, y_pred_normal, alpha=0.5, label='Normal GWAS', color='red')
    axs[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    axs[1, 2].set_title("Predicted vs Actual Phenotype")
    axs[1, 2].set_xlabel("Actual Phenotype")
    axs[1, 2].set_ylabel("Predicted Phenotype")
    axs[1, 2].legend()

    # Plot HDBSCAN clustering and scatterplot
    if X_test_eigen.shape[0] > 5:  # Only perform clustering if we have enough samples
        clusterer = HDBSCAN(min_cluster_size=min(5, X_test_eigen.shape[0]))
        cluster_labels = clusterer.fit_predict(X_test_eigen)
        scatter = axs[2, 0].scatter(X_test_eigen[:, 0], X_test_eigen[:, 1], c=y_test, cmap='viridis')
        unique_labels = set(cluster_labels)
        colors = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(unique_labels) - 1))
        for k, col in zip(sorted(list(unique_labels - {-1})), colors):
            class_member_mask = (cluster_labels == k)
            xy = X_test_eigen[class_member_mask, 0:2]
            axs[2, 0].scatter(xy[:, 0], xy[:, 1], s=50, facecolors='none', edgecolors=col, linewidth=0.6, alpha=0.5)
        axs[2, 0].set_title("Individual Embeddings (HDBSCAN Clustering)")
    else:
        scatter = axs[2, 0].scatter(X_test_eigen[:, 0], X_test_eigen[:, 1], c=y_test, cmap='viridis')
        axs[2, 0].set_title("Individual Embeddings (Scatter Plot)")
    
    axs[2, 0].set_xlabel("Dimension 1")
    axs[2, 0].set_ylabel("Dimension 2")
    plt.colorbar(scatter, ax=axs[2, 0], label='Phenotype')

    # Plot p-values for both GWAS methods
    axs[2, 1].bar(range(1, len(p_values_eigen)+1), -np.log10(p_values_eigen), alpha=0.5, label='Eigenvector GWAS')
    axs[2, 1].bar(range(1, len(p_values_normal)+1), -np.log10(p_values_normal), alpha=0.5, label='Normal GWAS')
    axs[2, 1].set_title("GWAS -log10(p-values)")
    axs[2, 1].set_xlabel("SNP / Embedding Dimension")
    axs[2, 1].set_ylabel("-log10(p-value)")
    axs[2, 1].axhline(-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
    axs[2, 1].legend()

    # Clear the unused subplot
    fig.delaxes(axs[2, 2])
    
    plt.suptitle(f"Eigenvector GWAS R² = {r_squared_eigen:.4f} (p = {overall_p_eigen:.2e})\n"
                 f"Normal GWAS R² = {r_squared_normal:.4f} (p = {overall_p_normal:.2e})")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()



def run_simulation(n_base_seqs: int, seq_length: int, n_variants: int, n_individuals: int, 
                   n_dimensions: int, n_snps: int, snp_weight: float) -> Tuple[float, float, float, float]:
    """Run a single simulation with the given parameters."""
    G, sequences, snp_positions, snp_effects = create_pangenome_graph(n_base_seqs, seq_length, n_variants, n_snps)
    node_list = list(G.nodes())

    L = compute_laplacian(G)
    eigenvectors = compute_eigenvectors(L, n_dimensions)

    individuals, genotypes, snp_genotypes = simulate_individuals(G, n_individuals, eigenvectors, sequences, snp_positions)
    individuals = stats.zscore(individuals, axis=0)
    phenotype_func = create_phenotype_function(sequences, snp_positions, snp_effects, snp_weight)
    phenotypes = stats.zscore(phenotype_func(genotypes, node_list, snp_genotypes))

    X_train, X_test, y_train, y_test = train_test_split(snp_genotypes, phenotypes, test_size=0.2, random_state=42)
    X_train_eigen, X_test_eigen, y_train_eigen, y_test_eigen = train_test_split(individuals, phenotypes, test_size=0.2, random_state=42)



    model_normal, coefficients_normal, p_values_normal = perform_gwas(X_train, y_train)
    model_eigen, coefficients_eigen, p_values_eigen = perform_gwas(X_train_eigen, y_train_eigen)

    y_pred_normal = model_normal.predict(X_test)
    y_pred_eigen = model_eigen.predict(X_test_eigen)

    r_squared_normal_full = r2_score(y_test, y_pred_normal)
    r_squared_eigen_full = r2_score(y_test_eigen, y_pred_eigen)


    n_test = X_test.shape[0]
    n_features = X_test.shape[1]

    f_stat_normal = (r_squared_normal_full / (1 - r_squared_normal_full)) * ((n_test - n_features - 1) / n_features)
    overall_p_normal = 1 - stats.f.cdf(f_stat_normal, n_features, n_test - n_features - 1)

    f_stat_eigen = (r_squared_eigen_full / (1 - r_squared_eigen_full)) * ((n_test - n_features - 1) / n_features)
    overall_p_eigen = 1 - stats.f.cdf(f_stat_eigen, n_features, n_test - n_features - 1)

    filename = f"results_{n_individuals}.png"
    plot_results(G, X_test_eigen, y_test, y_pred_eigen, y_pred_normal, coefficients_eigen, coefficients_normal, 
                 r_squared_eigen_full, r_squared_normal_full, p_values_eigen, p_values_normal, 
                 overall_p_eigen, overall_p_normal, genotypes, node_list, filename)

    os.system(f"open {filename}")
    return r_squared_eigen_full, r_squared_normal_full, overall_p_eigen, overall_p_normal

def compare_sample_sizes(n_base_seqs: int, seq_length: int, n_variants: int, max_individuals: int, 
                         n_dimensions: int, n_snps: int, snp_weight: float) -> None:
    """Compare GWAS performance across different sample sizes."""
    min_individuals = 4  # Set min number of individuals
    num_steps = 10  # Number of steps between min and max
    
    # Create a log space between min_individuals and max_individuals
    sample_sizes = np.logspace(np.log10(min_individuals), np.log10(max_individuals), num=num_steps).astype(int)
    
    sample_sizes = np.unique(sample_sizes)

    eigen_r2, normal_r2, eigen_p, normal_p = [], [], [], []
    
    for n_individuals in tqdm(sample_sizes, desc="Simulating different sample sizes"):
        r2_eigen, r2_normal, p_eigen, p_normal = run_simulation(n_base_seqs, seq_length, n_variants, n_individuals, n_dimensions, n_snps, snp_weight)
        eigen_r2.append(r2_eigen)
        normal_r2.append(r2_normal)
        eigen_p.append(p_eigen)
        normal_p.append(p_normal)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    ax1.semilogx(sample_sizes, eigen_r2, label='Eigenvector GWAS')
    ax1.semilogx(sample_sizes, normal_r2, label='Normal GWAS')
    ax1.set_xlabel('Number of Individuals (log scale)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score vs Sample Size')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.semilogx(sample_sizes, -np.log10(eigen_p), label='Eigenvector GWAS')
    ax2.semilogx(sample_sizes, -np.log10(normal_p), label='Normal GWAS')
    ax2.set_xlabel('Number of Individuals (log scale)')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Significance vs Sample Size')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
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
        'n_base_seqs': 50, # Backbone of pangenome graph
        'seq_length': 200, # How long each base sequence is
        'n_variants': 50, # Randomly mutate 50 letters. Creates variants which indviduals may or may not have
        'max_individuals': 10000,
        'n_dimensions': 50, # Maximum number of dimensions equals the number of nodes in the graph 
        'n_snps': 50, # How many letters may end up influencing phenotype
        'snp_weight': 1
    }
    
    print("Starting pangenome GWAS simulation...")
    compare_sample_sizes(**params)
    print("Simulation completed. Check the generated plots for results.")

if __name__ == "__main__":
    main()
