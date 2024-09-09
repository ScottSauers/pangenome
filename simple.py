import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy import linalg, stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import Tuple, List, Dict, Callable
import pandas as pd
import time

def set_random_seed(seed: int = int(time.time())) -> None:
    """Set the random seed based on current time"""
    np.random.seed(seed)

def create_sequence(length: int) -> np.ndarray:
    """Create a random binary sequence of given length."""
    return np.random.randint(0, 2, size=length)

def create_pangenome_graph(n_base_seqs: int, seq_length: int, n_variants: int) -> Tuple[nx.DiGraph, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Create a pangenome graph with SNP positions for each node."""
    G = nx.DiGraph()
    sequences = {}
    snp_positions = {}
    
    # Create base sequences
    for i in range(n_base_seqs):
        seq = create_sequence(seq_length)
        node_name = f"base_{i}"
        G.add_node(node_name, sequence=seq, weight=np.random.uniform(0.5, 1.0))
        sequences[node_name] = seq
        snp_positions[node_name] = np.sort(np.random.choice(seq_length, size=n_variants, replace=False))
        if i > 0:
            G.add_edge(f"base_{i-1}", node_name)

    # Create variants (modifications and insertions)
    for i in range(n_variants):
        if np.random.random() < 0.7:  # 70% chance of modification, 30% chance of insertion
            base_node = np.random.choice([node for node in G.nodes() if node.startswith('base_')])
            base_seq = sequences[base_node]
            var_seq = base_seq.copy()
            
            # Modify sequence
            n_mutations = np.random.randint(1, seq_length // 10)
            mutation_positions = np.random.choice(seq_length, size=n_mutations, replace=False)
            var_seq[mutation_positions] = 1 - var_seq[mutation_positions]
            
            node_name = f"variant_{i}"
            G.add_node(node_name, sequence=var_seq, weight=np.random.uniform(0.1, 0.5))
            sequences[node_name] = var_seq
            
            # Inherit some SNPs, remove some, and add new ones
            inherited_snps = set(snp_positions[base_node]) - set(mutation_positions)
            new_snps = np.random.choice(seq_length, size=n_variants - len(inherited_snps), replace=False)
            snp_positions[node_name] = np.sort(list(inherited_snps) + list(new_snps))
            
            G.add_edge(base_node, node_name)
        else:
            # Create a completely new insertion
            insert_seq = create_sequence(seq_length)
            node_name = f"insertion_{i}"
            G.add_node(node_name, sequence=insert_seq, weight=np.random.uniform(0.05, 0.2))
            sequences[node_name] = insert_seq
            snp_positions[node_name] = np.sort(np.random.choice(seq_length, size=n_variants, replace=False))
            
            # Connect to a random existing node
            existing_node = np.random.choice(list(G.nodes()))
            G.add_edge(existing_node, node_name)

    return G, sequences, snp_positions

def compute_laplacian(G: nx.Graph) -> np.ndarray:
    """Compute the Laplacian matrix of the graph."""
    return nx.laplacian_matrix(G).toarray()

def compute_eigenvectors(L: np.ndarray, k: int) -> np.ndarray:
    """Compute the first k eigenvectors of the Laplacian matrix."""
    eigenvalues, eigenvectors = linalg.eigh(L)
    idx = eigenvalues.argsort()[::-1]
    return eigenvectors[:, idx[:k]]

def simulate_individuals(G: nx.DiGraph, n_individuals: int, node_embeddings: np.ndarray, 
                         sequences: Dict[str, np.ndarray], snp_positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], List[List[str]]]:
    """Simulate individuals based on the pangenome graph."""
    node_list = list(G.nodes())
    individuals, genotypes, paths = [], [], []
    
    for _ in range(n_individuals):
        # Generate individual path based on node weights
        path = []
        current_node = np.random.choice([node for node in G.nodes() if node.startswith('base_')])
        while True:
            path.append(current_node)
            neighbors = list(G.neighbors(current_node))
            if not neighbors:
                break
            weights = [G.nodes[n]['weight'] for n in neighbors]
            current_node = np.random.choice(neighbors, p=np.array(weights) / sum(weights))
        
        # Generate genotype
        genotype = []
        for node in path:
            node_genotype = sequences[node][snp_positions[node]]
            genotype.extend(node_genotype)
        
        # Calculate individual embedding
        path_indices = [node_list.index(node) for node in path]
        individual_embedding = node_embeddings[path_indices].sum(axis=0)
        norm = np.linalg.norm(individual_embedding)
        if norm > 0:
            individual_embedding /= norm
        else:
            individual_embedding = np.zeros_like(individual_embedding)
        
        individuals.append(individual_embedding)
        genotypes.append(np.array(genotype))
        paths.append(path)
    
    return np.array(individuals), genotypes, paths

def create_phenotype_function(G: nx.DiGraph, snp_positions: Dict[str, np.ndarray]) -> Callable:
    effect_sizes = {}
    for node in G.nodes():
        effect_sizes[node] = np.random.normal(0, 1, len(snp_positions[node]))
    
    def phenotype_function(genotypes: List[np.ndarray], paths: List[List[str]]) -> np.ndarray:
        phenotypes = []
        for genotype, path in zip(genotypes, paths):
            phenotype = 0
            start = 0
            for node in path:
                end = start + len(snp_positions[node])
                phenotype += np.dot(genotype[start:end], effect_sizes[node])
                start = end
            phenotypes.append(phenotype)
        return stats.zscore(np.array(phenotypes))
    return phenotype_function

def perform_gwas(X: List[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_length = max(len(x) for x in X)
    betas = np.zeros(max_length)
    p_values = np.zeros(max_length)

    for i in range(max_length):
        valid_indices = [j for j, x in enumerate(X) if i < len(x)]
        if len(valid_indices) < 2:
            betas[i] = 0
            p_values[i] = 1
        else:
            x_values = np.array([X[j][i] for j in valid_indices])
            y_values = y[valid_indices]
            if np.all(x_values == x_values[0]):
                betas[i] = 0
                p_values[i] = 1
            else:
                slope, _, _, p_value, _ = stats.linregress(x_values, y_values)
                betas[i] = slope
                p_values[i] = p_value

    return betas, p_values

def predict(X: List[np.ndarray], betas: np.ndarray, top_k: int) -> np.ndarray:
    top_indices = np.argsort(np.abs(betas))[-top_k:]
    predictions = []
    for x in X:
        valid_indices = [i for i in top_indices if i < len(x)]
        pred = np.dot(x[valid_indices], betas[valid_indices])
        predictions.append(pred)
    return stats.zscore(np.array(predictions))

def run_simulation(n_base_seqs: int, seq_length: int, n_variants: int, n_individuals: int, 
                   test_size: float, top_k: int) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    G, sequences, snp_positions = create_pangenome_graph(n_base_seqs, seq_length, n_variants)

    L = compute_laplacian(G)
    eigenvectors = compute_eigenvectors(L, top_k)

    individuals, genotypes, paths = simulate_individuals(G, n_individuals, eigenvectors, sequences, snp_positions)

    phenotype_func = create_phenotype_function(G, snp_positions)
    phenotypes = phenotype_func(genotypes, paths)

    X_train, X_test, X_train_eigen, X_test_eigen, y_train, y_test = train_test_split(
        genotypes, individuals, phenotypes, test_size=test_size)

    betas_normal, _ = perform_gwas(X_train, y_train)
    betas_eigen, _ = perform_gwas(X_train_eigen, y_train)

    y_pred_normal = predict(X_test, betas_normal, top_k)
    y_pred_eigen = predict(X_test_eigen, betas_eigen, top_k)

    r_normal = np.corrcoef(y_test, y_pred_normal)[0, 1]
    r_eigen = np.corrcoef(y_test, y_pred_eigen)[0, 1]

    return r_normal, r_eigen, y_test, y_pred_normal, y_pred_eigen, phenotypes

def main():
    set_random_seed()
    
    n_base_seqs = 25
    seq_length = 400
    n_variants = 160
    test_size = 0.5
    top_k = 200
    n_individuals = 80000
    
    try:
        r_normal, r_eigen, y_test, y_pred_normal, y_pred_eigen, phenotypes = run_simulation(n_base_seqs, seq_length, n_variants, n_individuals, test_size, top_k)

        print(f"Results for {n_individuals} individuals:")
        print(f"Correlation (Normal GWAS): {r_normal}")
        print(f"Correlation (Eigenvector GWAS): {r_eigen}")

        y_test_norm = stats.zscore(y_test)
        y_pred_normal_norm = stats.zscore(y_pred_normal)
        y_pred_eigen_norm = stats.zscore(y_pred_eigen)

        # Create and save the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('GWAS Results Comparison', fontsize=20, fontweight='bold')

        # Scatter plot
        ax1.scatter(y_test_norm, y_pred_normal_norm, alpha=0.6, label='Normal GWAS', color='#1f77b4', edgecolor='w', s=50)
        ax1.scatter(y_test_norm, y_pred_eigen_norm, alpha=0.6, label='Eigenvector GWAS', color='#ff7f0e', edgecolor='w', s=50)
        ax1.plot([y_test_norm.min(), y_test_norm.max()], [y_test_norm.min(), y_test_norm.max()], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Phenotype', fontsize=14)
        ax1.set_ylabel('Predicted Phenotype', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.set_title('Predicted vs Actual Phenotype', fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=12)

        # Bar plot
        bar_width = 0.35
        index = np.arange(2)
        ax2.bar(index, [r_normal, r_eigen], bar_width, alpha=0.8, 
                color=['#1f77b4', '#ff7f0e'], 
                label=['Normal GWAS', 'Eigenvector GWAS'])
        ax2.set_ylabel('Correlation (r)', fontsize=14)
        ax2.set_title('Correlation Comparison', fontsize=16)
        ax2.set_xticks(index)
        ax2.set_xticklabels(['Normal GWAS', 'Eigenvector GWAS'], rotation=45, ha='right')
        ax2.tick_params(axis='both', which='major', labelsize=12)

        for i, v in enumerate([r_normal, r_eigen]):
            ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plt.savefig('gwas_results.png', dpi=300, bbox_inches='tight')
        plt.close()

        os.system('open gwas_results.png')

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
