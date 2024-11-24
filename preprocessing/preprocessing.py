import dgl
import torch
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse
import community as community_louvain
import random 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

class Preprocessing:
    
    def __init__(self, data_path: str, folder_name: str, num_sampled_vectors: int):
        self.data_path = data_path
        self.gene2vec_path = 'Gene2vec/'
        self.folder_name = folder_name
        self.num_sampled_vectors = num_sampled_vectors
        try:
            os.makedirs(folder_name)
        except Exception as e:
            print(e)
        self.plot_folder = os.path.join(folder_name, 'plots')
        self.csv_folder = os.path.join(folder_name, 'csv')
        self.pkl_folder = os.path.join(folder_name, 'pkl')
        self.metrics_folder = os.path.join(folder_name, 'metrics')
        self.gridsearch_folder = os.path.join(self.plot_folder, 'louvain_gridsearch')
        self.evaluation_folder = os.path.join(self.plot_folder, 'evaluation')
        for folder in [self.plot_folder, self.csv_folder, self.pkl_folder, self.metrics_folder, self.gridsearch_folder, self.evaluation_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                
        if not os.path.exists(self.gene2vec_path):
            try:
                os.system("git clone https://github.com/jingcheng-du/Gene2vec.git")
            except Exception as e:
                print(e)
        else:
            self.gene2vec_path += 'pre_trained_emb/gene2vec_dim_200_iter_9.txt'
        self.data = pd.read_csv(self.data_path)

    def gene2vec_data(self):
        self.gene2vec_dict = {}
        with open(self.gene2vec_path, 'r') as file:
            for line in file:
                parts = line.split()
                gene_name = parts[0]
                values = [float(x) for x in parts[1:]]
                values = np.array(values, dtype=np.float32)
                self.gene2vec_dict[gene_name] = values
        print(f'Number of Gene2Vec genes: {len(list(self.gene2vec_dict.keys()))}')

    def missing_G2V_genes(self):
        # Missing genes from Gene2Vec data 
        self.missing_genes = list(np.setdiff1d(self.nodes, np.array(list(self.gene2vec_dict.keys()), dtype=np.object_)))
        self.missing_genes_df = self.data[self.data['gene1'].isin(self.missing_genes) | self.data['gene2'].isin(self.missing_genes)]
        print(f"Number of missing genes: {len(self.missing_genes)}")
        print(f"Number of missing genes edges: {len(self.missing_genes_df)}")
        # Save missing_genes
        missing_genes_path = os.path.join(self.pkl_folder, 'missing_genes.pkl')
        with open(missing_genes_path, 'wb') as f:
            pickle.dump(self.missing_genes, f)
        # Save missing_genes_df
        missing_genes_df_path = os.path.join(self.pkl_folder, 'missing_genes_df.pkl')
        with open(missing_genes_df_path, 'wb') as f:
            pickle.dump(self.missing_genes_df, f)

    def detect_communities(self, resolution):
        gene_name_edges = [(gene1, gene2) for gene1, gene2 in zip(self.data['gene1'], self.data['gene2'])]
        G = nx.Graph()
        G.add_edges_from(gene_name_edges)  
        partition = community_louvain.best_partition(G, resolution=resolution)
        communities_dict = {}
        for gene, comm_id in partition.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(gene)
        communities_list = list(communities_dict.values())   
        count = sum(1 for community in communities_list if len(community) == 1)
        print(f"Number of singletons: {count}")
        return partition, communities_dict, G

    def find_missing_bridges(self):
        bridges = nx.bridges(self.G)
        print(f"Total number of nodes: {len(self.G.edges())}")
        bridge_edges_set = set(bridges)
        # find missing bridges 
        missing_edges_set = set(tuple(x) for x in self.missing_genes_df[['gene1', 'gene2']].to_numpy())
        missing_bridges = bridge_edges_set.intersection(missing_edges_set)
        print(f"Number of missing bridges: {len(missing_bridges)}")
        #find missing bridges genes 
        self.missing_gene_bridges = {gene for edge in missing_bridges for gene in edge}
        self.missing_gene_bridges = self.missing_gene_bridges.intersection(self.missing_genes)
        print(f'Number of missing genes in bridges: {len(self.missing_gene_bridges)}')
        # Check if bridge edges are in the DGL graph
        bridge_edges_id = {(self.node_to_id[gene1], self.node_to_id[gene2]) for gene1, gene2 in bridge_edges_set}
        missing_bridges_in_graph = set()
        for edge in bridge_edges_id:
            if not self.g.has_edges_between(edge[0], edge[1]):
                missing_bridges_in_graph.add(edge)
        if missing_bridges_in_graph:
            print(f"Bridges not in the DGL graph: {len(missing_bridges_in_graph)}")
        else:
            print("All identified bridge edges are present in the DGL graph.")
        bridge_edges_file = bridge_edges_id - missing_bridges_in_graph
        print(f'Number of bridges in the file: {len(bridge_edges_file)}')
        with open(os.path.join(self.pkl_folder, 'bridge_edges_set.pkl'), 'wb') as file:
            pickle.dump(bridge_edges_file, file)
        print(f"Bridge edges set saved to: {self.pkl_folder}")

    def louvain_gridsearch(self):
        resolution_values = np.linspace(0, 2, 1)
        mean_community_sizes = []
        singletons_count = []
        modularity_values = []
        num_communities = []
        
        for i, resolution in enumerate(resolution_values):
            print(f'resolution {resolution}')
            partition, communities, G = self.detect_communities(resolution)
            community_sizes = [len(community) for community in list(communities.values())]
            singletons = sum(1 for community in community_sizes if community == 1)
            singletons_count.append(singletons)
            mean_size = np.mean(community_sizes) if community_sizes else 0
            mean_community_sizes.append(round(mean_size))
            modularity = community_louvain.modularity(partition, G)
            print(modularity)
            modularity_values.append(modularity)
            num_communities.append(len(communities))
            print(f"Mean community size: {round(mean_size)}")
        
        max_modularity_index = np.argmax(modularity_values)
        highlight_resolution = resolution_values[max_modularity_index]
        highlight_color = 'red'
        
        plt.figure()
        plt.plot(resolution_values, modularity_values, label='Modularity', color='darkblue', marker='o')
        plt.scatter([highlight_resolution], [modularity_values[max_modularity_index]], color=highlight_color, zorder=5, label='Maximum Modularity Point')
        plt.grid(True)
        plt.xlabel('Resolution')
        plt.ylabel('Modularity')
        plt.savefig(os.path.join(self.gridsearch_folder, 'modularity.png'))

        plt.figure()
        plt.plot(resolution_values, mean_community_sizes, color='darkblue', marker='o')
        plt.scatter([highlight_resolution], [mean_community_sizes[max_modularity_index]], color=highlight_color, zorder=5)
        plt.grid(True)
        plt.xlabel('Resolution')
        plt.ylabel('Mean Community Size')
        plt.savefig(os.path.join(self.gridsearch_folder, 'mean_community_size.png'))

        plt.figure()
        plt.plot(resolution_values, singletons_count, color='darkblue', marker='o')
        plt.scatter([highlight_resolution], [singletons_count[max_modularity_index]], color=highlight_color, zorder=5)
        plt.grid(True)
        plt.xlabel('Resolution')
        plt.ylabel('Singletons')
        plt.savefig(os.path.join(self.gridsearch_folder, 'singletons.png'))

        plt.figure()
        plt.plot(resolution_values, num_communities, color='darkblue', marker='o')
        plt.scatter([highlight_resolution], [num_communities[max_modularity_index]], color=highlight_color, zorder=5)
        plt.grid(True)
        plt.xlabel('Resolution')
        plt.ylabel('Number of Communities')
        plt.savefig(os.path.join(self.gridsearch_folder, 'num_communities.png'))

        return highlight_resolution

    def convex_generator(self, missing_genes, resolution):
        node_vectors = self.gene2vec_dict.copy()
        partition, communities, _ = self.detect_communities(resolution)
        missing_gene_communities = {gene: partition[gene] for gene in missing_genes if gene in partition}
        vector_type = {}
        singleton_missing = 0
        print('Number of missing gene communities', len(list(missing_gene_communities.values())))
        for gene, comm_id in missing_gene_communities.items():
            community = communities[comm_id]
            gene2vec_members = [member for member in community if member in self.gene2vec_dict and gene != member]
            if len(gene2vec_members) >= 2:
                members_degrees = np.array([self.degrees_gene[member] for member in gene2vec_members])
                members_vectors = np.array([self.gene2vec_dict[member] for member in gene2vec_members])
                coefficients = members_degrees / members_degrees.sum()
                generated_vector = np.dot(coefficients, members_vectors)
                if np.isnan(generated_vector).any():
                    print(f'Error with Gene: {gene}, coefficients: {coefficients.shape}, number of community members: {len(gene2vec_members)}, degrees summation: {members_degrees.sum()}')
                node_vectors[gene] = generated_vector
                vector_type[gene] = 'convex'
            else:
                node_vectors[gene] = np.mean(np.array(list(self.gene2vec_dict.values())), axis=0)
                singleton_missing += 1 
                vector_type[gene] = 'mean'
        print(f'Number of missing genes that has fewer than 2 neighbors: {singleton_missing}')        
        return node_vectors, vector_type

    def convex_evaluation(self, resolution):
        intersected_genes = list(set(self.gene2vec_dict.keys()) & set(self.nodes))
        sampled_genes = random.sample(intersected_genes, self.num_sampled_vectors)
        generated_vectors, types = self.convex_generator(sampled_genes, resolution)
        comparisons = {}
        cosine_similarities = []
        euclidean_distances = []
        pearson_correlations = []
        sampled_genes_giaemena = random.sample(list(self.gene2vec_dict.keys()), 2)
        print('Cosine similarity between two random Genes', cosine_similarity([self.gene2vec_dict[sampled_genes_giaemena[0]]], [self.gene2vec_dict[sampled_genes_giaemena[1]]])[0][0])

        debug_samples = random.sample(sampled_genes, 5)
        for gene in debug_samples:
            print(f"Original vector for {gene}: {self.gene2vec_dict[gene][:5]}...")
            print(f"Generated vector for {gene}: {generated_vectors[gene][:5]}...")
        
        mean_count = 0
        convex_count = 0
        convex_genes = []
        mean_genes = []
        for gene in sampled_genes:
            if types[gene] == 'mean':
                mean_count += 1
                mean_genes.append(gene)
            elif types[gene] == 'convex':
                convex_count += 1
                convex_genes.append(gene)
                original_vector = self.gene2vec_dict[gene]
                generated_vector = generated_vectors[gene]
                if np.isnan(original_vector).any():
                    continue
                cosine_similarity_value = cosine_similarity([original_vector], [generated_vector])[0][0]
                cosine_similarities.append(cosine_similarity_value)
                euclidean_distance_value = np.linalg.norm(original_vector - generated_vector)
                euclidean_distances.append(euclidean_distance_value)
                pearson_correlation_value = np.corrcoef(original_vector, generated_vector)[0, 1]
                pearson_correlations.append(pearson_correlation_value)
                comparisons[gene] = (cosine_similarity_value, euclidean_distance_value, pearson_correlation_value, types[gene])

        print(mean_count)

        thresholds = [0.7, 0.9]
        cosine_similarity_categories = pd.cut(cosine_similarities, bins=[-1] + thresholds + [1], labels=["Low", "Moderate", "High"])
        euclidean_distance_categories = pd.cut(euclidean_distances, bins=[-np.inf, thresholds[0], thresholds[1], np.inf], labels=["Low", "Moderate", "High"])
        pearson_correlation_categories = pd.cut(pearson_correlations, bins=[-1] + thresholds + [1], labels=["Low", "Moderate", "High"])
        
        cosine_similarity_df = pd.DataFrame({
            "Gene": convex_genes,
            "Cosine Similarity": cosine_similarities,
            "Category": cosine_similarity_categories
        })
        euclidean_distance_df = pd.DataFrame({
            "Gene": convex_genes,
            "Euclidean Distance": euclidean_distances,
            "Category": euclidean_distance_categories
        })
        pearson_correlation_df = pd.DataFrame({
            "Gene": convex_genes,
            "Pearson Correlation": pearson_correlations,
            "Category": pearson_correlation_categories
        })

        cosine_similarity_df.to_csv(os.path.join(self.csv_folder, 'cosine_similarity.csv'), index=False)
        euclidean_distance_df.to_csv(os.path.join(self.csv_folder, 'euclidean_distance.csv'), index=False)
        pearson_correlation_df.to_csv(os.path.join(self.csv_folder, 'pearson_correlation.csv'), index=False)

        mean_cosine_similarity = np.mean([i for i, _, _, _ in comparisons.values()])
        mean_euclidean_distance = np.mean([j for _, j, _, _ in comparisons.values()])
        mean_pearson_correlation = np.mean([k for _, _, k, _ in comparisons.values()])
        print(f'Mean cosine similarity: {mean_cosine_similarity}')
        print(f'Mean Euclidean distance: {mean_euclidean_distance}')
        print(f'Mean Pearson correlation: {mean_pearson_correlation}')
        print(f'Number of sampled genes with mean values due to lack of neighbors: {mean_count}')
        print(f'Number of sampled genes with convex combination: {convex_count}')

 
        plt.figure()
        plt.hist(cosine_similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Convex Combination')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, 'cosine_similarity_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(euclidean_distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Convex Combination')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, 'euclidean_distance_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(pearson_correlations, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Convex Combination')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, 'pearson_correlation_distribution.png'))
        plt.close()

        tsne = TSNE(n_components=2, random_state=42, n_iter=100000)
        all_vectors = np.array([self.gene2vec_dict[gene] for gene in convex_genes if gene in self.gene2vec_dict])
        generated_vecs = np.array([generated_vectors[gene] for gene in convex_genes if gene in generated_vectors])
        
        if len(mean_genes) > 0:
            mean_vectors = np.array([generated_vectors[gene] for gene in mean_genes if gene in generated_vectors])
            tsne_results = tsne.fit_transform(np.vstack((all_vectors, generated_vecs, mean_vectors)))
            original_tsne = tsne_results[:len(all_vectors)]
            generated_tsne = tsne_results[len(all_vectors):len(all_vectors) + len(generated_vecs)]
            mean_tsne = tsne_results[len(all_vectors) + len(generated_vecs):]
            plt.figure(figsize=(10, 5))
            plt.scatter(original_tsne[:, 0], original_tsne[:, 1], color='blue', label='Original Embeddings', alpha=0.5)
            plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], color='red', label='Generated Convex Embeddings', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(self.evaluation_folder, 'tsne_visualization.png'))
            plt.close()
        else:
            tsne_results = tsne.fit_transform(np.vstack((all_vectors, generated_vecs)))
            original_tsne = tsne_results[:len(all_vectors)]
            generated_tsne = tsne_results[len(all_vectors):]
            plt.figure(figsize=(10, 5))
            plt.scatter(original_tsne[:, 0], original_tsne[:, 1], color='blue', label='Original Embeddings', alpha=0.5)
            plt.scatter(generated_tsne[:, 0], generated_tsne[:, 1], color='red', label='Generated Convex Embeddings', alpha=0.5)
            plt.legend()
            plt.savefig(os.path.join(self.evaluation_folder, 'tsne_visualization.png'))
            plt.close()

        mean_cosine_similarities = []
        mean_euclidean_distances = []
        mean_pearson_correlations = []
        for gene in mean_genes:
            original_vector = self.gene2vec_dict[gene]
            generated_vector = generated_vectors[gene]
            if np.isnan(original_vector).any():
                continue
            cosine_similarity_value = cosine_similarity([original_vector], [generated_vector])[0][0]
            mean_cosine_similarities.append(cosine_similarity_value)
            euclidean_distance_value = np.linalg.norm(original_vector - generated_vector)
            mean_euclidean_distances.append(euclidean_distance_value)
            mean_pearson_correlation_value = np.corrcoef(original_vector, generated_vector)[0, 1]
            mean_pearson_correlations.append(mean_pearson_correlation_value)

        mean_mean_cosine_similarity = np.mean(mean_cosine_similarities)
        mean_mean_euclidean_distance = np.mean(mean_euclidean_distances)
        mean_mean_pearson_correlation = np.mean(mean_pearson_correlations)
        print(f'Mean cosine similarity for mean vectors: {mean_mean_cosine_similarity}')
        print(f'Mean Euclidean distance for mean vectors: {mean_mean_euclidean_distance}')
        print(f'Mean Pearson correlation for mean vectors: {mean_mean_pearson_correlation}')

    def generate_random_vectors(self, dim=200):
        random_vectors = {}
        for gene in self.nodes:
            random_vectors[gene] = np.random.rand(dim)
        return random_vectors
    
    def simple_imputation(self, strategy='mean'):
        node_vectors = self.gene2vec_dict.copy()
        all_genes = set(self.nodes)
        gene_list = list(all_genes)
        vectors = []

        for gene in gene_list:
            if gene in node_vectors:
                vectors.append(node_vectors[gene])
            else:
                vectors.append([np.nan] * len(next(iter(node_vectors.values()))))

        vectors = np.array(vectors)
        
        simple_imputer = SimpleImputer(strategy=strategy)
        imputed_vectors = simple_imputer.fit_transform(vectors)

        imputed_node_vectors = {gene: imputed_vectors[i] for i, gene in enumerate(gene_list)}

        return imputed_node_vectors

    def evaluate_simple_imputer(self, strategy='mean'):
        intersected_genes = list(set(self.gene2vec_dict.keys()) & set(self.nodes))
        sampled_genes = random.sample(intersected_genes, self.num_sampled_vectors)
        original_vectors = {gene: self.gene2vec_dict[gene] for gene in sampled_genes}

        gene_list = list(self.gene2vec_dict.keys())
        vectors = []
        for gene in gene_list:
            if gene in sampled_genes:
                vectors.append([np.nan] * len(next(iter(self.gene2vec_dict.values()))))
            else:
                vectors.append(self.gene2vec_dict[gene])
        vectors = np.array(vectors)

        simple_imputer = SimpleImputer(strategy=strategy)
        imputed_vectors = simple_imputer.fit_transform(vectors)
        imputed_vectors_dict = {gene: imputed_vectors[i] for i, gene in enumerate(gene_list) if gene in sampled_genes}

        comparisons = {}
        cosine_similarities = []
        euclidean_distances = []
        pearson_correlations = []
        for gene in sampled_genes:
            original_vector = original_vectors[gene]
            imputed_vector = imputed_vectors_dict[gene]
            cosine_similarity_value = cosine_similarity([original_vector], [imputed_vector])[0][0]
            cosine_similarities.append(cosine_similarity_value)
            euclidean_distance_value = np.linalg.norm(original_vector - imputed_vector)
            euclidean_distances.append(euclidean_distance_value)
            pearson_correlation_value = np.corrcoef(original_vector, imputed_vector)[0, 1]
            pearson_correlations.append(pearson_correlation_value)
            comparisons[gene] = (cosine_similarity_value, euclidean_distance_value, pearson_correlation_value)
        mean_cosine_similarity = np.mean(cosine_similarities)
        mean_euclidean_distance = np.mean(euclidean_distances)
        mean_pearson_correlation = np.mean(pearson_correlations)
        print(f'Mean cosine similarity for {strategy} vectors: {mean_cosine_similarity}')
        print(f'Mean Euclidean distance for {strategy} vectors: {mean_euclidean_distance}')
        print(f'Mean Pearson correlation for {strategy} vectors: {mean_pearson_correlation}')

        simple_cosine_similarity_df = pd.DataFrame({
            "Gene": sampled_genes,
            "Cosine Similarity": cosine_similarities
        })
        simple_euclidean_distance_df = pd.DataFrame({
            "Gene": sampled_genes,
            "Euclidean Distance": euclidean_distances
        })
        simple_pearson_correlation_df = pd.DataFrame({
            "Gene": sampled_genes,
            "Pearson Correlation": pearson_correlations
        })
        simple_cosine_similarity_df.to_csv(os.path.join(self.csv_folder, f'{strategy}_cosine_similarity.csv'), index=False)
        simple_euclidean_distance_df.to_csv(os.path.join(self.csv_folder, f'{strategy}_euclidean_distance.csv'), index=False)
        simple_pearson_correlation_df.to_csv(os.path.join(self.csv_folder, f'{strategy}_pearson_correlation.csv'), index=False)

        if strategy == 'mean':
            strategy = 'Mean'
        elif strategy == 'most_frequent':
            strategy = 'Most Frequent'
        # Visualize distributions of similarities - separate plots
        plt.figure()
        plt.hist(cosine_similarities, bins=50, color='purple', edgecolor='black', alpha=0.7, label=f'{strategy}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, f'{strategy}_cosine_similarity_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(euclidean_distances, bins=50, color='purple', edgecolor='black', alpha=0.7, label=f'{strategy}')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, f'{strategy}_euclidean_distance_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(pearson_correlations, bins=50, color='purple', edgecolor='black', alpha=0.7, label=f'{strategy}')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, f'{strategy}_pearson_correlation_distribution.png'))
        plt.close()

        generated_cosine_similarities = pd.read_csv(os.path.join(self.csv_folder, 'cosine_similarity.csv'))["Cosine Similarity"]
        generated_euclidean_distances = pd.read_csv(os.path.join(self.csv_folder, 'euclidean_distance.csv'))["Euclidean Distance"]
        generated_pearson_correlations = pd.read_csv(os.path.join(self.csv_folder, 'pearson_correlation.csv'))["Pearson Correlation"]

        plt.figure()
        plt.hist(generated_cosine_similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Generated')
        plt.hist(cosine_similarities, bins=50, color='purple', edgecolor='black', alpha=0.5, label=f'{strategy}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.evaluation_folder, f'combined_cosine_similarity_distribution_{strategy}.png'))
        plt.close()

        plt.figure()
        plt.hist(generated_euclidean_distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Generated')
        plt.hist(euclidean_distances, bins=50, color='purple', edgecolor='black', alpha=0.5, label=f'{strategy}')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.evaluation_folder, f'combined_euclidean_distance_distribution_{strategy}.png'))
        plt.close()

        plt.figure()
        plt.hist(generated_pearson_correlations, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Generated')
        plt.hist(pearson_correlations, bins=50, color='purple', edgecolor='black', alpha=0.5, label=f'{strategy}')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.evaluation_folder, f'combined_pearson_correlation_distribution_{strategy}.png'))
        plt.close()

        return comparisons, mean_cosine_similarity, mean_euclidean_distance, mean_pearson_correlation

    def evaluate_random_vectors(self):
        random_vectors = self.generate_random_vectors()
        intersected_genes = list(set(self.gene2vec_dict.keys()) & set(self.nodes))
        sampled_genes = random.sample(intersected_genes, self.num_sampled_vectors)
        comparisons = {}
        cosine_similarities = []
        euclidean_distances = []
        pearson_correlations = []
        for gene in sampled_genes:
            original_vector = self.gene2vec_dict[gene]
            random_vector = random_vectors[gene]
            cosine_similarity_value = cosine_similarity([original_vector], [random_vector])[0][0]
            cosine_similarities.append(cosine_similarity_value)
            euclidean_distance_value = np.linalg.norm(original_vector - random_vector)
            euclidean_distances.append(euclidean_distance_value)
            pearson_correlation_value = np.corrcoef(original_vector, random_vector)[0, 1]
            pearson_correlations.append(pearson_correlation_value)
            comparisons[gene] = (cosine_similarity_value, euclidean_distance_value, pearson_correlation_value)
        mean_cosine_similarity = np.mean(cosine_similarities)
        mean_euclidean_distance = np.mean(euclidean_distances)
        mean_pearson_correlation = np.mean(pearson_correlations)
        print(f'Mean cosine similarity for random vectors: {mean_cosine_similarity}')
        print(f'Mean Euclidean distance for random vectors: {mean_euclidean_distance}')
        print(f'Mean Pearson correlation for random vectors: {mean_pearson_correlation}')

        random_cosine_similarity_df = pd.DataFrame({
            "Gene": sampled_genes,
            "Cosine Similarity": cosine_similarities
        })
        random_euclidean_distance_df = pd.DataFrame({
            "Gene": sampled_genes,
            "Euclidean Distance": euclidean_distances
        })
        random_pearson_correlation_df = pd.DataFrame({
            "Gene": sampled_genes,
            "Pearson Correlation": pearson_correlations
        })
        random_cosine_similarity_df.to_csv(os.path.join(self.csv_folder, 'random_cosine_similarity.csv'), index=False)
        random_euclidean_distance_df.to_csv(os.path.join(self.csv_folder, 'random_euclidean_distance.csv'), index=False)
        random_pearson_correlation_df.to_csv(os.path.join(self.csv_folder, 'random_pearson_correlation.csv'), index=False)


        plt.figure()
        plt.hist(cosine_similarities, bins=50, color='orange', edgecolor='black', alpha=0.7, label='Random')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, 'random_cosine_similarity_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(euclidean_distances, bins=50, color='orange', edgecolor='black', alpha=0.7, label='Random')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, 'random_euclidean_distance_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(pearson_correlations, bins=50, color='orange', edgecolor='black', alpha=0.7, label='Random')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.evaluation_folder, 'random_pearson_correlation_distribution.png'))
        plt.close()

  
        generated_cosine_similarities = pd.read_csv(os.path.join(self.csv_folder, 'cosine_similarity.csv'))["Cosine Similarity"]
        generated_euclidean_distances = pd.read_csv(os.path.join(self.csv_folder, 'euclidean_distance.csv'))["Euclidean Distance"]
        generated_pearson_correlations = pd.read_csv(os.path.join(self.csv_folder, 'pearson_correlation.csv'))["Pearson Correlation"]

        plt.figure()
        plt.hist(generated_cosine_similarities, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Generated')
        plt.hist(cosine_similarities, bins=50, color='orange', edgecolor='black', alpha=0.5, label='Random')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.evaluation_folder, 'combined_cosine_similarity_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(generated_euclidean_distances, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Generated')
        plt.hist(euclidean_distances, bins=50, color='orange', edgecolor='black', alpha=0.5, label='Random')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.evaluation_folder, 'combined_euclidean_distance_distribution.png'))
        plt.close()

        plt.figure()
        plt.hist(generated_pearson_correlations, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Generated')
        plt.hist(pearson_correlations, bins=50, color='orange', edgecolor='black', alpha=0.5, label='Random')
        plt.xlabel('Pearson Correlation')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(self.evaluation_folder, 'combined_pearson_correlation_distribution.png'))
        plt.close()

        return comparisons, mean_cosine_similarity, mean_euclidean_distance, mean_pearson_correlation

    def graph_initialization(self):
        self.nodes = pd.concat([self.data['gene1'], self.data['gene2']]).unique()
        print(f"number of nodes: {len(self.nodes)}")
        self.node_to_id = {node: i for i, node in enumerate(self.nodes)}
        with open(os.path.join(self.pkl_folder, 'node_id.pkl'), 'wb') as file:
            pickle.dump(self.node_to_id, file)
        self.edges = [(self.node_to_id[gene1], self.node_to_id[gene2]) for gene1, gene2 in zip(self.data['gene1'], self.data['gene2'])]
        print(f'Number of edges: {len(self.edges)}')
        self.g = dgl.graph(self.edges)
        self.g = dgl.add_reverse_edges(self.g, 
                                       exclude_self=True,
                                       copy_edata=True)
        print(f'Number of edges after adding reverse: {self.g.num_edges()}')
        degrees = self.g.in_degrees()
        self.degrees_gene = {}
        for gene_name, gene_id in self.node_to_id.items():
             self.degrees_gene[gene_name] = int(degrees[gene_id])
        self.G = nx.from_pandas_edgelist(self.data, 'gene1', 'gene2')

    def graph_construction(self, resolution):
        node_vectors, _ = self.convex_generator(self.missing_genes, resolution=resolution)
        num_nodes = self.nodes.shape[0]
        features = np.zeros((num_nodes, 200), dtype=np.float32)
        for gene_name, gene_id in self.node_to_id.items():
            features[gene_id] = node_vectors.get(gene_name, np.zeros(200, dtype=np.float32)) 
        print(f'feature matrix dimensions: {features.shape}')
        rows_with_zeros = np.all(features == 0, axis=1)
        print(f'number of rows in features with zeros: {sum(rows_with_zeros)}')
        with open(os.path.join(self.pkl_folder, 'Gene2Vec_features.pkl'), 'wb') as file:
            pickle.dump(node_vectors, file)
        features = torch.tensor(features)
        self.g.ndata['feat'] = features
        self.g = dgl.remove_self_loop(self.g)
        n = len(self.edges)
        m = self.g.number_of_edges()  
        reverse_edges = torch.arange(n, m)
        self.g.edata['he'] = torch.arange(m)
        self.g.edata['he'] = torch.cat([self.g.edata['he'][:n], self.g.edata['he'][n:]])
        self.g = dgl.remove_edges(self.g, reverse_edges)
        print(self.g.number_of_edges())
        dgl.save_graphs(os.path.join(self.pkl_folder, 'graph.dgl'), self.g)
        
    def metrics(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes) 
        G.add_edges_from(self.edges) 
        sys.stdout = open(os.path.join(self.metrics_folder, 'metrics.txt'), "w")
        if self.g.is_homogeneous:
            print('The graph is Homogeneous')
        else:
            print('The graph is Heterogeneous')
        loops = list(nx.nodes_with_selfloops(G))
        if len(loops) > 0:
            print(f'The graph has {len(loops)} self-edges')
        else:
            print('The graph doesnt contain loops')
        degrees = self.g.in_degrees().float()
        if torch.sum(degrees == 0).item() == 0:
            print('All the nodes has at least one edge')
        else:
            print('The graph contain nodes with zero degree')
    #    adj_matrix = self.g.adjacency_matrix().to_dense()
    #    is_symmetric = np.allclose(adj_matrix, adj_matrix.T)
    #    if not is_symmetric:
    #        print("The graph is directed")
    #    else:
    #        print("The graph is undirected")
        print(f"Number of nodes: {self.g.number_of_nodes()}")
        print(f"Number of edges: {self.g.number_of_edges()}")
        print(f"Density: {nx.density(G)}")
        print(f"Number of connected components: {nx.number_connected_components(G)}")
        print(f"Average degree: {degrees.mean().item()}")
        print(f"Max node degree:  {torch.max(degrees)}")
        if nx.is_connected(G):
            print(f"Diameter: {nx.diameter(G)}")
            print(f"Average shortest path length: {nx.average_shortest_path_length(G)}")
        sys.stdout = sys.__stdout__
        sys.stdout.close()
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        plt.hist(degree_sequence, bins=50, color='skyblue', edgecolor='black', alpha=0.7, log=True)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xlabel("Degree")
        plt.ylabel("Frequency (log scale)")
        plt.savefig(os.path.join(self.plot_folder, 'degree_distr_plot.png'))   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("--path", type=str, required=True, help="path to the training data")
    parser.add_argument("--folder", type=str, required=True, help="folder to store the output")
    parser.add_argument("--num_sampled_vectors", type=int, default=100, help="number of sampled vectors for evaluation")
    args = parser.parse_args()
    pre = Preprocessing(data_path=args.path, folder_name=args.folder, num_sampled_vectors=args.num_sampled_vectors)
    pre.graph_initialization()
    pre.gene2vec_data()
    pre.missing_G2V_genes()
    best_resolution = pre.louvain_gridsearch()
    pre.convex_evaluation(resolution=best_resolution)
    pre.evaluate_random_vectors()
    strategies = ['mean', 'median', 'most_frequent']
    for strategy in strategies:
        pre.evaluate_simple_imputer(strategy=strategy)
    pre.graph_construction(resolution=best_resolution)
    pre.find_missing_bridges()
    pre.metrics()
