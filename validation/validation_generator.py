import pandas as pd
import random
import pickle
import argparse
import numpy as np
import torch
import dgl
import os

class Validation_Set:


    def __init__(self,args):

        ucsc_path = args.ucsc #'/home/angelosmath/MSc/thesis_ppi_mean/data/UCSC/gg_ppi.csv'
        string_path = args.string #'/home/angelosmath/MSc/thesis_ppi_mean/mapping/STRING_gene.csv'
        
        self.string_df = pd.read_csv(string_path, low_memory=False)
        self.ucsc_df = pd.read_csv(ucsc_path, low_memory=False)

        self.node_vectors_path = args.features #'/home/angelosmath/MSc/thesis_ppi_mean/preprocessing/final_final/Gene2Vec_features.pkl'
        self.node_to_id_path = args.mapping #'/home/angelosmath/MSc/thesis_ppi_mean/preprocessing/final_final/node_id.pkl'
        
        self.n = 100
        self.score = args.score

        self.folder_name = args.output

        try:
            os.mkdir(self.folder_name)
        except Exception as e:
                print(e)
    
    def create_samples(self):

        # Unique genes 
        ucsc_genes = set(self.ucsc_df['gene1'].unique()).union(self.ucsc_df['gene2'].unique())
        string_genes = set(self.string_df['gene1'].unique()).union(self.string_df['gene2'].unique())
        print(f'String unique Genes: {len(list(string_genes))}')
        print(f'UCSC unique Genes: {len(list(ucsc_genes))}')
        
        # Common genes
        common_genes = list(string_genes.intersection(ucsc_genes))
        print(f'Common genes: {len(common_genes)}')
        
        # Unique pairs
        string_gene_pairs = set(zip(self.string_df['gene1'], self.string_df['gene2']))
        ucsc_gene_pairs = self.ucsc_df[['gene1', 'gene2']].drop_duplicates()
        ucsc_gene_pairs.set_index(['gene1', 'gene2'], inplace=True)
        self.string_df.set_index(['gene1', 'gene2'], inplace=True)

        ucsc_gene_pairs_set = set(zip(ucsc_gene_pairs.index.get_level_values(0), ucsc_gene_pairs.index.get_level_values(1)))

        # Positive samples
        filtered_string_df = self.string_df[
            (self.string_df.index.get_level_values(0).isin(common_genes)) &
            (self.string_df.index.get_level_values(1).isin(common_genes))
        ]

        filtered_string_df = filtered_string_df[~filtered_string_df.index.isin(ucsc_gene_pairs_set)]
        threshold_positive = filtered_string_df[filtered_string_df['combined_score'] > self.score]

        if threshold_positive.empty:
            print('No positive samples found with the given score threshold.')
            raise f"Choose a score below {self.score}"
        else:
            threshold_positive.reset_index(inplace=True)
            min_score = threshold_positive['combined_score'].min()
            max_score = threshold_positive['combined_score'].max()
            print(f"Range of scores in threshold_positive: {min_score} to {max_score}")
            positive_samples = threshold_positive[['gene1', 'gene2']]
        
        print(f'positive samples found {threshold_positive.shape[0]}')
        print('positive samples processing done!')
        
        pos_genes = list(pd.concat([positive_samples['gene1'],positive_samples['gene2']]).unique())
        # Negative samples
        negative_samples = set()
        while len(negative_samples) <= positive_samples.shape[0] * 2:
            gene_pair = (random.choice(pos_genes), random.choice(pos_genes))
            if gene_pair not in string_gene_pairs and gene_pair not in ucsc_gene_pairs and gene_pair not in ucsc_gene_pairs_set and gene_pair not in negative_samples:
                negative_samples.add(gene_pair)
        negative_samples = pd.DataFrame(list(negative_samples), columns=['gene1', 'gene2'])

        print('negative samples processing done!')

        # Save to CSV
        positive_samples.to_csv(self.folder_name + '/' +'positive_samples.csv', index=False)
        negative_samples.to_csv(self.folder_name + '/' +'negative_samples.csv', index=False)
    
        return positive_samples, negative_samples

    def create_graphs(self, pos_samples, neg_samples):
        
        with open(self.node_vectors_path, 'rb') as file:
            node_vectors = pickle.load(file)


        with open(self.node_to_id_path, 'rb') as file:
            node_to_id = pickle.load(file)

        ucsc_genes = set(self.ucsc_df['gene1'].unique()).union(self.ucsc_df['gene2'].unique())

        pos_nodes = pd.concat([pos_samples['gene1'], pos_samples['gene2']]).unique()
        neg_nodes = pd.concat([neg_samples['gene1'], neg_samples['gene2']]).unique()

        pos_edges = [(node_to_id[gene1], node_to_id[gene2]) for gene1, gene2 in zip(pos_samples['gene1'], pos_samples['gene2'])]
        neg_edges = [(node_to_id[gene1], node_to_id[gene2]) for gene1, gene2 in zip(neg_samples['gene1'], neg_samples['gene2'])]

        print(f'positive edges: {len(pos_edges)}\n'
              f'negative edges: {len(neg_edges)}')
        print('edges intercection:',set(pos_edges).intersection(set(neg_edges)),'|must be an empty set!|')
        print(f'positive nodes: {len(pos_nodes)}\n'
              f'negative nodes: {len(neg_nodes)}')
        print(f'nodes intercection: {len(set(pos_nodes).intersection(set(neg_nodes)))}')


        
        pos_g = dgl.graph(pos_edges,num_nodes=len(ucsc_genes))
        #pos_g = dgl.add_reverse_edges(pos_g)


        #positive graph
        num_nodes = len(list(node_to_id.keys()))
        pos_features = np.zeros((num_nodes, 200), dtype=np.float32)
        for gene_name, gene_id in node_to_id.items():
            pos_features[gene_id] = node_vectors.get(gene_name,np.zeros(200, dtype=np.float32)) 

        pos_features = torch.tensor(pos_features)
        pos_g.ndata['feat'] = pos_features

        print(f'positive graph number of nodes: {pos_g.number_of_edges()}')

        dgl.save_graphs(self.folder_name + '/' + 'pos_graph.dgl', pos_g)
        print('positive graph done!')
        #print('positive graph:','\n',pos_g)

        #negative graph
        neg_g = dgl.graph(neg_edges,num_nodes=len(ucsc_genes))
        #neg_g = dgl.add_reverse_edges(neg_g)

        num_nodes = len(list(node_to_id.keys()))
        neg_features = np.zeros((num_nodes, 200), dtype=np.float32)
        for gene_name, gene_id in node_to_id.items():
            neg_features[gene_id] = node_vectors.get(gene_name,np.zeros(200, dtype=np.float32)) 

        neg_features = torch.tensor(pos_features)
        neg_g.ndata['feat'] = neg_features

        dgl.save_graphs(self.folder_name + '/' + 'neg_graph.dgl', neg_g)

        print('negative graph done!')
        #print('negative graph:','\n',neg_g)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucsc", type=str)
    parser.add_argument("--string", type=str)
    parser.add_argument("--features", type=str)
    parser.add_argument("--mapping", type=str)
    parser.add_argument(
        "--score",
        type=int,
        default=600,
        help="STRING database ppi score to use as a threshold",
    ) 

    parser.add_argument("--output", type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    v = Validation_Set(args)
    positive_samples, negative_samples = v.create_samples()
    v.create_graphs(positive_samples, negative_samples)

if __name__ == "__main__":
    main()
