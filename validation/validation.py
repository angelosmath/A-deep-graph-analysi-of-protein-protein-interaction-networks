import pandas as pd
import random
import pickle

class Validation_Set:
    def __init__(self):
        
        ucsc_path = '/home/angelosmath/MSc/thesis_ppi_mean/data/UCSC/gg_ppi.csv'
        string_path = '/home/angelosmath/MSc/thesis_ppi_mean/mapping/STRING_gene.csv'
        
        self.string_df = pd.read_csv(string_path, low_memory=False)
        self.ucsc_df = pd.read_csv(ucsc_path, low_memory=False)
        
        self.n = 100
        self.score = 400
        
        self.filename = '1'
    
    def create_samples(self):
        
        #=========== unique genes 
        ucsc_genes = set(self.ucsc_df['gene1'].unique()).union(self.ucsc_df['gene2'].unique())
        string_genes = set(self.string_df['gene1'].unique()).union(self.string_df['gene2'].unique())
        print(f'UCSC unique Genes: {len(list(string_genes))}')
        print(f'String unique Genes: {len(list(ucsc_genes))}')

        common_genes = list(string_genes.intersection(ucsc_genes)) #Common genes: 14648
        print(f'Common genes: {len(common_genes)}')
        
        #=========== unique pairs
        
        string_gene_pairs = set(zip(self.string_df['gene1'], self.string_df['gene2']))
        ucsc_gene_pairs = set(zip(self.ucsc_df['gene1'], self.ucsc_df['gene2']))

        common_pairs = string_gene_pairs.intersection(ucsc_gene_pairs)

        #common_pairs_df = pd.DataFrame(list(common_pairs), columns=['gene1', 'gene2'])

        print(f'Common pairs of genes: {len(common_pairs)}')
        
        
        #=========== positive samples
        
        # Find gene pairs that exist in STRING_df but not in UCSC_df
        unique_pairs_in_string = string_gene_pairs - ucsc_gene_pairs

        #print("Number of gene pairs unique to STRING_df:", len(unique_pairs_in_string))

        unique_string_df = pd.DataFrame(list(unique_pairs_in_string), columns=['gene1', 'gene2'])

        unique_string_df = self.string_df[self.string_df[['gene1', 'gene2']].apply(tuple, axis=1).isin(unique_string_df.apply(tuple, axis=1))]

        threshold_positive = unique_string_df[unique_string_df['combined_score'] > self.score]

        print(f'found {threshold_positive.shape[0]} gene paris with score greater than {self.score}')
        
        positive_samples = threshold_positive[['gene1','gene2']].sample(self.n)
        
        #=========== negative samples
        
        negative_samples = set()
        
        while len(negative_samples) < self.n:
            
            gene_pair = (random.choice(list(common_genes)), random.choice(list(common_genes)))
    
            if gene_pair not in common_pairs and gene_pair not in negative_samples:
                negative_samples.add(gene_pair)
            
        negative_samples = pd.DataFrame(list(negative_samples), columns=['gene1', 'gene2'])
        
        positive_samples.to_csv(f'positive_samples_{self.filename}.csv',index = False)
        negative_samples.to_csv(f'negative_samples_{self.filename}.csv',index = False)
    
        return positive_samples, negative_samples

    def create_graphs(self,pos_samples,neg_samples):

        gene2vec_path = '/home/angelosmath/MSc/thesis_ppi_mean/preprocessing/final_final/Gene2Vec_features.pkl'
        
        with open(gene2vec_path, 'rb') as file:
            Gene2Vec_dict = pickle.load(file)

        node_to_id_path = '/home/angelosmath/MSc/thesis_ppi_mean/preprocessing/final_final/node_id.pkl'
        
        with open(node_to_id_path, 'rb') as file:
            node_to_id = pickle.load(file)

        pos_nodes = pd.concat([pos_samples['gene1'], pos_samples['gene2']]).unique()
        
        neg_nodes = pd.concat([neg_samples['gene1'], neg_samples['gene2']]).unique()
        print(len(list(node_to_id.keys())))
        
        print(len(set(list(node_to_id.keys())).intersection(set(pos_nodes))))
        print(len(pos_nodes))
        

        
        
def main():
    v = Validation_Set()
    positive_samples,negative_samples = v.create_samples()
    v.create_graphs(positive_samples,negative_samples)

if __name__ == "__main__":
    main()

        
        
