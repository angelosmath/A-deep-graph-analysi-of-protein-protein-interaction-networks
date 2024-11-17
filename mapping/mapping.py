import pandas as pd

class Mapping:
    def __init__(self):
        self.mapping_path = '/work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/gtf_parsing/Homo_sapiens_df.csv'
        self.ppi_path = '/work/ke74dex/DGL/A-deep-graph-analysi-of-protein-protein-interaction-networks/data/string/9606.physical.links.csv'
    
    def mapping_dataframe(self):
        try:
            mapping_df = pd.read_csv(self.mapping_path, dtype={'protein_id': str, 'gene_name': str}, low_memory=False)
            ppi_df = pd.read_csv(self.ppi_path, dtype={'protein1': str, 'protein2': str, 'combined_score': int}, low_memory=False)
        except pd.errors.DtypeWarning as e:
            print(f"DtypeWarning: {e}")
        
        unique_list = set(ppi_df['protein1']).union(ppi_df['protein2'])
        print(f'Unique Ensembl protein IDs: {len(unique_list)}')
        
        # mapping file preprocessing
        mapping_df = mapping_df[['protein_id', 'gene_name']]
        mapping_df = mapping_df.drop_duplicates()
        mapping_df = mapping_df.dropna()
        mapping_df = mapping_df.reset_index(drop=True)

        # mapping protein id --> gene name 
        mapped_data = pd.merge(ppi_df, mapping_df, left_on='protein1', right_on='protein_id', how='left')
        gene_data = pd.merge(mapped_data, mapping_df, left_on='protein2', right_on='protein_id', how='left', suffixes=('_1', '_2'))
        gene_data = gene_data[['gene_name_1', 'gene_name_2', 'combined_score']]
        gene_data.columns = ['gene1','gene2','combined_score']
        
        
        mapped_IDs = set(gene_data['gene1']).union(gene_data['gene2'])
        
        print(f'Mapped Ensembl protein IDs: {len(mapped_IDs)}')

        gene_data = gene_data.dropna()
        
        gene_data.to_csv('STRING_gene.csv',index = False)

def main():
    m = Mapping()
    m.mapping_dataframe()

if __name__ == "__main__":
    main()
