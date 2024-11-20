"""Module containing information for input DBs."""

from enum import Enum
import pandas as pd 
from download import Download
import gzip


class Folder(str,Enum):
    """Class managing the output files"""


class StringDb(str, Enum):
    """Class for StringDB related constants."""
    URL = "https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz"
    FILENAME = "9606.protein.physical.links.v12.0.txt.gz"
    FILENAME_CSV = "string_db_interactions.csv"


    def extract(self):
        
        try:
            filename = Download(StringDb.FILENAME)
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                lines = [line.strip() for line in f]
            
            processed_lines = []
                
            for line in lines:
                if '.' in line:
                    elements = line.split()
                    elements[0] = elements[0].split('.')[1]
                    elements[1] = elements[1].split('.')[1]
                else:
                    elements = line.split()
                
                processed_lines.append(elements)
        except Exception as e:
            print(f"Error processing STRING data: {e}")

        DF = pd.DataFrame(processed_lines[1:], columns=processed_lines[0]) # type: ignore
        DF.to_csv('../data/string_db_interactions.csv', index=False)
        print("STRING data processed successfully.")

        return DF
    

class UcscDb(str, Enum):
    """Class for UCSC DB related constants."""

    URL = "http://hgdownload.soe.ucsc.edu/goldenPath/hgFixed/database/ggLink.txt.gz"
    FILENAME = "ggLink.txt.gz"
    FILENAME_CSV = "gg_ppi.csv"

    def extract(self):
        try:
            filename = Download(UcscDb.FILENAME)
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                lines = f.readlines()

            df_rows = []
            for line in lines:
                elements = line.strip().split("\t")
                if len(elements) == 10:
                    df_rows.append(elements)

            df = pd.DataFrame(df_rows, columns=["gene1", "gene2", "linkTypes", "pairCount", "oppCount", 
                                                "docCount", "dbList", "minResCount", "snippet", "context"])

            df = df.loc[~df['gene1'].str.isdigit() & ~df['gene2'].str.isdigit()]
            df = df[df['linkTypes'] == 'ppi']
            df.to_csv('gg_ppi.csv', index=False)
            print("UCSC data processed successfully.")

        except Exception as e:
            print(f"Error processing UCSC data: {e}")



class CorumDb(str, Enum):
    """Class for CORUM DB related constants."""

    #old version url, only access for version 5 by API.
    #URL = "https://mips.helmholtz-muenchen.de/corum/download/releases/current/humanComplexes.txt.zip"
    FILENAME = "humanComplexes.txt" # "humanComplexes.txt.zip"
    OUTPUT_CSV = "human_complexes.csv"

    def extract(self):
        try:
            # Read the extracted .txt file
            df = pd.read_csv(CorumDb.FILENAME, sep='\t')

            print('DataFrame shape:', df.shape)

            # Exclude rows with NaN values in 'subunits(Gene name)' or 'GO ID'
            df = df.dropna(subset=['subunits(Gene name)', 'GO ID'])

            # Remove duplicates
            df = df.drop_duplicates(subset=['ComplexName'])

            # Split the gene names and GO terms into lists
            df['subunits_gene_list'] = df['subunits(Gene name)'].apply(lambda x: x.split(';'))
            df['GO_terms_list'] = df['GO ID'].apply(lambda x: x.split(';'))

            # Add a column to count the number of gene members in each complex
            df['num_genes'] = df['subunits_gene_list'].apply(len)

            # Save to CSV
            df.to_csv('human_complexes.csv', index=False)
            print("Data saved to 'human_complexes.csv'.")

        except Exception as e:
            print(f"Error processing extracted file: {e}")


class NegatomeDb(str, Enum):
    """Class for Negatome DB related constants."""

    #only access via API and copy the data to a txt file
    #URL = "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/"
    FILENAME = "negatome_2.csv"