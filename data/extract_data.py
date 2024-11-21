"""Module containing functionalities for extracting input data from DBs."""

import gzip
from typing import ClassVar
from urllib.request import urlretrieve

import pandas as pd
from pydantic import BaseModel, FileUrl

from data.databases import CorumDb, NegatomeDb, StringDb, UcscDb, EnsemblDb


class Folder():
    """Class managing the output files"""





class Data(BaseModel):
    """Class for extracting DBs' data to the required format."""

    string_db: ClassVar = StringDb
    ucsc_db: ClassVar = UcscDb
    corum_db: ClassVar = CorumDb
    negatome_db: ClassVar = NegatomeDb
    ensembl_db: ClassVar = EnsemblDb

    @staticmethod
    def download_file(file_url: FileUrl, output_filename: str) -> str:
        """Download file from a given URL."""
        try:
            print(f"Downloading from {file_url} to {output_filename}")
            urlretrieve(str(file_url), output_filename)
        except Exception as e:
            raise Exception(f"Error downloading file: {e}")

        return output_filename
    
    @property
    def gtf_parsing(self) -> None:
        """Retrieve GTF(Gene Tanfer File) for Homo Sapiens from Ensembl DB and extract them in a CSV file."""   
        try:
            filename = self.download_file(self.ensembl_db.PPI_URL, self.ensembl_db.PPI_FILENAME)
            
            keys = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame']
            with gzip.open(self.filename, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
                ens_dict = {key: [None] * len(lines) for key in keys}

                for i in range(len(lines)):
                    line = lines[i]
                    if not line.startswith('#'):
                        parts = line.strip().split('\t')
                        attributes = parts[8].split('; ')
                        for index, key in enumerate(keys):
                            ens_dict[key][i] = parts[index]
                        for attr in attributes:
                            key, value = attr.split(' ', 1)
                            key = key.strip('"')
                            value = value.strip('"')
                            if key not in ens_dict:
                                ens_dict[key] = [None] * len(lines)
                            ens_dict[key][i] = value

            raw_dataframe = pd.DataFrame(ens_dict)
            # must be 5 its the file readme  
            none_rows = raw_dataframe[raw_dataframe.isna().all(axis=1)]
            #print(f"Lines with only None values: {none_rows.shape[0]}")

            dataframe = raw_dataframe.dropna(how='all')
            dataframe.to_csv('ensembl_gtf_homosapiens.csv',index = False)
            print('GTF file parsed successfully')
        
            return dataframe
        
        except Exception as e:
            print(f"[ERROR]: While processing data from Ensembl DB: {e}")
    
    @property
    def extract_from_string_db(self) -> pd.DataFrame:
        """Retrieve data from STRING DB and extract them in a CSV file."""
        processed_lines = []

        try:
            filename = self.download_file(self.string_db.PPI_URL, self.string_db.PPI_FILENAME)
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                lines = [line.strip() for line in f]

            for line in lines:
                if "." in line:
                    elements = line.split()
                    elements[0] = elements[0].split(".")[1]
                    elements[1] = elements[1].split(".")[1]
                else:
                    elements = line.split()

                processed_lines.append(elements)
        except Exception as e:
            print(f"[ERROR]: While processing data from STRING DB: {e}")
        
        if not processed_lines:
            raise ValueError(f"No lines were selected from {self.string_db.PPI_FILENAME}.")

        dataframe = pd.DataFrame(processed_lines[1:], columns=processed_lines[0])
        dataframe.to_csv("data/string_db_ID_interactions.csv", index=False)
        
        # mapping process
        gtf_dataframe = self.gtf_parsing()
        gtf_dataframe = gtf_dataframe[['protein_id', 'gene_name']]
        gtf_dataframe = gtf_dataframe.drop_duplicates()
        gtf_dataframe = gtf_dataframe.dropna()
        gtf_dataframe = gtf_dataframe.reset_index(drop=True)

        unique_list = set(dataframe['protein1']).union(dataframe['protein2'])
        #print(f'Unique Ensembl protein IDs: {len(unique_list)}')

        mapped_dataframe = pd.merge(dataframe, gtf_dataframe, left_on='protein1', right_on='protein_id', how='left')
        dataframe = pd.merge(mapped_dataframe, gtf_dataframe, left_on='protein2', right_on='protein_id', how='left', suffixes=('_1', '_2'))
        dataframe = dataframe[['gene_name_1', 'gene_name_2', 'combined_score']]
        dataframe.columns = ['gene1','gene2','combined_score']
        
        print("STRING DB data processed successfully.")

        return dataframe

    @property
    def extract_from_ucsc_db(self) -> None:
        """Retrieve data from UCSC and extract them in a CSV file."""
        try:
            filename = self.download_file(self.ucsc_db.PPI_URL, self.ucsc_db.PPI_FILENAME)
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                lines = f.readlines()

            dataframe_rows = []
            for line in lines:
                elements = line.strip().split("\t")
                if len(elements) == 10:
                    dataframe_rows.append(elements)

            dataframe = pd.DataFrame(
                dataframe_rows,
                columns=[
                    "gene1",
                    "gene2",
                    "linkTypes",
                    "pairCount",
                    "oppCount",
                    "docCount",
                    "dbList",
                    "minResCount",
                    "snippet",
                    "context",
                ],
            )

            dataframe = dataframe.loc[
                ~dataframe["gene1"].str.isdigit() & ~dataframe["gene2"].str.isdigit()
            ]
            dataframe = dataframe[dataframe["linkTypes"] == "ppi"]
            dataframe.to_csv("gg_ppi.csv", index=False)
            print("UCSC data processed successfully.")
        
            return dataframe

        except Exception as e:
            print(f"[ERROR]: While processing data from UCSC: {e}")

    @property
    def extract_from_corum_db(self) -> None:
        """Extract data from Corum DB in a CSV file."""
        try:
            dataframe = pd.read_csv(self.corum_db.FILENAME, sep="\t")
            print("DataFrame shape:", dataframe.shape)

            # Drop rows with NaN values in 'subunits(Gene name)' or 'GO ID'
            dataframe = dataframe.dropna(subset=["subunits(Gene name)", "GO ID"])
            dataframe = dataframe.drop_duplicates(subset=["ComplexName"])

            # Split the gene names and GO terms into lists
            dataframe["subunits_gene_list"] = dataframe["subunits(Gene name)"].apply(
                lambda x: x.split(";")
            )
            dataframe["GO_terms_list"] = dataframe["GO ID"].apply(
                lambda x: x.split(";")
            )

            # Add a column to count the number of gene members in each complex
            dataframe["num_genes"] = dataframe["subunits_gene_list"].apply(len)

            dataframe.to_csv(self.corum_db.OUTPUT_CSV, index=False)
            print(f"Data saved to {self.corum_db.OUTPUT_CSV}.")

            return dataframe

        except Exception as e:
            print(f"[ERROR]: While processing extracted file: {e}")

    @property
    def extract_from_negatome_db(self):
        """Extract data from Negatome DB in a CSV file."""
        
        return None
