"""Module containing functionalities for extracting input data from DBs."""

import gzip
import os
import pickle
import subprocess
from typing import ClassVar
from urllib.request import urlretrieve

import pandas as pd
import numpy as np
from pydantic import BaseModel, FileUrl
from pathlib import Path

from databases import CorumDb, EnsemblDb, NegatomeDb, StringDb, UcscDb, Gene2Vec

from file_manager import Folder



class Data(BaseModel):
    """Class for extracting processing DBs' data to the required format."""

    string_db: ClassVar = StringDb
    ucsc_db: ClassVar = UcscDb
    corum_db: ClassVar = CorumDb
    negatome_db: ClassVar = NegatomeDb
    ensembl_db: ClassVar = EnsemblDb
    gene2vec_gh: ClassVar = Gene2Vec
    folder_manager: ClassVar = Folder  # save using this: self.folder_manager.save_dataframe(dataframe, "string_db_results.csv")

    @staticmethod
    def download_file(file_url: FileUrl, output_filename: str) -> str:
        """Download file from a given URL."""
        target_path = Folder.get_path(output_filename)

        if target_path.is_file():
            print(f"{target_path} already exists.")
            return target_path

        try:
            print(f"Downloading from {file_url} to {target_path}")
            Folder._ensure_dir(target_path.parent)  # Ensure the directory exists
            urlretrieve(file_url, target_path)
        except Exception as e:
            raise Exception(f"Error downloading file: {e}")

        return target_path

    def gtf_parsing(self) -> pd.DataFrame:
        """Retrieve GTF(Gene Tanfer File) for Homo Sapiens from Ensembl DB and extract them in a CSV file."""
        try:
            filename = self.download_file(self.ensembl_db.GTF_URL, self.ensembl_db.GTF_FILENAME)

            keys = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame"]
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                lines = f.readlines()
                ens_dict = {key: [None] * len(lines) for key in keys}

                for i in range(len(lines)):
                    line = lines[i]
                    if not line.startswith("#"):
                        parts = line.strip().split("\t")
                        attributes = parts[8].split("; ")
                        for index, key in enumerate(keys):
                            ens_dict[key][i] = parts[index]
                        for attr in attributes:
                            key, value = attr.split(" ", 1)
                            key = key.strip('"')
                            value = value.strip('"')
                            if key not in ens_dict:
                                ens_dict[key] = [None] * len(lines)
                            ens_dict[key][i] = value

            raw_dataframe = pd.DataFrame(ens_dict)
            # must be 5 its the file readme
            none_rows = raw_dataframe[raw_dataframe.isna().all(axis=1)]
            # print(f"Lines with only None values: {none_rows.shape[0]}")

            if "protein_id" not in raw_dataframe.columns or "gene_name" not in raw_dataframe.columns:
                print(f"[ERROR]: Missing expected columns in GTF DataFrame")
                return pd.DataFrame()
            
            dataframe = raw_dataframe.dropna(how="all")
            self.folder_manager.save_dataframe(dataframe, "ensembl_gtf_homosapiens.csv", index=False)
            print("GTF file parsed successfully")

            return dataframe

        except Exception as e:
            print(f"[ERROR]: While processing data from Ensembl DB: {e}")

    def extract_from_string_db(self) -> pd.DataFrame:
        """Retrieve data from STRING DB and extract them in a CSV file."""
        #add mapping in the description
        
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
        #dataframe.to_csv("data/string_db_ID_interactions.csv", index=False)
        self.folder_manager.save_dataframe(dataframe, "string_db_raw.csv")

        # mapping process
        gtf_dataframe = self.gtf_parsing()

        if gtf_dataframe.empty: 
            print("[ERROR]: GTF parsing failed, cannot proceed with mapping.")
            return pd.DataFrame()
        
        gtf_dataframe = gtf_dataframe[["protein_id", "gene_name"]]
        gtf_dataframe = gtf_dataframe.drop_duplicates()
        gtf_dataframe = gtf_dataframe.dropna()
        gtf_dataframe = gtf_dataframe.reset_index(drop=True)

        unique_list = set(dataframe["protein1"]).union(dataframe["protein2"])
        # print(f'Unique Ensembl protein IDs: {len(unique_list)}')

        mapped_dataframe = pd.merge(
            dataframe, gtf_dataframe, left_on="protein1", right_on="protein_id", how="left"
        )
        dataframe = pd.merge(
            mapped_dataframe,
            gtf_dataframe,
            left_on="protein2",
            right_on="protein_id",
            how="left",
            suffixes=("_1", "_2"),
        )
        dataframe = dataframe[["gene_name_1", "gene_name_2", "combined_score"]]
        dataframe.columns = ["gene1", "gene2", "combined_score"]
        self.folder_manager.save_dataframe(dataframe, filename = "string_db.csv")
        print("STRING DB data processed successfully.")

        return dataframe

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
            self.folder_manager.save_dataframe(dataframe, "ucsc_db.csv", index=False)
            print("UCSC data processed successfully.")

            return dataframe

        except Exception as e:
            print(f"[ERROR]: While processing data from UCSC: {e}")

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
            dataframe["GO_terms_list"] = dataframe["GO ID"].apply(lambda x: x.split(";"))

            # Add a column to count the number of gene members in each complex
            dataframe["num_genes"] = dataframe["subunits_gene_list"].apply(len)

            dataframe.to_csv(self.corum_db.OUTPUT_CSV, index=False)
            print(f"Data saved to {self.corum_db.OUTPUT_CSV}.")

            return dataframe

        except Exception as e:
            print(f"[ERROR]: While processing extracted file: {e}")

    def extract_from_negatome_db(self):
        """Extract data from Negatome DB in a CSV file."""

        #MAPPING!

        return None
    
    def extract_from_Gene2Vec_gh(self) -> None:
        """Extract pre trained embeddings from Gene2Vec GitHub."""

        try:
            git_url = self.gene2vec_gh.GIT_LINK
            embeddings_file = self.gene2vec_gh.EMB_PATH
            repo_folder = Folder.get_path(self.gene2vec_gh.FOLDER)
            output_pkl = self.gene2vec_gh.FILENAME

            if not repo_folder.exists():
                try:
                    print(f"Using Git URL: {git_url}")
                    print(f"Running: git clone {git_url} {repo_folder}")
                    subprocess.run(["git", "clone", git_url, str(repo_folder)], check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception(f"Error cloning repository: {e}")
            else:
                print(f"Gene2Vec repository already exists at {repo_folder}")

            embedding_file_path = repo_folder / embeddings_file
            if not embedding_file_path.exists():
                raise FileNotFoundError(f"Embedding file not found at {embedding_file_path}. Check the repository structure.")

            gene2vec_dict = {}
            with open(embedding_file_path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    gene_name = parts[0]
                    values = np.array(parts[1:], dtype=np.float32)
                    gene2vec_dict[gene_name] = values

            pickle_path = Folder.get_path(output_pkl)
            with open(pickle_path, "wb") as pkl_file:
                pickle.dump(gene2vec_dict, pkl_file)

            print(f"Gene2Vec embeddings saved as a pickle file to {pickle_path}")

        except Exception as e:
            print(f"[ERROR]: While processing Gene2Vec data: {e}")



def main():
    data = Data()
    #data.extract_from_string_db()
    #data.extract_from_ucsc_db()
    data.extract_from_Gene2Vec_gh()


if __name__ == "__main__":
    main()

