"""Module containing information for input DBs."""

import gzip
import logging
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from pydantic import FileUrl

from data.databases import CorumDb, NegatomeDb, StringDb, UcscDb

logger = logging.getLogger(__name__)


class Folder:
    """Class managing the output files"""


class Data:
    """Class for extracting DBs' data to the required format."""

    @staticmethod
    def download_file(file_url: FileUrl, output_filename: str) -> str:
        """Download file from a given URL."""
        if Path(output_filename).exists:
            logger.info(f"{output_filename!s} already exists.")
            return output_filename

        try:
            logger.info(f"Downloading from {file_url} to {output_filename}")
            urlretrieve(file_url, output_filename)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

        return output_filename

    @property
    def extract_from_string_db(self) -> pd.DataFrame:
        """Retrieve data from STRING DB and extract them in a CSV file."""
        try:
            filename = self.download_file(StringDb.PPI_URL, StringDb.PPI_FILENAME)
            with gzip.open(filename, "rt", encoding="utf-8") as f:
                lines = [line.strip() for line in f]

            processed_lines = []

            for line in lines:
                if "." in line:
                    elements = line.split()
                    elements[0] = elements[0].split(".")[1]
                    elements[1] = elements[1].split(".")[1]
                else:
                    elements = line.split()

                processed_lines.append(elements)
        except Exception as e:
            logger.error(f"While processing data from STRING DB: {e}")

        dataframe = pd.DataFrame(processed_lines[1:], columns=processed_lines[0])
        dataframe.to_csv("../data/string_db_interactions.csv", index=False)
        logger.info("STRING DB data processed successfully.")

        return dataframe

    @property
    def extract_from_ucsc_db(self) -> None:
        """Retrieve data from UCSC and extract them in a CSV file."""
        try:
            filename = self.download_file(UcscDb.PPI_URL, UcscDb.PPI_FILENAME)
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
            logger.info("UCSC data processed successfully.")

        except Exception as e:
            logger.error(f"EWhile processing data from UCSC: {e}")

    @property
    def extract_from_corum_db(self) -> None:
        """Extract data from Corum DB in a CSV file."""
        try:
            dataframe = pd.read_csv(CorumDb.FILENAME, sep="\t")
            logger.info("DataFrame shape:", dataframe.shape)

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

            dataframe.to_csv(CorumDb.OUTPUT_CSV, index=False)
            logger.info(f"Data saved to {CorumDb.OUTPUT_CSV}.")

        except Exception as e:
            logger.error(f"While processing extracted file: {e}")

    @property
    def extract_from_negatome_db(self):
        """Extract data from Negatome DB in a CSV file."""
        return None
