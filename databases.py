"""Module containing information for input DBs."""

from dataclasses import dataclass
from pydantic import FileUrl


@dataclass
class StringDb:
    """Class for StringDB related input."""

    PPI_DOWNLOADS_URL: str = "https://stringdb-downloads.org/download/protein.physical.links.v12.0"
    PPI_FILENAME: str = "9606.protein.physical.links.v12.0.txt.gz"
    PPI_URL: FileUrl = f"{PPI_DOWNLOADS_URL}/{PPI_FILENAME}"
    PPI_FILENAME_CSV: str = "9606.physical.links.csv"


@dataclass
class UcscDb:
    """Class for UCSC DB related input."""

    PPI_DOWNLOAD_URL: str = "http://hgdownload.soe.ucsc.edu/goldenPath/hgFixed/database"
    PPI_FILENAME: str = "ggLink.txt.gz"
    PPI_URL: FileUrl = f"{PPI_DOWNLOAD_URL}/{PPI_FILENAME}"
    PPI_FILENAME_CSV: str = "gg_ppi.csv"


@dataclass
class CorumDb:
    """Class for Corum DB related input."""

    # Older version URL. Access to version 5 only by API.
    # URL: https://mips.helmholtz-muenchen.de/corum/download/releases/current/humanComplexes.txt.zip
    COMPLEXES_FILENAME: str = "humanComplexes.txt"  # from humanComplexes.txt.zip file
    COMPLEXES_OUTPUT_CSV: str = "human_complexes.csv"


@dataclass
class NegatomeDb:
    """Class for Negatome DB related input."""

    # Accessible only via API and copy to a text file.
    # URL: http://mips.helmholtz-muenchen.de/proj/ppi/negatome/
    
    NEGATIVE_PPI_FILENAME: str = "negatome_2.csv"


@dataclass
class EnsemblDb:
    """Class for Ensembl GTF(Gene Tranfer File) for Homo Sapiens"""

    GTF_URL: FileUrl = (
        "http://ftp.ensembl.org/pub/release-104/gtf/homo_sapiens/Homo_sapiens.GRCh38.104.gtf.gz"
    )
    GTF_FILENAME: str = "Homo_sapiens.GRCh38.104.gtf.gz"


@dataclass
class Gene2Vec:
    """Class for Gene2Vec pre trained embeddings."""
    GIT_LINK: FileUrl = (
        "https://github.com/jingcheng-du/Gene2vec.git"
    )
    EMB_PATH = "pre_trained_emb/gene2vec_dim_200_iter_9.txt"
    FOLDER = "gene2vec"
    FILENAME = "gene2vec_embeddings.pkl"    

