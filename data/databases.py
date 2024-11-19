"""Module containing information for input DBs."""

from enum import Enum


class StringDb(str, Enum):
    """Class for StringDB related constants."""

    PPI_URL = "https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz"
    PPI_FILENAME = "9606.protein.physical.links.v12.0.txt.gz"
    PPI_FILENAME_CSV = "9606.physical.links.csv"


class UcscDb(str, Enum):
    """Class for UCSC DB related constants."""

    PPI_URL = "http://hgdownload.soe.ucsc.edu/goldenPath/hgFixed/database/ggLink.txt.gz"
    PPI_FILENAME = "ggLink.txt.gz"
    PPI_FILENAME_CSV = "gg_ppi.csv"
