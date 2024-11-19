"""Contains functionality to download input data."""

import gzip
import logging
from urllib.request import urlretrieve

import pandas as pd
from pydantic import BaseModel, FilePath, FileUrl, computed_field

logger = logging.getLogger(__name__)

class DownloadInputData(BaseModel):
    """Class of functionalities related to downloading input data."""

    url: FileUrl
    output_filename: FilePath
    database_name: str

    @property
    @computed_field
    def download_ppi_file_from_url(self) -> FilePath:
        """Download PPI file from a given URL."""

        if self.output_filename.exists():
            logger.info(f"{self.output_filename!s} already exists.")
            return self.output_filename

        try:
            logger.info(f"Retrieving file from {self.url}")
            urlretrieve(self.url, self.output_filename)
        except Exception as error:
            raise ValueError(error)

        return self.output_filename
