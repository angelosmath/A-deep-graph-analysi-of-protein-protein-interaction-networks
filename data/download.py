"""Contains functionality to download input data."""

import gzip
import logging
import os
import pandas as pd
import zipfile
from io import BytesIO
from urllib.request import urlretrieve

from alive_progress import alive_bar
from databases import StringDb, UcscDb, CorumDb, NegatomeDb
from pydantic import BaseModel, FilePath, FileUrl, computed_field

logger = logging.getLogger(__name__)


class Download(BaseModel):
    """Class of functionalities related to downloading input data."""

    url: FileUrl
    output_filename: FilePath

    @property
    @computed_field
    def download_file(self) -> FilePath:
        """Download file from a given URL."""
        if os.path.exists(self.output_filename):
            logger.info(f"{self.output_filename!s} already exists.")
            return self.output_filename

        try:
            logger.info(f"Downloading from {self.url} to {self.output_filename}")
            urlretrieve(self.url, self.output_filename)
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise

        return self.output_filename