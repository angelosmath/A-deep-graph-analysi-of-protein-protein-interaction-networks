from pathlib import Path
import inspect
from pydantic import DirectoryPath
import pickle

class Folder:
    """Class managing the output files."""

    base_path: DirectoryPath = Path.cwd()
    dga_folder: DirectoryPath = base_path.parent / "dga_analysis" #base_path.parent.parent / "dga_analysis"

    # @staticmethod
    def _ensure_dir(path: Path) -> None:
        """Ensure the necessary directory exists."""

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)


    def _get_caller_class():
        """Determine the class name of the calling method"""

        stack = inspect.stack()
        for frame in stack:
            if 'self' in frame.frame.f_locals:
                frame.frame.f_locals['self'].__class__.__name__
                return frame.frame.f_locals['self'].__class__.__name__

        return None

    @staticmethod
    def get_path(filename: str) -> Path:
        """Generates the full path for a file based on the caller class."""
        
        caller_class = Folder._get_caller_class()
        if caller_class == "Data":
            folder = Folder.dga_folder / "data"
        else:
            return  ValueError("Caller class is not recognized.")
        
        Folder._ensure_dir(folder)

        return folder / filename
    
    @staticmethod
    def save_dataframe(dataframe, filename: str, index: bool = False) -> Path:
        """Save a DataFrame to a CSV file in the appropriate folder."""

        file_path = Folder.get_path(filename) 
        dataframe.to_csv(file_path, index=index)
        # print(f"Data saved to {file_path}")
        return file_path

    def list_files(self, caller: str) -> list:
        """List all files in a speciffic folder"""

        folder = self.data_folder if caller == "data" else self.other_folder
        return [file for file in folder.iterdir() if file.is_file()]
