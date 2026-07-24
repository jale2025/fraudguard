import os
from pathlib import Path
import shutil
from dataclasses import dataclass


def move_file(source_path: str, destination_path: str) -> None:
    """Move a file from source_path to destination_path

    Args:
        source_path (str): source path
        destination_path (str): destination path
    """    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.move(source_path, destination_path)


def data_available(dir_path: str | Path = os.environ["DATA_DIR_INCOMING"]) -> bool: 
    """
    This function checks a file does exist in the corresponding folder.

    Returns:
        bool: A file does exist (true) or not (false).
    """

    # Create a path by the string
    dir_path = Path(dir_path)

    # Check the pathn does exist and it's a path
    if not dir_path.exists() or not dir_path.is_dir():
        return False

    # Check a file does exist in the folder
    return any(item.is_file() for item in dir_path.iterdir())

@dataclass
class FileLists:
    valid_list: list[str]
    invalid_list: list[str]

def validate_data_files(dir_path: str | Path = os.environ["DATA_DIR_INCOMING"]) -> FileLists:

    # Iterate through all data files
    for item in dir_path.iterdir():

        





def valid_data_file() -> bool:





