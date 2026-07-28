import shutil
import os

def copy_file(source_path: str, destination_path: str) -> None:
    """Copy a file from source_path to destination_path

    Args:
        source_path (str): source path
        destination_path (str): destination path
    """    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return