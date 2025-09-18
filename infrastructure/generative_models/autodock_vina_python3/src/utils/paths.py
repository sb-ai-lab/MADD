from pathlib import Path


def get_project_path() -> Path:
    """
    Creates path to project root dir
    :return: path(str) to project root dir
    """
    return Path(__file__).parent.parent.parent
