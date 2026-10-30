from pathlib import Path

from .DFLJPG import DFLJPG

class DFLIMG():

    @staticmethod
    def load(filepath, loader_func=None):
        if filepath is None:
            return None

        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        suffix = filepath.suffix.lower()
        if suffix in ('.jpg', '.jpeg'):
            return DFLJPG.load(str(filepath), loader_func=loader_func)

        return None
