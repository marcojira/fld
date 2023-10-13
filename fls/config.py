import os
from pathlib import Path

current_file_path = Path(__file__).absolute()


try:
    from fls.default_config import Cfg
except ModuleNotFoundError:

    class Cfg:
        FEATURES_CACHE_PATH = os.path.join(
            current_file_path.parent, "features", "cache"
        )
