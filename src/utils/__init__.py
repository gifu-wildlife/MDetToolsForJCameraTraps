import pathlib
import sys
import re
from utils.tag import BaseTag

utils = pathlib.Path(__file__).parent
if not str(utils) in sys.path:
    sys.path.append(str(utils))


def is_in_list(list_a: list, list_b: list) -> bool:
    for la in list_a:
        if la in list_b:
            return True
        else:
            pass
    return False


def glob_multiext(ext_tags: BaseTag, path: pathlib.Path):
    # ex) .(mp4|avi) -> .mp4 or .avi or .MP4 or .AVI
    pattern = f".({'|'.join([ext.name for ext in ext_tags])})"
    # print(pattern)
    return sorted(
        [p for p in path.glob("**/*") if re.match(pattern, str(p.suffix), flags=re.IGNORECASE)]
    )
