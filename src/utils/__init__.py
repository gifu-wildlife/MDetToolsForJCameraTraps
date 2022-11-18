import json
import os
import re
from pathlib import Path
from logging import getLogger

import pandas as pd

# import sys
# utils = Path(__file__).parent
# mdetlib = Path(__file__).parent.parent.joinpath("megadetector")
# if not str(utils) in sys.path:
#     sys.path.append(str(utils))
# if not str(mdetlib) in sys.path:
#     sys.path.append(str(mdetlib))

from utils.tag import BaseTag
from megadetector.detection.run_detector import ImagePathUtils


log = getLogger(__file__)


def is_in_list(list_a: list, list_b: list) -> bool:
    for la in list_a:
        if la in list_b:
            return True
        else:
            pass
    return False


def image_pathlist_load_from_file(
    image_source: Path, recursivce: bool, exchange: bool = False
) -> list[str]:
    if isinstance(image_source, str):
        image_source = Path(image_source)
    if image_source.is_dir():
        image_file_names = ImagePathUtils.find_images(
            dir_name=str(image_source), recursive=recursivce
        )
        image_file_names = [
            image_file_name
            for image_file_name in image_file_names
            if not os.path.join(image_source, "exept_dif") in image_file_name
        ]
        log.info("{} image files found in the input directory".format(len(image_file_names)))
    # A json list of image paths
    elif image_source.is_file() and image_source.suffix == ".json":
        with open(image_source) as f:
            image_file_names = json.load(f)
        log.info("{} image files found in the json list".format(len(image_file_names)))
    elif image_source.is_file() and image_source.suffix == ".csv":
        df = pd.read_csv(str(image_source), header=0)
        image_file_names = df["fullpath"].to_list()
        log.info("{} image files found in the csv list".format(len(image_file_names)))
    # A single image file
    elif image_source.is_file and ImagePathUtils.is_image_file(str(image_source)):
        image_file_names = [image_source]
        log.info("A single image at {} is the input file".format(image_source))
        # photo_data_dir = image_source.parent
    else:
        raise ValueError(
            "image_source specified is a directory, a csv list, a json list, or an image file, "
            "(or does not have recognizable extensions)."
        )
    if exchange:
        image_file_names = [Path(path) for path in image_file_names]
    return image_file_names


def glob_multiext(ext_tags: BaseTag, path: Path):
    # ex) .(mp4|avi) -> .mp4 or .avi or .MP4 or .AVI
    pattern = f".({'|'.join([ext.name for ext in ext_tags])})"
    # print(pattern)
    return sorted(
        [p for p in path.glob("**/*") if re.match(pattern, str(p.suffix), flags=re.IGNORECASE)]
    )
