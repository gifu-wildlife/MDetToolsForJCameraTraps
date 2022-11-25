# import json
# import re
from pathlib import Path

import numpy as np

# import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.classifire.transforms import LongsideResizeSquarePadding
from utils import image_pathlist_load_from_file


class PredictionDetectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source: Path,
        transform: transforms.Compose = None,
    ) -> None:
        super().__init__()
        if isinstance(data_source, str):
            data_source = Path(data_source)
        if data_source.is_dir():
            # print("directory")
            self.data_paths = sorted(
                list(data_source.glob("**/*.png")) + list(data_source.glob("**/*.jpg"))
            )
        self.data_paths = image_pathlist_load_from_file(data_source, recursivce=True, exchange=True)

        if transform is None:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize(224),
                    LongsideResizeSquarePadding(
                        size=224,
                        interpolation=TF.InterpolationMode.NEAREST,
                        antialias=False,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010),
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]

        return self.preprocess(data_path), str(data_path)

    def preprocess(self, item):
        img = np.float32(np.array(Image.open(item)) / 255)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = np.uint8(img * 255)
            img = Image.fromarray(img)
        else:
            img = Image.open(item)

        img = self.transform(img)
        return img
