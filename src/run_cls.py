import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from src.classifire.transforms import LongsideResizeSquarePadding


class PredictionDetectorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mdet_output_path: Path,
        transform: transforms.Compose = None,
    ) -> None:
        super().__init__()
        if isinstance(mdet_output_path, str):
            mdet_output_path = Path(mdet_output_path)
        with open(mdet_output_path) as f:
            detector_output = json.load(f)

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
