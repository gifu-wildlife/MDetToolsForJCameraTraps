from typing import Optional, Union
from torch import Tensor
from PIL.Image import Image
import torchvision.transforms.functional as F


class LongsideResizeSquarePadding:
    def __init__(
        self,
        size: int,
        interpolation: F.InterpolationMode = F.InterpolationMode.NEAREST,
        antialias: Optional[bool] = None,
    ) -> None:
        if not isinstance(size, int):
            raise TypeError(
                "Argument size shoud be a int, please input size after resizing the long side."
            )

        self.size = size
        self.interpolation = interpolation
        self.antialias = antialias

    def __call__(self, img) -> Union[Tensor, Image]:
        w, h = F.get_image_size(img)
        if w > h:
            r_h, r_w = int((self.size / w) * h), self.size
            r_img = F.resize(
                img, size=[r_h, r_w], interpolation=self.interpolation, antialias=self.antialias
            )
            pad_size = self.size - r_h
            odd_pad = 1 if pad_size % 2 == 1 else 0
            pad_img = F.pad(r_img, padding=[0, pad_size // 2, 0, pad_size // 2 + odd_pad])
        else:
            r_h, r_w = self.size, int((self.size / h) * w)
            r_img = F.resize(
                img, size=[r_h, r_w], interpolation=self.interpolation, antialias=self.antialias
            )

            pad_size = self.size - r_w
            odd_pad = 1 if pad_size % 2 == 1 else 0
            pad_img = F.pad(r_img, padding=[pad_size // 2, 0, pad_size // 2 + odd_pad, 0])
        # print(img.size)
        # print(r_img.size)
        # print(pad_img.size, pad_size, pad_size // 2)
        # print()
        return pad_img
