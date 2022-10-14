from enum import Enum, auto
from typing import Any


class BaseTag(Enum):
    @classmethod
    def value_of(cls, target_value: Any) -> Any:
        for e in cls:
            if e.value == target_value:
                return e
        raise ValueError(f"{target_value} is invalid value")

    # def __eq__(self, __o: object) -> bool:
    #     if not isinstance(__o, BaseTag):
    #         return NotImplemented
    #     return bool(self.value == __o.value)


class SessionTag(BaseTag):
    MDet = auto()
    MDetCrop = auto()
    MDetRender = auto()
    Clip = auto()
    Cls = auto()


session_tag_list = [tag for tag in SessionTag]


class ImageSuffix(BaseTag):
    JPG = "jpg"
    PNG = "png"
    BMP = "bmp"


class VideoSuffix(BaseTag):
    MP4 = "mp4"
    AVI = "avi"
