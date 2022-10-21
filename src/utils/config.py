import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from omegaconf import II, MISSING

from src.utils.tag import ImageSuffix


def cpu_count() -> int:
    cores = os.cpu_count()
    if cores is None:
        return 1
    else:
        return cores


@dataclass
class MDetConfig:
    model_path: Path = Path("models/md_v5a.0.0.pt")
    image_source: Path = MISSING
    threshold: float = 0.95
    output_absolute_path: bool = True
    ncores: int = cpu_count()
    verbose: bool = False
    recursive: bool = True


@dataclass
class MDetCropConfig:
    image_source: Path = MISSING
    mdet_result_path: Path = MISSING
    output_dir: Path = MISSING
    threshold: float = 0.95
    ncores: int = cpu_count()


@dataclass
class MDetRenderConfig:
    pass


@dataclass
class ClipConfig:
    video_dir: Path = MISSING
    output_dir: Path = MISSING
    start_frame: int = 0
    end_frame: Optional[int] = None
    step: int = 30
    ext: Union[ImageSuffix, str] = ImageSuffix.JPG
    remove_banner: bool = True


@dataclass
class ClsConfig:
    pass


@dataclass
class RootConfig:
    config_path: Optional[Path] = None
    session_root: Path = MISSING
    image_list_file_path: Optional[str] = None
    output_dir: Optional[Path] = None
    mdet_result_path: Optional[Path] = None
    log_dir: Path = Path("logs")
    mdet_config: Optional[MDetConfig] = MDetConfig(image_source=II("session_root"))
    mdet_crop_config: Optional[MDetCropConfig] = MDetCropConfig(
        image_source=II("session_root"),
        output_dir=II("output_dir"),
        mdet_result_path=II("mdet_result_path"),
    )
    mdet_render_config: Optional[MDetRenderConfig] = MDetRenderConfig()
    clip_config: Optional[ClipConfig] = ClipConfig(
        video_dir=II("session_root"), output_dir=II("output_dir")
    )
    cls_config: Optional[ClsConfig] = ClsConfig()
