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
    image_source: Path = MISSING
    model_path: Path = Path("models/md_v4.1.0.pb")
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
class ClsConfig:
    image_source: Path = MISSING
    model_path: Path = Path("models/classifire/15cat_50epoch_resnet50.pth")
    category_list_path: Path = Path("models/classifire/category.txt")
    result_file_name: str = "classifire_prediction_result.csv"
    architecture: str = "resnet50"
    num_classes: int = -1
    use_gpu: bool = True
    is_all_category_probs_output: bool = False


@dataclass
class SummaryConfig:
    cls_result_dir: Path = MISSING
    cls_result_file_name: str = "classifire_prediction_result.csv"
    category_list_path: Path = Path("models/classifire/category.txt")
    mdet_result_path: Path = MISSING
    img_summary_name: str = "img_wise_cls_summary.csv"
    is_video_summary: bool = True


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
class RootConfig:
    config_path: Optional[Path] = None
    session_root: Path = MISSING
    image_list_file_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    mdet_result_path: Optional[Path] = None
    log_dir: Path = Path("logs")
    mdet_config: Optional[MDetConfig] = MDetConfig(
        image_source=II("session_root"),
    )
    mdet_crop_config: Optional[MDetCropConfig] = MDetCropConfig(
        image_source=II("session_root"),
        output_dir=II("output_dir"),
        mdet_result_path=II("mdet_result_path"),
    )
    mdet_render_config: Optional[MDetRenderConfig] = MDetRenderConfig()
    clip_config: Optional[ClipConfig] = ClipConfig(
        video_dir=II("session_root"),
        output_dir=II("output_dir"),
    )
    cls_config: Optional[ClsConfig] = ClsConfig(
        image_source=II("session_root"),
    )
    summary_config: Optional[SummaryConfig] = SummaryConfig(
        cls_result_dir=II("session_root"),
        mdet_result_path=II("mdet_result_path"),
    )
