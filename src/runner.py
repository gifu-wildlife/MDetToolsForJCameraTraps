import os
import shutil
import time
from typing import Union

from omegaconf import OmegaConf
from run_clip import clip
from src.run_cls import classifire_predict

from src.run_megadetector import run_mdet_crop, run_megadetector
from utils.config import (
    ClipConfig,
    ClsConfig,
    MDetConfig,
    MDetCropConfig,
    MDetRenderConfig,
    RootConfig,
)
from utils.logger import get_logger
from utils.tag import SessionTag, session_tag_list
from utils.timer import Timer


class Runner:
    def __init__(
        self,
        config: RootConfig,
        session_tag: Union[SessionTag, list[SessionTag], str, list[str]],
    ) -> None:
        self.__check_config(config)
        if isinstance(session_tag, (str, list[str])):
            self.session_tags: list[SessionTag] = (
                [SessionTag.value_of(tag) for tag in session_tag]
                if isinstance(session_tag, list)
                else [SessionTag.value_of(session_tag)]
            )
        else:
            self.session_tags: list[SessionTag] = (
                session_tag if isinstance(session_tag, list) else [session_tag]
            )

        _session_tags_str = "-".join([session_tag.name for session_tag in self.session_tags])
        self.logdir = config.log_dir.joinpath(
            f'{time.strftime("%Y%m%d%H%M%S")}_{_session_tags_str}_{config.session_root.name}'
        )
        os.makedirs(self.logdir, exist_ok=True)
        self.logger = get_logger(out_dir=self.logdir, logname=f"{session_tag}.log")
        OmegaConf.save(config=config, f=self.logdir.joinpath("config.raw.yaml"))
        self.config = config

    def exec_mdet(self, config: MDetConfig) -> None:
        input_file_path = config.image_source if config.image_source.is_file() else None
        output_file_path = (
            config.image_source if config.image_source.is_dir() else config.image_source.parent
        ).joinpath("detector_output.json")
        self.logger.info(f"Start {config.image_source} MegaDetector Detection...")
        self.logger.info(f"Output file: {output_file_path}")
        run_megadetector(detector_config=config)
        self.logger.info("Detection Complete")
        shutil.copyfile(str(output_file_path), str(self.logdir.joinpath(output_file_path.name)))
        if input_file_path is not None:
            shutil.copyfile(str(input_file_path), str(self.logdir.joinpath(input_file_path.name)))

    def exec_mdet_crop(self, config: MDetCropConfig) -> None:
        self.logger.info(f"Start {config.image_source} MegaDetector Cropping...")
        self.logger.info(f"MDet Output file: {config.mdet_result_path}")
        self.logger.info(f"Output file: {config.output_dir}")
        if config.mdet_result_path is None:
            raise ValueError(
                f"Invalid Value of mdet_result_path: {config.mdet_result_path}. Please enter."
            )
        if config.output_dir is None:
            raise ValueError(f"Invalid Value of output_dir: {config.output_dir}. Please enter.")
        run_mdet_crop(config=config)
        self.logger.info("MDet cropping Complete!")

    def exec_mdet_render(self, config: MDetRenderConfig) -> None:
        pass

    def exec_clip(self, config: ClipConfig) -> None:
        self.logger.info(f"Start {config.video_dir.name} Clopping...")
        self.logger.info(f"Save Dir: {config.output_dir}")
        _end_frame = config.end_frame if config.end_frame is not None else "end"
        self.logger.info(f"Frame: {config.start_frame}-{_end_frame}")
        self.logger.info(f"Remove Banner: {config.remove_banner}")
        with Timer(verbose=True, logger=self.logger, timer_tag="Clip"):
            clip(config)
        self.logger.info("Clip Complete!")

    def exec_cls(self, config: ClsConfig) -> None:
        self.logger.info(f"Start {config.image_source} Classifire Prediction...")
        classifire_predict(cls_config=config)
        self.logger.info(f"Result file: {str(config.result_file_name)}")
        self.logger.info("Prediction Complete!")

    def __check_config(self, config: RootConfig) -> None:
        assert (
            config.session_root.exists()
        ), f"{config.session_root} does not Exists. Please enter the path where it exists"
        assert os.access(
            str(config.session_root), os.R_OK
        ), f"{config.session_root} does not Readable. Please enter the path where it readable"
        assert os.access(
            str(config.session_root), os.W_OK
        ), f"{config.session_root} does not Writable. Please enter the path where it writable"
        assert (
            config.session_root.is_absolute()
        ), f"{config.session_root} does not Absolute Path. Please enter the absolute path"

    def __drop_config(self, config: RootConfig, exec_list: dict) -> RootConfig:
        if exec_list[SessionTag.MDet] is False:
            config.mdet_config = None
        if exec_list[SessionTag.MDetCrop] is False:
            config.mdet_crop_config = None
        if exec_list[SessionTag.MDetRender] is False:
            config.mdet_render_config = None
        if exec_list[SessionTag.Clip] is False:
            config.clip_config = None
        if exec_list[SessionTag.Cls] is False:
            config.cls_config = None
        return config

    def __exec_session(
        self,
        config: RootConfig,
        session_tag: SessionTag,
    ) -> None:
        self.logger.info(session_tag)
        if session_tag == SessionTag.MDet:
            if config.mdet_config is not None:
                self.exec_mdet(config=config.mdet_config)
        elif session_tag == SessionTag.MDetCrop:
            if config.mdet_crop_config is not None:
                self.exec_mdet_crop(config=config.mdet_crop_config)
        elif session_tag == SessionTag.MDetRender:
            if config.mdet_render_config is not None:
                self.exec_mdet_render(config=config.mdet_render_config)
        elif session_tag == SessionTag.Clip:
            if config.clip_config is not None:
                self.exec_clip(config=config.clip_config)
        elif session_tag == SessionTag.Cls:
            if config.cls_config is not None:
                self.exec_cls(config=config.cls_config)

    def execute(self) -> None:
        exec_list = {k: False for k in session_tag_list}

        with Timer(verbose=True, logger=self.logger, timer_tag="AllProcess"):
            for session_tag in self.session_tags:
                self.__exec_session(config=self.config, session_tag=session_tag)
                exec_list[session_tag] = True

        exec_config = self.__drop_config(config=self.config, exec_list=exec_list)
        OmegaConf.save(config=exec_config, f=self.logdir.joinpath("config.yaml"))
