import argparse
import os
from concurrent import futures
from logging import getLogger
from pathlib import Path
from typing import Union

import cv2
from tqdm import tqdm
from utils import glob_multiext
from utils.config import ClipConfig
from utils.tag import ImageSuffix, VideoSuffix

logger = getLogger("root")


def save_frame(
    video_path: Path,
    save_path: Path,
    start_frame: int = 0,
    end_frame: Union[int, None] = None,
    step: int = 1,
    ext: ImageSuffix = ImageSuffix.JPG,
    remove_banner: bool = True,
    banner_size: int = 100,
    verbose: bool = False,
) -> None:

    """
    crop frames from a video at given intervals.
    """

    # check the existance of the video file.
    assert video_path.exists(), "File does not exist: {}".format(video_path)
    if not save_path.exists():
        # make output directory to save frames.
        os.makedirs(save_path, exist_ok=True)

    # load the video file.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    # get properties of the video.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # check end_frame and the number of frames.
    if end_frame is None:
        end_frame = count
    if count < end_frame:
        end_frame = count

    result_info = f"filename: {video_path} - size: ({height},{width}) - count: {count} - fps: {fps}"
    if verbose:
        logger.info(result_info)

    digit = len(str(int(count)))

    for i in range(count):
        ret = cap.grab()
        if ret is False:
            break

        if i % step == 0:
            # get a taget frame.
            ret, img = cap.retrieve()

            # remove a banner from the frame if remove_banner is True.
            if remove_banner:
                img = img[: height - banner_size, :, :]

            # save the frame.
            frame_name = f"{str(i).zfill(digit)}.{ext.value}"
            cv2.imwrite(str(save_path.joinpath(frame_name)), img)
    return result_info


def get_video_path(src_dir: Path, dst_dir: Path) -> list[list[Path]]:
    """
    create a list of video path and output directory pairs
    to extract frames from the video.
    """
    # get all filepath with .mp4 extention from the input directory.
    video_paths = glob_multiext(VideoSuffix, src_dir)
    # print(video_paths)
    num_videos = len(video_paths)

    src_dst_list = []
    for i in range(num_videos):
        video_path = video_paths[i]

        # get filename from video_path
        video_name = video_path.stem

        # make list of output dirctory to save frames
        dir_parts = list(video_path.parent.parts)
        dir_parts[dir_parts.index(str(src_dir.name))] = str(dst_dir.name)
        _dst_dir = Path(*dir_parts).joinpath(video_name)
        src_dst_list.append([video_path, _dst_dir])

    return src_dst_list


def get_img_path(img_dirs: list[Path]) -> Union[tuple[list[str], list[str]], list[str]]:
    path_list = []
    for img_dir in img_dirs:
        logger.info(f"Getting Image Path : {img_dir}......")

        paths = glob_multiext(ImageSuffix, img_dir)

        path_list = path_list + paths

    return path_list


def clip(config: Union[ClipConfig, argparse.ArgumentParser]):
    # get video path and save path pairs.
    src_dst_list = get_video_path(config.video_dir, config.output_dir)

    # future_list = []
    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tasks = [
            executor.submit(
                save_frame,
                video_path=video_path,
                save_path=save_path,
                start_frame=config.start_frame,
                end_frame=config.end_frame,
                step=config.step,
                ext=config.ext
                if isinstance(config.ext, ImageSuffix)
                else ImageSuffix.value_of(config.ext),
                remove_banner=config.remove_banner,
            )
            for video_path, save_path in src_dst_list
        ]
        for _ in tqdm(futures.as_completed(tasks), total=len(src_dst_list)):
            pass
        # for video_path, save_path in src_dst_list:
        #     # extract frames from the given video.
        #     future = executor.submit(
        #         save_frame,
        #         video_path=video_path,
        #         save_path=save_path,
        #         start_frame=config.start_frame,
        #         end_frame=config.end_frame,
        #         step=config.step,
        #         ext=config.ext
        #         if isinstance(config.ext, ImageSuffix)
        #         else ImageSuffix.value_of(config.ext),
        #         remove_banner=config.remove_banner,
        #     )
        #     future_list.append(future)
        # _ = futures.as_completed(fs=future_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Clip frame from video")

    parser.add_argument("--video_dir", type=str, default="data")
    parser.add_argument("--output_dir_path", type=str, default="output")
    parser.add_argument("--start_frame", type=int, default=10)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--ext", type=str, default="jpg")
    parser.add_argument("--remove_banner", action="store_true")

    config = parser.parse_args()

    clip(config)
