import json
import os
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from megadetector.data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name,
)
from megadetector.detection.run_detector import ImagePathUtils
from megadetector.detection.run_detector_batch import (
    load_and_run_detector_batch,
    write_results_to_file,
)
from megadetector.visualization import visualization_utils as vis_utils
from utils.config import MDetConfig
from utils.timer import Timer

DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

logger = getLogger("root")


def run_detector(
    detector_config: MDetConfig,
    output_json_name: str = "detector_output.json",
) -> None:
    image_source: Path = detector_config.image_source
    image_data_dir = image_source if image_source.is_dir() else image_source.parent

    assert (
        detector_config.model_path.exists()
    ), f"detector file {str(detector_config.model_path)} does not exist"
    assert (
        0.0 < detector_config.threshold <= 1.0
    ), "Confidence threshold needs to be between 0 and 1"  # Python chained comparison
    assert output_json_name.endswith(
        ".json"
    ), "output_file specified needs to end with .json"
    if not detector_config.output_absolute_path:
        assert (
            image_source.is_dir()
        ), "image_file must be a directory when megadetector.output_absolute_path is not True"

    if image_source.is_dir():
        image_file_names = ImagePathUtils.find_images(
            str(image_source), detector_config.recursive
        )
        image_file_names = [
            image_file_name
            for image_file_name in image_file_names
            if not os.path.join(image_source, "exept_dif") in image_file_name
        ]
        logger.info(
            "{} image files found in the input directory".format(len(image_file_names))
        )
    # A json list of image paths
    elif image_source.is_file() and image_source.suffix == ".json":
        with open(image_source) as f:
            image_file_names = json.load(f)
        logger.info(
            "{} image files found in the json list".format(len(image_file_names))
        )
    elif image_source.is_file() and image_source.suffix == ".csv":
        df = pd.read_csv(str(image_source), header=0)
        image_file_names = df["fullpath"].to_list()
        logger.info(
            "{} image files found in the csv list".format(len(image_file_names))
        )
    # A single image file
    elif image_source.is_file and ImagePathUtils.is_image_file(str(image_source)):
        image_file_names = [image_source]
        logger.info("A single image at {} is the input file".format(image_source))
        # photo_data_dir = image_source.parent
    else:
        raise ValueError(
            "image_source specified is not a directory, a json list, or an image file, "
            "(or does not have recognizable extensions)."
        )

    assert (
        len(image_file_names) > 0
    ), "Specified image_source does not point to valid image files"
    assert os.path.exists(
        image_file_names[0]
    ), f"The first image to be scored does not exist at {image_file_names[0]}"

    logger.info(f"Photo data directory contains {len(image_file_names)} images.")

    results = []
    with Timer(timer_tag="MegaDetector", verbose=True, logger=logger):
        results = load_and_run_detector_batch(
            model_file=str(detector_config.model_path),
            image_file_names=image_file_names,
            checkpoint_path=None,
            confidence_threshold=detector_config.threshold,
            checkpoint_frequency=-1,
            results=results,
            n_cores=detector_config.ncores,
            use_image_queue=True,
            quiet=not detector_config.verbose,
        )

    logger.info(f"Finished inference for {len(results)} images.")

    relative_path_base = None
    if not detector_config.output_absolute_path:
        relative_path_base = str(image_data_dir)
    write_results_to_file(
        results,
        str(image_data_dir.joinpath(output_json_name)),
        relative_path_base=relative_path_base,
        detector_file=str(detector_config.model_path),
    )


def detection_result_visualize(
    detector_output_file: Path,
    original_image_dir: Path,
    output_dir: Path,
    confidence: float,
    output_image_width: int = 700,
):
    assert (
        confidence > 0 and confidence < 1
    ), f"Confidence threshold {confidence} is invalid, must be in (0, 1)."
    assert (
        detector_output_file.exists()
    ), f"Detector output file does not exist at {detector_output_file}."
    assert original_image_dir

    with open(detector_output_file) as f:
        detector_output = json.load(f)
    assert (
        "images" in detector_output
    ), 'Detector output file should be a json with an "images" field.'
    images = detector_output["images"]

    detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
    if "detection_categories" in detector_output:
        logger.info("detection_categories provided")
        detector_label_map = detector_output["detection_categories"]

    num_images = len(images)
    logger.info(f"Detector output file contains {num_images} entries.")

    logger.info(
        "Rendering detections above a confidence threshold of {}...".format(confidence)
    )
    num_saved = 0
    annotated_img_paths = []
    image_obj: Any

    for entry in tqdm(images):
        image_id = entry["file"]
        if entry["max_detection_conf"] < confidence:
            continue

        if "failure" in entry:
            logger.info(f'Skipping {image_id}, failure: "{entry["failure"]}"')
            continue

        # image_obj = original_image_dir.joinpath(image_id)
        if Path(image_id).is_absolute():
            image_obj = Path(image_id)
        else:
            image_obj = original_image_dir.joinpath(image_id)

        if not image_obj.exists():
            logger.info(f"Image {image_id} not found in images_dir; skipped.")
            continue

        rendered_image = vis_utils.resize_image(
            vis_utils.open_image(image_obj), output_image_width
        )

        vis_utils.render_detection_bounding_boxes(
            entry["detections"],
            rendered_image,
            label_map=detector_label_map,
            confidence_threshold=confidence,
        )

        image = vis_utils.open_image(image_obj)
        images_cropped = vis_utils.crop_image(entry["detections"], image)

        # for char in ['/', '\\', ':']:
        #     image_id = image_id.replace(char, '~')
        # annotated_img_path = output_dir.joinpath(f'anno_{image_id}')
        # image_parts = list(Path(image_id).parts)
        # save_dir = output_dir.joinpath("/".join(image_parts[:-1]))
        image_parts = [
            parts
            for parts in list(image_obj.parts)
            if parts not in original_image_dir.parts
        ]
        # logger.info(output_dir)
        # logger.info(image_parts)
        # logger.info("/".join(image_parts[:-1]))
        save_dir = output_dir.joinpath("/".join(image_parts[:-1]))
        crop_save_dir = save_dir / "crop"

        if not crop_save_dir.exists():
            os.makedirs(crop_save_dir, exist_ok=True)
        for i_crop, cropped_image in enumerate(images_cropped):
            img_name, ext = image_parts[-1].split(".")
            crop_img_path = crop_save_dir.joinpath(f"{img_name}---{i_crop}.{ext}")
            cropped_image.save(crop_img_path)

        annotated_img_path = save_dir.joinpath(f"anno_{image_parts[-1]}")
        if not annotated_img_path.parent.exists():
            os.makedirs(str(annotated_img_path.parent), exist_ok=True)
        annotated_img_paths.append(annotated_img_path)
        rendered_image.save(annotated_img_path)
        num_saved += 1

    logger.info(
        f"Rendered detection results on {num_saved} images, " f"saved to {output_dir}."
    )

    return annotated_img_paths


def detection_result_crop(
    detector_output_file: Path,
    original_image_dir: Path,
    output_dir: Path,
    confidence: float,
):
    assert (
        confidence > 0 and confidence < 1
    ), f"Confidence threshold {confidence} is invalid, must be in (0, 1)."
    assert (
        detector_output_file.exists()
    ), f"Detector output file does not exist at {detector_output_file}."
    assert original_image_dir

    with open(detector_output_file) as f:
        detector_output = json.load(f)
    assert (
        "images" in detector_output
    ), 'Detector output file should be a json with an "images" field.'
    images = detector_output["images"]

    # detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
    # if 'detection_categories' in detector_output:
    #     logger.info('detection_categories provided')
    #     detector_label_map = detector_output['detection_categories']

    num_images = len(images)
    logger.info(f"Detector output file contains {num_images} entries.")

    logger.info(
        "Cropping detections above a confidence threshold of {}...".format(confidence)
    )
    num_saved = 0
    annotated_img_paths = []
    image_obj: Any

    for entry in tqdm(images):
        image_id = entry["file"]
        if entry["max_detection_conf"] < confidence:
            continue

        if "failure" in entry:
            logger.info(f'Skipping {image_id}, failure: "{entry["failure"]}"')
            continue

        # image_obj = original_image_dir.joinpath(image_id)
        if Path(image_id).is_absolute():
            image_obj = Path(image_id)
        else:
            image_obj = original_image_dir.joinpath(image_id)

        if not image_obj.exists():
            logger.info(f"Image {image_id} not found in images_dir; skipped.")
            continue

        image = vis_utils.open_image(image_obj)
        images_cropped = vis_utils.crop_image(
            entry["detections"], image, confidence_threshold=0.8
        )

        # for char in ['/', '\\', ':']:
        #     image_id = image_id.replace(char, '~')
        # annotated_img_path = output_dir.joinpath(f'anno_{image_id}')
        # image_parts = list(image_obj.parts) - list(original_image_dir.parts)
        image_parts = [
            parts
            for parts in list(image_obj.parts)
            if parts not in original_image_dir.parts
        ]
        # logger.info(output_dir)
        # logger.info(image_parts)
        # logger.info("/".join(image_parts[:-1]))
        save_dir = output_dir.joinpath("/".join(image_parts[:-1]))

        if not save_dir.exists():
            os.makedirs(save_dir, exist_ok=True)
        for i_crop, cropped_image in enumerate(images_cropped):
            # img_name, ext = image_parts[-1].split(".")
            img_name, ext = Path(image_id).stem, Path(image_id).suffix
            crop_img_path = save_dir.joinpath(f"{img_name}---{i_crop}{ext}")
            cropped_image.save(crop_img_path)
        num_saved += 1

    logger.info(
        f"Cropping detection results on {num_saved} images, " f"saved to {output_dir}."
    )

    return annotated_img_paths


def export_json2csv(
    detector_output_file: Path,
    original_image_dir: Path,
    confidence: float = 0.1,
):
    assert (
        detector_output_file.exists()
    ), f"Detector output file does not exist at {detector_output_file}."
    assert original_image_dir

    with open(detector_output_file) as f:
        detector_output = json.load(f)
    assert (
        "images" in detector_output
    ), 'Detector output file should be a json with an "images" field.'
    images = detector_output["images"]

    num_images = len(images)
    logger.info(f"Detector output file contains {num_images} entries.")

    detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
    if "detection_categories" in detector_output:
        logger.info("detection_categories provided")
        detector_label_map = detector_output["detection_categories"]

    num_bboxes = 0
    bboxes_rows = []
    for entry in tqdm(images):
        if entry["max_detection_conf"] < confidence:
            # continue
            pass

        image_id = entry["file"]
        if "failure" in entry:
            logger.info(f'Skipping {image_id}, failure: "{entry["failure"]}"')
            continue

        # image_obj = original_image_dir.joinpath(image_id)
        if Path(image_id).is_absolute():
            image_obj = Path(image_id)
        else:
            image_obj = original_image_dir.joinpath(image_id)

        if not image_obj.exists():
            logger.info(f"Image {image_id} not found in images_dir; skipped.")
            continue
        image = vis_utils.open_image(image_obj)

        for i, detection in enumerate(entry["detections"]):
            category = detector_label_map.get(detection["category"])
            confidence_score = float(detection["conf"])
            x1, y1, w_box, h_box = detection["bbox"]
            ymin, xmin, ymax, xmax = y1, x1, y1 + h_box, x1 + w_box
            im_width, im_height = image.size

            bboxes_rows.append(
                [
                    str(image_obj),
                    i,
                    im_height,
                    im_width,
                    category,
                    confidence_score,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                ]
            )

        num_bboxes += len(entry["detections"])

    columns = [
        "fullpath",
        "crop_id",
        "im_height",
        "im_width",
        "category",
        "confidence",
        "box_ymin",
        "box_xmin",
        "box_ymax",
        "box_xmax",
    ]
    assert len(bboxes_rows[0]) == len(columns)
    bboxes_df = (
        pd.DataFrame(bboxes_rows, columns=columns)
        .sort_values("fullpath")
        .reset_index(drop=True)
    )
    bboxes_df.to_csv(str(original_image_dir.joinpath("detector_output.csv")))

    logger.info(
        f"JSON to CSV exchanging results on {num_images} images, "
        f"{num_bboxes} bboxes."
    )


def detection_crop_annotation(
    detector_output_csv_file: Path, annotation_csv_file: Path
):
    detection_df = pd.read_csv(
        str(detector_output_csv_file), header=0, index_col=0
    ).loc[:, ["fullpath", "crop_id"]]
    annotation_df = pd.read_csv(
        str(annotation_csv_file),
        header=0,
        index_col=None,
        converters={"fullpath": str, "category": str},
    ).loc[:, ["fullpath", "category", "learning_phase"]]

    output_df = pd.merge(detection_df, annotation_df)
    vis_path_list = []
    original_image_dir = detector_output_csv_file.parent
    for row_name, row in output_df.iterrows():
        fullpath = Path(row.fullpath)
        image_parts = [
            parts
            for parts in list(fullpath.parts)
            if parts not in list(original_image_dir.parts)
        ]
        output_dir = original_image_dir.parent.joinpath(
            str(original_image_dir.stem) + "-crop"
        )
        vis_path = output_dir.joinpath("/".join(image_parts))
        vis_path_list.append(
            f"{vis_path.parent}/{vis_path.stem}---{row.crop_id}{vis_path.suffix}"
        )
    output_df["vis_path"] = vis_path_list
    logger.info(output_df.head())

    output_df.to_csv(
        str(
            detector_output_csv_file.parent.joinpath(
                f"{detector_output_csv_file.stem}-annotation_data.csv"
            )
        ),
        index=None,
    )


def detection_visualize_annotation(
    detector_output_csv_file: Path, annotation_csv_file: Path
):
    detection_df = pd.read_csv(
        str(detector_output_csv_file), header=0, index_col=0
    ).loc[:, ["fullpath", "crop_id"]]
    annotation_df = pd.read_csv(
        str(annotation_csv_file),
        header=0,
        index_col=None,
        converters={"fullpath": str, "category": str},
    ).loc[:, ["fullpath", "category", "learning_phase"]]

    output_df = pd.merge(detection_df, annotation_df)
    vis_path_list = []
    original_image_dir = detector_output_csv_file.parent
    for row_name, row in output_df.iterrows():
        fullpath = Path(row.fullpath)
        image_parts = [
            parts
            for parts in list(fullpath.parts)
            if parts not in list(original_image_dir.parts)
        ]
        output_dir = original_image_dir.parent.joinpath(
            str(original_image_dir.stem) + "-visualize"
        )
        vis_path = output_dir.joinpath("/".join(image_parts))
        vis_path_list.append(f"{vis_path.parent}/anno_{vis_path.stem}{vis_path.suffix}")
    output_df["vis_path"] = vis_path_list
    logger.info(output_df.head())

    output_df.to_csv(
        str(
            detector_output_csv_file.parent.joinpath(
                f"{detector_output_csv_file.stem}-annotation_data_vis.csv"
            )
        ),
        index=None,
    )
