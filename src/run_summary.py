import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import SummaryConfig


def img_cls_summary(
    config: SummaryConfig,
    split_symbol: str = "---",
) -> Path:
    crop_session_root = config.cls_result_dir
    mdet_result_path = config.mdet_result_path
    filename = config.cls_result_file_name
    img_session_root = mdet_result_path.parent
    summary_result_path = img_session_root.joinpath(f"{mdet_result_path.stem}_cls.json")

    category_list: np.ndarray = (
        pd.read_csv(config.category_list_path, header=None, index_col=None)
        .values[0]
        .tolist()
    )

    result_file_path = crop_session_root.joinpath(filename)
    with open(mdet_result_path) as f:
        detector_output = json.load(f)
    cls_df = (
        pd.read_csv(result_file_path, header=0)
        .sort_values("filepath")
        .reset_index(drop=True)
    )

    for filepath, category in zip(cls_df["filepath"].values, cls_df["category"].values):
        filepath = Path(filepath)
        filename, crop_id = filepath.stem.split(split_symbol)
        relative_filepath = filepath.relative_to(crop_session_root).parent.joinpath(
            filename + filepath.suffix
        )
        output_index = dict_search_value(
            detector_output["images"],
            target_key="file",
            target_value=str(relative_filepath),
        )
        # print(output_index, crop_id, relative_filepath)
        detector_output["images"][output_index]["detections"][int(crop_id)][
            "predict"
        ] = category

    detector_output["info"]["prediction_category"] = category_list

    with open(summary_result_path, "w") as f:
        json.dump(detector_output, f, indent=2)

    filepath_list = []
    substance_list = []
    n_bbox_list = []
    for image_output in detector_output["images"]:
        filepath = img_session_root.joinpath(image_output["file"])
        categories = sorted([det["predict"] for det in image_output["detections"]])
        substance = "_".join(categories)
        n_bbox = len(image_output["detections"])

        filepath_list.append(filepath)
        substance_list.append(substance)
        n_bbox_list.append(n_bbox)

    img_summary_df = pd.DataFrame(
        [filepath_list, substance_list, n_bbox_list],
        index=["filepath", "substance", "n_bbox"],
    ).T
    img_summary_df.to_csv(
        crop_session_root.joinpath(config.img_summary_name), index=None
    )

    return summary_result_path


def video_cls_summary(config: SummaryConfig) -> Path:
    mdet_result_path = config.mdet_result_path
    img_session_root = mdet_result_path.parent
    img_summary_result_path = img_session_root.joinpath(
        f"{mdet_result_path.stem}_cls.json"
    )
    with open(img_summary_result_path) as f:
        detector_output = json.load(f)

    category_list = detector_output["info"]["prediction_category"]
    filename_list = sorted(
        [output["file"].split(".")[0] for output in detector_output["images"]]
    )
    category_onehot_list = []
    for img_output in detector_output["images"]:
        if img_output["detections"]:
            categories = sorted([det["predict"] for det in img_output["detections"]])
            category_indices = [category_list.index(ci) for ci in categories]
            category_onehot = np.zeros(len(category_list), dtype=int)
            category_onehot[category_indices] = 1
        else:
            category_onehot = np.zeros(len(category_list), dtype=int)
        category_onehot_list.append(category_onehot.tolist())
    onehot_df = pd.DataFrame(
        [
            fname.split("/") + onehot
            for fname, onehot in zip(filename_list, category_onehot_list)
        ],
        columns=["loc", "movie", "frame"] + category_list,
    )
    # print(onehot_df.iloc[90:95])

    def l2s_category_sequence(d) -> pd.Series:  # list to string
        sequence_list = []
        for category in category_list:
            sequence_list.append("".join(map(str, d[category].values.tolist())))
        return pd.Series(sequence_list, index=category_list)

    sequence_group_df = onehot_df.groupby(["loc", "movie"]).apply(l2s_category_sequence)
    sequence_group_df.to_csv(img_session_root.joinpath("appearance_by_category.csv"))
    # print(type(sequence_group_df), sequence_group_df)
    loc_list = []
    movie_list = []
    sequence_list = []
    sequence_category_list = []
    for i, row_name in enumerate(sequence_group_df.index):
        seq = sequence_group_df.loc[row_name]
        category_sequence = np.array(
            [np.array(list(map(int, list(seq[ci])))) for ci in category_list]
        )
        appearance_category = category_list[np.argmax(category_sequence.sum(axis=1))]
        summary_sequence = np.where(
            category_sequence.sum(axis=0) >= 10, -1, category_sequence.sum(axis=0)
        ).tolist()
        summary_sequence_str = "".join(
            ["*" if ei == -1 else str(ei) for ei in summary_sequence]
        ).replace("0", "-")

        loc_list.append(row_name[0])
        movie_list.append(row_name[1])
        sequence_list.append(summary_sequence_str)
        sequence_category_list.append(appearance_category)
        # if i == 2:
        #     print(summary_sequence)
        #     print(summary_sequence_str, appearance_category)
    pd.DataFrame(
        [
            [li, mi, sci, si]
            for li, mi, si, sci in zip(
                loc_list, movie_list, sequence_list, sequence_category_list
            )
        ],
        columns=["location", "movie", "category", "sequence"],
    ).to_csv(
        img_session_root.joinpath("sequence_summary.csv"),
        index=None,
    )

    return img_session_root.joinpath("sequence_summary.csv")


def dict_search_value(
    dict_list: list[dict[any, any]], target_key: str, target_value: any
) -> int:
    for i, d in enumerate(dict_list):
        if d.get(target_key) == target_value:
            break
    return i


def _img_cls_summary(
    config: SummaryConfig,
    split_symbol: str = "---",
    summary_name: str = "img_wise_cls_summary.csv",
):
    root = config.cls_result_dir
    filename = config.cls_result_file_name

    result_file_path = root.joinpath(filename)
    summary_file_path = root.joinpath(summary_name)
    # result_file_path = (
    #     "_test_dataset/R3_Kinkazan_REST_Boar_Samples-crop/classifire_prediction_result.csv"
    # )
    cls_df = (
        pd.read_csv(result_file_path, header=0)
        .sort_values("filepath")
        .reset_index(drop=True)
    )
    img_summary_df = (
        pd.read_csv(summary_file_path, header=0)
        .sort_values("filepath")
        .reset_index(drop=True)
    )
    session_root = Path(img_summary_df["filepath"][0]).parent.parent
    # print(session_root)

    crop_ids = []
    src_filepaths = []
    for filepath in cls_df["filepath"].values:
        src_filename, crop_id = Path(filepath).stem.split(split_symbol)
        ext = Path(filepath).suffix
        # src_filepaths.append(Path(filepath).parent.joinpath(src_filename + ext))
        src_filepaths.append(
            session_root.joinpath(Path(filepath).parent.name).joinpath(
                src_filename + ext
            )
        )
        crop_ids.append(crop_id)
    cls_df["src_filepath"] = src_filepaths
    cls_df["crop_id"] = crop_ids
    n_bbox = pd.value_counts(src_filepaths)
    # print(cls_df)
    # print(n_bbox)

    num_of_bbox_list = []
    substance_list = []
    for src_filepath in sorted(list(set(src_filepaths))):
        num_of_bbox = n_bbox[src_filepath]
        categories = sorted(
            list(
                set(cls_df[cls_df["src_filepath"] == src_filepath]["category"].tolist())
            )
        )
        if len(categories) == 1:
            substance = categories[0]
        elif len(categories) > 1:
            substance = "_".join(categories)
        else:
            substance = "N/A"
        substance_list.append(substance)
        num_of_bbox_list.append(num_of_bbox)

    img_summary_update_df = pd.DataFrame(
        [sorted(list(set(map(str, src_filepaths)))), substance_list, num_of_bbox_list],
        index=["filepath", "substance", "n_bbox"],
    ).T

    # print(img_summary_update_df["filepath"].values.tolist()[0])
    # print(img_summary_df["filepath"].values.tolist()[0])
    # print(
    #     list(set(img_summary_update_df["filepath"].values.tolist())
    #     & set(img_summary_df["filepath"].values.tolist()))
    # )

    non_NA_filepath_list = list(
        set(img_summary_update_df["filepath"].values.tolist())
        & set(img_summary_df["filepath"].values.tolist())
    )
    non_NA_bool_list = np.array(
        [
            filepath in non_NA_filepath_list
            for filepath in img_summary_df["filepath"].values
        ]
    )
    # print(non_NA_bool_list)
    # print(
    #     len(img_summary_df),
    #     len(img_summary_df.loc[non_NA_bool_list, :]),
    #     len(img_summary_update_df),
    # )
    img_summary_df.loc[
        non_NA_bool_list, ["substance", "n_bbox"]
    ] = img_summary_update_df.loc[:, ["substance", "n_bbox"]]

    # for i in range(len(img_summary_df)):
    #     pass
    # img_summary_df["substance"].iloc[:, 0] = img_summary_update_df[
    #     img_summary_update_df["filepath"] == img_summary_df["filepath"]
    # ]
    # img_summary_df = img_summary_df.set_index("filepath", inplace=False)
    # img_summary_df.update(img_summary_update_df)
    img_summary_df.reset_index(drop=True).to_csv(
        root.joinpath(summary_name), index=None
    )
    # print(result_df)
