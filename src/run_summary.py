from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.utils.config import SummaryConfig
except ModuleNotFoundError:
    from utils.config import SummaryConfig


def img_cls_summary(
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
