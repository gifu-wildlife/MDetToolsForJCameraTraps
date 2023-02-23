from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# from scipy.stats import entropy
from tqdm import tqdm

try:
    from src.classifire.dataset import PredictionDetectorDataset
    from src.classifire.models.resnet import Classifire
    from src.utils.config import ClsConfig
except ModuleNotFoundError:
    from classifire.dataset import PredictionDetectorDataset
    from classifire.models.resnet import Classifire
    from utils.config import ClsConfig

log = getLogger(__file__)
debug = True


def classifire_predict(cls_config: ClsConfig) -> Path:
    log.info("Start prediction...")

    log.info("Instantiating model")
    category: np.ndarray = pd.read_csv(
        cls_config.category_list_path, header=None, index_col=None
    ).values[0]
    if not debug:
        ckpt_path = cls_config.model_path
        log.info(f"load ckpt from {str(ckpt_path)}")
        checkpoint = torch.load(str(ckpt_path))
        model = Classifire(
            arch=cls_config.architecture, num_classes=len(category), pretrain=False
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = Classifire(
            arch=cls_config.architecture, num_classes=len(category), pretrain=True
        )

    if torch.cuda.is_available() and cls_config.use_gpu:
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    log.info("Instantiating dataset")
    dataset = PredictionDetectorDataset(data_source=cls_config.image_source)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    log.info("Start prediction loop")
    model.eval()
    top1preds = []
    # entropies = []
    top1pred_probs = []
    pred_categorys = []
    all_category_probs = []
    filepaths = []

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch[0].to(device)
            filepath = batch[1]

            output = model(images)
            prob = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            pred_category = [category[_p.item()] for _p in pred]
            pred_prob = prob.squeeze()[pred]
            # output_entropy = entropy(prob.squeeze().cpu().detach().numpy())

            all_category_probs.append(prob)
            top1preds.append(pred)
            top1pred_probs.append(pred_prob)
            # entropies.append(output_entropy)
            pred_categorys.extend(pred_category)
            filepaths.extend(filepath)

    top1pred_probs = torch.cat(top1pred_probs).tolist()
    print(torch.cat(all_category_probs, dim=0).size())
    all_category_probs = torch.cat(all_category_probs, dim=0).permute(1, 0).tolist()
    df = pd.DataFrame(
        [filepaths, pred_categorys, top1pred_probs],
        index=["filepath", "category", "probability"],
    ).T
    all_category_probs_df = pd.DataFrame(
        [filepaths] + all_category_probs,
        index=["filepath"] + category.tolist(),
    ).T
    data_dir = (
        cls_config.image_source
        if cls_config.image_source.is_dir()
        else cls_config.image_source.parent
    )
    result_path = data_dir.joinpath(cls_config.result_file_name)
    df.sort_values("filepath").reset_index(drop=True).to_csv(result_path, index=None)
    if cls_config.is_all_category_probs_output:
        all_category_probs_df.sort_values("filepath").reset_index(drop=True).to_csv(
            data_dir.joinpath("all_category_probs.csv", index=None)
        )
    return result_path
