from logging import getLogger

import pandas as pd
import torch
from scipy.stats import entropy
from tqdm import tqdm

try:
    from src.classifire.dataset import PredictionDetectorDataset
    from src.classifire.models.resnet import Classifire
    from src.utils import seed_everything
    from src.utils.config import ClsConfig
except ImportError:
    from classifire.dataset import PredictionDetectorDataset
    from classifire.models.resnet import Classifire
    from utils import seed_everything
    from utils.config import ClsConfig

log = getLogger(__file__)


def classifire_predict(cls_config: ClsConfig):
    log.info("Start prediction...")
    if cls_config.get("seed"):
        seed_everything(cls_config.seed)

    log.info("Instantiating model")
    category = pd.read_csv(cls_config.category_list_path, header=None, index_col=None).values[0]
    ckpt_path = cls_config.model_path
    log.info(f"load ckpt from {str(ckpt_path)}")
    checkpoint = torch.load(str(ckpt_path))
    model = Classifire(arch=cls_config.architecture, num_classes=len(category), pretrain=False)
    model.load_state_dict(checkpoint["state_dict"])

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
    preds = []
    entropies = []
    pred_probs = []
    pred_categorys = []
    filepaths = []

    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch[0].to(device)
            filepath = batch[1]

            output = model(images)
            prob = torch.nn.functional.softmax(output, dim=1).squeeze()
            pred = torch.argmax(output, dim=1)
            pred_category = [category[_p.item()] for _p in pred]
            pred_prob = prob[pred]
            output_entropy = entropy(prob.cpu().detach().numpy())

            preds.append(pred)
            pred_probs.append(pred_prob)
            entropies.append(output_entropy)
            pred_categorys.extend(pred_category)
            filepaths.extend(filepath)

    pred_probs = torch.cat(pred_probs).tolist()
    df = pd.DataFrame(
        [filepaths, pred_categorys, pred_probs, entropies],
        index=["filepath", "category", "probability", "entropy"],
    ).T
    data_dir = (
        cls_config.image_source
        if cls_config.image_source.is_dir()
        else cls_config.image_source.parent
    )
    result_path = data_dir.joinpath(cls_config.result_file_name)
    df.to_csv(result_path)
    return result_path
