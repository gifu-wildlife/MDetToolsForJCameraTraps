try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from torch import nn
from torchvision.models import (
    resnet18,
    resnet50,
    resnet101,
    wide_resnet50_2,
    wide_resnet101_2,
)


class Classifire(nn.Module):
    def __init__(
        self,
        arch: Literal["resnet18", "resnet50", "resnet101"],
        pretrain: bool = True,
        num_classes: int = 50,
    ) -> None:
        super().__init__()
        self.model = {
            "resnet18": resnet18(pretrained=pretrain, progress=True),
            "resnet50": resnet50(pretrained=pretrain, progress=True),
            "resnet101": resnet101(pretrained=pretrain, progress=True),
            "wide_resnet50_2": wide_resnet50_2(pretrained=pretrain, progress=True),
            "wide_resnet101_2": wide_resnet101_2(pretrained=pretrain, progress=True),
        }.get(
            arch,
            resnet50(pretrained=pretrain, progress=True),
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        logit = self.model(x)
        return logit
