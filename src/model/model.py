from typing import Any, Union

from pytorch_lightning import LightningModule
from torch import Tensor, argmax, rand
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.nn.functional import softmax
from torch.optim import Adam, Optimizer
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import resnet18

from .margin import ArcMarginProduct


# use "DEFAULT" for weights if want to use imagenet pretraining
def get_model(model_name: str = "resnet18", weights: Union[None, "str"] = None) -> Any:
    # TODO: Fix return type to pytorch module.

    # init a pretrained resnet
    model = resnet18(weights=weights)
    return model


class FaceRecLearning(LightningModule):
    name: str = "FaceRecLearning"

    def __init__(
        self,
        inpput_shape: tuple[int, int, int] = (3, 224, 224),
        num_target_classes: int = 10000,
        learning_rate: float = 3e-4,
        classifier_type: str = "",
        margin_penalty: float = 0.5,
        margin_scale: float = 30.0,
    ) -> None:
        super().__init__()

        self.input_shape = inpput_shape
        self.num_target_classes = num_target_classes
        self.learning_rate = learning_rate
        self.classifier_type = classifier_type
        self.margin_penalty = margin_penalty
        self.margin_scale = margin_scale

        self.save_hyperparameters()

        backbone = get_model()
        num_filters = backbone.fc.in_features
        num_output_filters = 128
        layers = list(backbone.children())[:-1]
        self.feature_extractor = Sequential(*layers)
        self.reduce_feature_dim = Sequential(
            Linear(num_filters, 256),
            ReLU(),
            Linear(256, num_output_filters),
            ReLU(),
        )
        if classifier_type is not None and classifier_type == "arcproduct":
            self.classifier = ArcMarginProduct(
                num_output_filters,
                num_target_classes,
                s=margin_penalty,
                m=margin_scale,
            )
        else:
            self.classifier = Linear(num_output_filters, num_target_classes)

        self.criterion = CrossEntropyLoss()
        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_target_classes, average="micro"
        )
        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_target_classes, average="micro"
        )

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape) -> int:
        batch_size = 1
        tmp_input = Variable(rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x) -> Tensor:
        x = self.feature_extractor(x)
        return x

    def forward(self, x) -> Tensor:
        # embedding / feature extraction
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.reduce_feature_dim(x)

        return x

    def classify(self, x, y=None) -> Tensor:
        if self.classifier_type is not None and self.classifier_type == "arcproduct":
            x = self.classifier(x, y)
        else:
            x = self.classifier(x)

        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> float:
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        out = self.classify(out, gt)
        loss = self.criterion(out, gt)
        pred = argmax(softmax(out), 1)

        acc = self.train_accuracy(pred, gt)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> float:
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        out = self.classify(out, gt)
        loss = self.criterion(out, gt)
        acc = self.val_accuracy(out, gt)

        self.log("val/acc", acc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Any]]:
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = Adam(trainable_parameters, lr=self.learning_rate)
        return [optimizer], []
