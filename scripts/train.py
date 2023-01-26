import os
from typing import Any, Union

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningCLI
from torch import Tensor, rand
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18

import wandb

_DATA_MEAN = [0.5, 0.5, 0.5]
_DATA_STD = [0.5, 0.5, 0.5]


class FaceRecDataModule(LightningDataModule):
    name: str = "FaceRecDataModule"

    def __init__(
        self,
        data_dir: str = "data/digiface1m/aligned/",
        image_size: int = 224,
        num_workers: int = 8,
        batch_size: int = 32,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last

        self.save_hyperparameters()

    def setup(self, stage=None) -> None:
        train_dir = os.path.join(self.data_dir, "train")
        normalize = transforms.Normalize(mean=_DATA_MEAN, std=_DATA_STD)

        full_train_dataset = ImageFolder(
            train_dir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size=self.image_size + 32, scale=(0.8, 1.0)
                    ),
                    transforms.RandomRotation(degrees=15),
                    transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(size=self.image_size),
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        # Random split
        train_dataset_size = int(len(full_train_dataset) * 0.8)
        val_dataset_size = len(full_train_dataset) - train_dataset_size
        train_set, val_set = random_split(
            full_train_dataset, [train_dataset_size, val_dataset_size]
        )

        self.train_dataset = train_set
        self.val_dataset = val_set

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )


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
    ) -> None:
        super().__init__()

        self.input_shape = inpput_shape
        self.num_target_classes = num_target_classes
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        backbone = get_model()
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = Sequential(*layers)
        self.reduce_feature_dim = Sequential(
            Linear(num_filters, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
        )
        self.classifier = Linear(128, num_target_classes)

        self.criterion = CrossEntropyLoss()
        self.train_accuracy = MulticlassAccuracy(num_target_classes)
        self.val_accuracy = MulticlassAccuracy(num_target_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = Variable(rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    def forward(self, x) -> Tensor:
        # embedding / feature extraction
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.reduce_feature_dim(x)

        return x

    def training_step(self, batch, batch_idx):
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        out = self.classifier(out)
        loss = self.criterion(out, gt)
        acc = self.train_accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx) -> float:
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        out = self.classifier(out)
        loss = self.criterion(out, gt)
        acc = self.val_accuracy(out, gt)

        self.log("val/acc", acc, sync_dist=True)
        self.log("val/loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Any]]:
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = Adam(trainable_parameters, lr=self.learning_rate)
        return [optimizer], []


def main():
    cli = LightningCLI(FaceRecLearning, FaceRecDataModule, save_config_callback=None)
    wandb.finish()


if __name__ == "__main__":
    main()
