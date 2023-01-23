import os
from typing import Any

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, rand
from torch.autograd import Variable
from torch.autograd.grad_mode import no_grad
from torch.nn import CrossEntropyLoss, Linear, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50

import wandb

_DATA_MEAN = [0.5, 0.5, 0.5]
_DATA_STD = [0.225, 0.225, 0.225]


class FaceRecDataModule(LightningDataModule):
    name = "facerec"

    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = True,
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
        self.drop_last = drop_last

    def setup(self) -> None:
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        normalize = transforms.Normalize(mean=_DATA_MEAN, std=_DATA_STD)

        self.train_dataset = ImageFolder(
            train_dir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.val_dataset = ImageFolder(
            val_dir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


def get_model(model_name: str = "resnet50", weights: str = "DEFAULT") -> Any:
    # TODO: Fix return type to pytorch module.

    # init a pretrained resnet
    model = resnet50(weights=weights)
    return model


class FaceRecLearning(LightningModule):
    def __init__(self, inpput_shape, num_target_classes, learning_rate=2e-4) -> None:
        super().__init__()

        self.input_shape = inpput_shape
        self.num_target_classes = num_target_classes
        self.learning_rate = learning_rate

        backbone = get_model()
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = Sequential(*layers)

        # use the pretrained model to classify faces
        self.classifier = Linear(num_filters, num_target_classes)

        self.criterion = CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_target_classes)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        tmp_input = Variable(rand(batch_size, *shape))

        output_feat = self._forward_features(tmp_input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    def forward(self, x) -> Tensor:
        self.feature_extractor.eval()
        embeddings = self.feature_extractor(x).flatten(1)
        x = self.classifier(embeddings)

        return x

    def training_step(self, batch, batch_idx) -> float:
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        acc = self.accuracy(out, gt)

        self.log("train/loss", loss)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx) -> float:
        batch, gt = batch[0], batch[1]
        out = self.forward(batch)
        loss = self.criterion(out, gt)

        self.log("val/loss", loss)

        acc = self.accuracy(out, gt)
        self.log("val/acc", acc)

        return loss

    def configure_optimizers(self) -> Adam:
        return Adam(self.parameters(), lr=self.learning_rate)


def main():
    data_dir = "data/digiface1m/aligned/"
    accelerator = "gpu"
    data_module = FaceRecDataModule(data_dir, num_workers=16)
    model = FaceRecLearning((3, 224, 224), num_target_classes=10000, learning_rate=1e-1)
    wandb_logger = WandbLogger(project="facerec")

    trainer = Trainer(
        logger=wandb_logger,
        # max_epochs=10,
        accelerator=accelerator,
    )

    trainer.fit(model, data_module)
    wandb.finish()

    print("Hello World!")


if __name__ == "__main__":
    main()
