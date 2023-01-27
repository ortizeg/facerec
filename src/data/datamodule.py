from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

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
