from pytorch_lightning.cli import LightningCLI

import wandb
from src.data import FaceRecDataModule
from src.model import FaceRecLearning


def main():
    LightningCLI(FaceRecLearning, FaceRecDataModule, save_config_callback=None)
    wandb.finish()


if __name__ == "__main__":
    main()
