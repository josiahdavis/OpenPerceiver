import os
from datetime import datetime
from zoneinfo import ZoneInfo

import lightning as L
import torch
import torch_optimizer as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from model import Perceiver

torch.set_float32_matmul_precision("high")

class ImageNetConfig:
    learning_rate = 0.004
    weight_decay = 0.05
    batch_size = 32
    max_epochs = 120
    accumulate_grad_batches = 2
    data_dir = "data/imagenet"
    num_freq_bands = 64
    max_freq = 224
    depth = 2
    num_latents = 512
    latent_dim = 1024
    cross_heads = 1
    cross_dim_head = 261
    latent_heads = 8
    latent_dim_head = 64
    self_per_cross_attn = 4

# ---------
# TRAINER
# ---------

class ModelTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.model = Perceiver(
            input_channels=3,
            input_axis=2,
            num_freq_bands=config.num_freq_bands,
            max_freq=config.max_freq,
            depth=config.depth,
            num_latents=config.num_latents,
            latent_dim=config.latent_dim,
            cross_heads=config.cross_heads,
            latent_heads=config.latent_heads,
            cross_dim_head=config.cross_dim_head,
            latent_dim_head=config.latent_dim_head,
            num_classes=1000,
            self_per_cross_attn=config.self_per_cross_attn,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        inputs, labels = batch
        inputs = inputs.permute(0, 2, 3, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == labels.data)
        acc = correct / inputs.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch):
        inputs, labels = batch
        inputs = inputs.permute(0, 2, 3, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == labels.data)
        acc = correct / inputs.shape[0]
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Lamb(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[84, 102, 114], gamma=0.1)  # Factor of 10 reduction (multiply by 0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", current_lr)

# ---------
# DATA
# ---------

class ImageNetData(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = 8
        self.data_dir = config.data_dir

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1), ratio=(0.75, 1.33333), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, "train"), transform=self.train_transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        val_dataset = torchvision.datasets.ImageFolder(os.path.join(self.data_dir, "val"), transform=self.val_transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=True)

if __name__ == "__main__":
    config = ImageNetConfig()
    central_time = datetime.now(ZoneInfo("America/Chicago"))
    formatted_time = central_time.strftime("%Y-%m-%d__%I-%M-%S__%p-%Z")
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        print(f"EXP_NAME=training__{formatted_time}")
    tb_logger = TensorBoardLogger("lightning_logs_0", name=f"training__{formatted_time}")
    model = ModelTrainer(config)
    data = ImageNetData(config)
    trainer = L.Trainer(
        max_epochs=config.max_epochs, accelerator="auto", strategy="auto", accumulate_grad_batches=config.accumulate_grad_batches, logger=[tb_logger]
    )
    trainer.fit(model, data)