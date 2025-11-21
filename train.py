import os

import lightning as L
import torch
import torch_optimizer as optim
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from model import Perceiver

torch.set_float32_matmul_precision("high")

# ---------
# Model
# ---------


class ModelTrainer(L.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        num_freq_bands: int,
        max_freq: int,
        depth: int,
        num_latents: int,
        latent_dim: int,
        self_per_cross_attn: int,
        weight_share: bool = False,
        compile: bool = False
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        latent_heads = 8
        assert latent_dim % latent_heads == 0, 'latent_dim not divisible by latent_dim_head'
        latent_dim_head = latent_dim // 8
        self.model = Perceiver(
            input_channels=3,
            input_axis=2,
            num_freq_bands=num_freq_bands,
            max_freq=max_freq,
            depth=depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=1,
            latent_heads=latent_heads,
            cross_dim_head=261,
            latent_dim_head=latent_dim_head,
            num_classes=1000,
            self_per_cross_attn=self_per_cross_attn,
            weight_share=weight_share
        )
        if compile:
            self.model = torch.compile(self.model)
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
        no_decay = ['bias', 'norm.weight', 'norm_context.weight']
        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if not any(nd in n for nd in no_decay)]
        no_decay_params = [p for n, p in param_dict.items() if any(nd in n for nd in no_decay)]

        optimizer_grouped_params = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        optimizer = optim.Lamb(optimizer_grouped_params, lr=self.learning_rate, weight_decay=self.weight_decay)
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
    def __init__(self, batch_size: int, data_dir: str):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 8
        self.data_dir = data_dir

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
    cli = LightningCLI(model_class=ModelTrainer, datamodule_class=ImageNetData)
