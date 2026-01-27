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
        compile: bool = False,
        decay_all: bool = True,
        lamb: bool = True,
        label_smoothing: float = 0.0,
        num_classes: int = 1000
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_all = decay_all
        self.lamb = lamb
        # Store model hyperparameters for logging
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq
        self.depth = depth
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.self_per_cross_attn = self_per_cross_attn
        self.weight_share = weight_share
        self.num_classes = num_classes
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
            num_classes=num_classes,
            self_per_cross_attn=self_per_cross_attn,
            weight_share=weight_share
        )
        if compile:
            self.model = torch.compile(self.model)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _get_samples_seen(self):
        """Calculate total training samples seen, accounting for gradient accumulation and distributed training."""
        batch_size = self.trainer.datamodule.batch_size
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        world_size = self.trainer.world_size
        return (self.global_step + 1) * batch_size * accumulate_grad_batches * world_size

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.permute(0, 2, 3, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == labels.data)
        acc = correct / inputs.shape[0]
        
        # Standard logging for callbacks, progress bar, and aggregation
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)
        
        # Additional logging with samples_seen as x-axis for consistent comparison across batch sizes
        samples_seen = self._get_samples_seen()
        self.logger.experiment.add_scalar("train_loss_by_samples", loss, global_step=samples_seen)
        self.logger.experiment.add_scalar("train_acc_by_samples", acc, global_step=samples_seen)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.permute(0, 2, 3, 1)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == labels.data)
        acc = correct / inputs.shape[0]
        
        # Standard logging for callbacks, progress bar, and aggregation (auto-averaged at epoch end)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", acc, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # Log aggregated validation metrics with samples_seen as x-axis (single point per epoch)
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_acc")
        samples_seen = self._get_samples_seen()
        
        if val_loss is not None:
            self.logger.experiment.add_scalar("val_loss_by_samples", val_loss, global_step=samples_seen)
        if val_acc is not None:
            self.logger.experiment.add_scalar("val_acc_by_samples", val_acc, global_step=samples_seen)

    def configure_optimizers(self):
        if self.decay_all:
            optimizer_params = self.parameters()
        else:
            no_decay = ['bias', 'norm.weight', 'norm_context.weight']
            param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
            decay_params = [p for n, p in param_dict.items() if not any(nd in n for nd in no_decay)]
            no_decay_params = [p for n, p in param_dict.items() if any(nd in n for nd in no_decay)]

            optimizer_params = [
                {'params': decay_params, 'weight_decay': self.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
        if self.lamb:
            optimizer = optim.Lamb(optimizer_params, betas=(0.9, 0.999), eps=1e-06, lr=self.learning_rate)
        else:
            optimizer = torch.optim.AdamW(optimizer_params, lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=self.weight_decay)
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

    def on_fit_end(self):
        """Log hyperparameters and final metrics to HPARAMS tab after training ends."""
        train_loss = self.trainer.callback_metrics.get("train_loss")
        train_acc = self.trainer.callback_metrics.get("train_acc")
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_acc")
        
        final_metrics = {}
        if train_loss is not None:
            final_metrics["final_train_loss"] = train_loss.item()
        if train_acc is not None:
            final_metrics["final_train_acc"] = train_acc.item()
        if val_loss is not None:
            final_metrics["final_val_loss"] = val_loss.item()
        if val_acc is not None:
            final_metrics["final_val_acc"] = val_acc.item()
        final_metrics["final_epoch"] = float(self.current_epoch)
        final_metrics["total_samples_seen"] = float(self._get_samples_seen())
        
        hparams = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "decay_all": self.decay_all,
            "lamb": self.lamb,
            "batch_size": self.trainer.datamodule.batch_size,
            "num_freq_bands": self.num_freq_bands,
            "max_freq": self.max_freq,
            "depth": self.depth,
            "num_latents": self.num_latents,
            "latent_dim": self.latent_dim,
            "self_per_cross_attn": self.self_per_cross_attn,
            "weight_share": self.weight_share,
        }
        
        self.logger.experiment.add_hparams(hparams, final_metrics)
        self.logger.experiment.flush()


# ---------
# DATA
# ---------

class ImageNetData(L.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str, num_ops: int = 2, magnitude: int = 9):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 8
        self.data_dir = data_dir
        self.num_ops = num_ops
        self.magnitude = magnitude

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.08, 1), ratio=(0.75, 1.33333), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=self.num_ops, magnitude=self.magnitude),
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
