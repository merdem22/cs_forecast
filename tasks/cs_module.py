# tasks/cs_module.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional
from utils.metrics import binary_cls_metrics

def _mae(pred, targ): return torch.mean(torch.abs(pred - targ))
def _mse(pred, targ): return torch.mean((pred - targ) ** 2)

class CSDownstreamModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-3, weight_decay=1e-2,
                 task: str = "reg", cls_threshold: float = 0.5,
                 scheduler_cfg: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.task = str(task)
        self.cls_threshold = float(cls_threshold)
        self.scheduler_cfg = scheduler_cfg or {}

        self.save_hyperparameters(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "task": self.task,
                "cls_threshold": self.cls_threshold,
                "scheduler": self.scheduler_cfg,
            },
            ignore=["model"]
        )

        if self.task == "reg":
            self.criterion = nn.L1Loss()
        elif self.task == "cls":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError("task must be 'reg' or 'cls'.")

    def forward(self, x): return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y, _ = batch
        y_hat = self(x)

        if self.task == "reg":
            loss = self.criterion(y_hat, y)
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
            self.log(f"{stage}_mae",  F.l1_loss(y_hat, y), prog_bar=True, on_epoch=True, batch_size=x.size(0))
            self.log(f"{stage}_mse",  F.mse_loss(y_hat, y), on_epoch=True, batch_size=x.size(0))
        else:
            if y.dim() == 1:
                y = y.unsqueeze(1)
            y = y.float()
            loss = self.criterion(y_hat, y)
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)

            acc = binary_cls_metrics(y_hat.detach(), y.detach(), threshold=self.cls_threshold)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)

        return loss, y_hat, y

    def training_step(self, batch, idx):
        loss, _, _ = self._shared_step(batch, "train")
        return loss

    def on_test_epoch_start(self):
        self._test_y = []
        self._test_yhat = []

    def test_step(self, batch, idx):
        loss, y_hat, y = self._shared_step(batch, "test")
        self._test_y.append(y.detach().cpu())
        self._test_yhat.append(y_hat.detach().cpu())
        return loss
    
    def on_test_epoch_end(self):
        self.test_Y_true = torch.vstack(self._test_y)
        self.test_Y_pred = torch.vstack(self._test_yhat)
        self._test_y, self._test_yhat = [], []

    def configure_optimizers(self):
        custom = getattr(self, "custom_optimizer", None)
        if isinstance(custom, torch.optim.Optimizer):
            return custom
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if not self.scheduler_cfg:
            return opt

        sch_type = str(self.scheduler_cfg.get("type", "")).lower()
        if sch_type == "cosine":
            t_max = int(self.scheduler_cfg.get("t_max", 50))
            eta_min = float(self.scheduler_cfg.get("eta_min", 0.0))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "name": "cosine_lr",
                },
            }

        raise ValueError(f"Unsupported scheduler type '{sch_type}'.")
