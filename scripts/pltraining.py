from typing import Optional

import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from torch import nn
from torch.utils.data import DataLoader, Dataset

import cobot_ml.models as models


class TSModel(LightningModule):

    def __init__(self, model: nn.Module, loss: nn.Module):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, history, expected):
        output = self.model(history)
        return self.loss(output, expected), output

    def training_step(self, batch, batch_idx):
        history, expected = batch
        loss = self.forward(history, expected)
        return loss

    def validation_step(self, batch, batch_idx):
        history, expected = batch
        loss = self.forward(history, expected)
        return loss

    def test_step(self, batch, batch_idx):
        history, expected = batch
        loss = self.forward(history, expected)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='epoch'
            )
        )


class TSDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len(self):
        pass

    def __getitem__(self, item):
        return dict(
            history=None,
            expected=None
        )


class TSDataModule(LightningDataModule):
    def __init__(self, batch_size: int, workers_count: int):
        super().__init__()
        self.batch_size = batch_size
        self.workers_count = workers_count

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers_count
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers_count
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers_count
        )


if __name__ == "__main__":
    trainer = Trainer(
        devices=1
    )

    model = TSModel(loss=nn.MSELoss(), model=models.LSTM(25))
    data_module = TSDataModule(batch_size=64, workers_count=5)

    trainer.fit(model, data_module)
