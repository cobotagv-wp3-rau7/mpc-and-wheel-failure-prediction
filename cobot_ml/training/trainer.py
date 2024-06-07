import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from cobot_ml.data import datasets as dss, patchers
from cobot_ml.data.datasets import TensorPairsDataset
from cobot_ml.training import runners


def perform_training(model: nn.Module,
                     device: torch.device,
                     base_lr: float,
                     number_of_epochs: int,
                     patience: int,
                     batch_size: int,
                     train_dataset: dss.TensorPairsDataset,
                     valid_dataset: dss.TensorPairsDataset,
                     ):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size)

    model.to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_model_state, train_logs = runners.run_training(
        model,
        mse_loss,
        optimizer,
        train_loader,
        valid_loader,
        number_of_epochs,
        patience,
        scheduler,
        device,
    )
    return best_model_state, train_logs


def prepare_dataset(
        input: np.ndarray,
        input_steps: int,
        output_steps: int,
) -> TensorPairsDataset:
    """
    predicted column must be 0-th in the input data
    """
    patches = patchers.patch_with_stride(input, (input_steps + output_steps), stride=1)

    patches = [torch.from_numpy(patch.astype(np.float32)) for patch in patches]
    X = [patch[:input_steps, 0:] for patch in patches]
    y = [patch[input_steps:, 0] for patch in patches]
    return TensorPairsDataset(X, y)
