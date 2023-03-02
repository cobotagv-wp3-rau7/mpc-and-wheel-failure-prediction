import json
import os
import sys
import typing
from random import sample

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils import data

sys.path.append(os.getcwd())

from cobot_ml.data.datasets import DatasetInputData, Datasets

from cobot_ml import models, detectors
from cobot_ml.data import datasets as dss, patchers
from cobot_ml.training import runners
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px

from enum import Enum


class DsMode(Enum):
    UNIVARIATE = 1
    WITH_MPC = 2
    WITHOUT_MPC = 3


def plot_results(
        real_values: np.ndarray,
        predictions: np.ndarray,
        save_path: str,
) -> None:
    fig = px.line(pd.DataFrame(dict(
        real=real_values,
        predictions=predictions,
        error=np.abs(predictions - real_values))))
    fig.write_html(f"{save_path}.html")


def unravel_vector(vector: torch.Tensor) -> np.ndarray:
    """
    Extract first value from each sample in predicted vector (but all values
    from the last sample).
    """
    return torch.cat([vector[:-1, 0], vector[-1, :]]).detach().cpu().numpy()


def prepare_dataset(
        channel_values: typing.Union[pd.DataFrame, np.ndarray],
        input_steps: int,
        output_steps: int,
        ds_mode: DsMode,
        pad_beginning: bool = False,
        take_max_samples: int = None,
) -> dss.TensorPairsDataset:
    if pad_beginning:
        channel_values = detectors.pad_beginning(channel_values, input_steps)
    patches = patchers.patch_with_stride(
        channel_values, input_steps + output_steps, stride=1
    )
    if take_max_samples is not None and len(patches) > take_max_samples:
        patches = sample(patches, take_max_samples)

    patches = [torch.from_numpy(patch.astype(np.float32)) for patch in patches]
    if ds_mode == DsMode.UNIVARIATE:
        X = [patch[:input_steps, [0]] for patch in patches]
    elif ds_mode == DsMode.WITH_MPC:
        X = [patch[:input_steps, :] for patch in patches]
    else:
        X = [patch[:input_steps, 1:] for patch in patches]
    # y = [patch[input_steps:, 0] for patch in patches]
    y = [patch[input_steps:, :] for patch in patches]
    return dss.TensorPairsDataset(X, y)


def perform_training(model: nn.Module,
                     device: torch.device,
                     base_lr: float,
                     number_of_epochs: int,
                     patience: int,
                     batch_size: int,
                     train_dataset: dss.TensorPairsDataset,
                     valid_dataset: dss.TensorPairsDataset,
                     ):
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
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


def dumps_file(path, _object):
    with open(path, "w") as params_file:
        json.dump(_object, params_file, indent=4)


def process(
        input_length: int,
        forecast_length: int,
        model: nn.Module,
        dataset_name: str,
        channel_name,
        ds_mode,
        base
):
    number_of_epochs: int = 400
    base_lr: float = 0.01
    batch_size: int = 64
    valid_set_size: float = 0.1
    patience: int = 10000
    device = torch.device("cuda:0")

    #################################################################################
    os.chdir(os.path.join(os.path.dirname(__file__), "", ".."))
    dataset = DatasetInputData.create(dataset_name)

    experiment_subfolder_path = os.path.join(base, f"{str(model)},input_length={input_length},dataset={dataset_name}",
                                             channel_name)
    os.makedirs(experiment_subfolder_path, exist_ok=True)
    if os.path.exists(os.path.join(experiment_subfolder_path, "model.pt")):
        return

    print(experiment_subfolder_path)

    chname, columns, train_channel = dataset.channel(channel_name)
    if chname is None:
        return
    #################################################################################
    train_dataset = prepare_dataset(
        train_channel, input_length, forecast_length, ds_mode
    )

    valid_samples_count = int(len(train_dataset) * valid_set_size)
    train_subset, valid_subset = data.random_split(
        train_dataset,
        [len(train_dataset) - valid_samples_count, valid_samples_count],
        generator=torch.Generator().manual_seed(42),
    )
    #################################################################################

    best_model_state, train_logs = perform_training(
        model,
        device,
        base_lr,
        number_of_epochs,
        patience,
        batch_size,
        train_subset,
        valid_subset
    )

    dumps_file(os.path.join(experiment_subfolder_path, "summary.json"),
               {
                   "train_loss": min(train_logs["train_loss"]),
                   "valid_loss": min(train_logs["valid_loss"]),
                   "avg_epoch_time": np.mean(train_logs["epoch_times"]),
               }
               )

    dumps_file(os.path.join(experiment_subfolder_path, "losses.json"), train_logs)

    with open(os.path.join(experiment_subfolder_path, "losses.json"), "w") as losses_file:
        json.dump(train_logs, losses_file, indent=4)

    model.load_state_dict(best_model_state)

    torch.save(model, os.path.join(experiment_subfolder_path, "model.pt"))


######################


def cobot_202210():
    for input_length in [8]:#, 16, 32, 64]:#
        for dataset, subset in [
            (Datasets.CoBot202210, ["train"]),
        ]:
            for model, params in [
                # (models.LSTM, {"n_layers": 1}),
                # (models.LSTM, {"n_layers": 2}),
                # (models.GRU, {"n_layers": 1}),
                # (models.GRU, {"n_layers": 2}),
                (models.SCINet2, {})
            ]:
                yield input_length, 10, dataset, subset, model, params

class MThread:
    def __init__(self, feature_count, forecast_length, params, model_fun, dataset_name, channel, ds_mode, input_length, base):
        self.feature_count = feature_count
        self.forecast_length = forecast_length
        self.params = params
        self.model_fun = model_fun
        self.dataset_name = dataset_name
        self.channel = channel
        self.ds_mode = ds_mode
        self.input_length = input_length
        self.base = base

    def run(self) -> None:
        # print('xxx')
        model = self.model_fun(features_count=self.feature_count, forecast_length=self.forecast_length, window_length=self.input_length, **self.params)
        # model = self.model_fun(features_count=self.feature_count, forecast_length=self.forecast_length, **self.params)
        # model.__setattr__("input_length", self.input_length)
        process(self.input_length, self.forecast_length, model, self.dataset_name, self.channel, self.ds_mode, self.base)


if __name__ == "__main__":
    threads = []

    to_be_processed = list(cobot_202210())
    ds_mode = DsMode.WITH_MPC
    base = "c:\\experiments\\cobot_2023_scinet2\\"

    for input_length, forecast_length, dataset_name, subset, model_fun, params in to_be_processed:
        dataset = DatasetInputData.create(dataset_name)
        _, cols, _ = dataset.channel(dataset.channel_names()[0])
        feature_count = len(cols) # HERE another one switch - 1
        if subset is None:
            channels = dataset.channel_names()
        else:
            channels = subset
        for ch in channels:
            threads.append(MThread(
                feature_count, forecast_length, params, model_fun, dataset_name, ch, ds_mode, input_length, base
            ))

    threads = sorted(threads, key=lambda x: x.dataset_name)

    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, t in enumerate(threads):
            f = ex.submit(t.run)
