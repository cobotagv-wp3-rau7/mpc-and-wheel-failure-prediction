import glob
import os
import typing
import json
import time

import natsort
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from cobot_ml import detectors
from cobot_ml.data import datasets, patchers
from cobot_ml.data.datasets import DatasetInputData
from cobot_ml.training import runners
from cobot_ml.evaluation import forecasting_metrics as fm


def dumps_file(path, _object):
    with open(path, "w") as params_file:
        json.dump(_object, params_file, indent=4)


def plot_results(
        title: str,
        real_values: np.ndarray,
        predictions: np.ndarray,
        save_path: str,
) -> None:
    import plotly.express as px
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
) -> datasets.TensorPairsDataset:
    channel_values = detectors.pad_beginning(channel_values, input_steps)
    patches = patchers.patch_with_stride(
        channel_values, input_steps + output_steps, stride=1
    )
    patches = [torch.from_numpy(patch.astype(np.float32)) for patch in patches]
    X = [patch[:input_steps, 1:] for patch in patches]
    y = [patch[input_steps:, 0] for patch in patches]
    return datasets.TensorPairsDataset(X, y)


def run_prediction(output_path: str,
                   model_path: str,
                   input_steps: int,
                   output_steps: int,
                   dataset_name: str,
                   subset,
                   batch_size: int):
    device = torch.device("cuda:0")
    model = torch.load(model_path, map_location=device)

    dataset_ = DatasetInputData.create(dataset_name)

    metrics = dict()

    for channel_name in dataset_.channel_names():
        name, cols, chdata = dataset_.channel(channel_name)
        if name is None or name not in subset:
            continue
        ds = prepare_dataset(chdata, input_steps, output_steps)

        beg = time.time()
        predictions = runners.run_inference(
            model,
            data.DataLoader(ds, batch_size=batch_size),
            device=device,
        )
        fin = time.time()

        y_pred = unravel_vector(predictions)[input_steps:]
        real_values = np.array(ds.get_unraveled_targets())[input_steps:]

        np.save(
            os.path.join(output_path, f"{channel_name}_predictions.npy"), y_pred
        )
        # plot_results(
        #     f"{channel_name}_predictions.png",
        #     real_values,
        #     y_pred,
        #     os.path.join(output_path, f"{channel_name}_predictions.png"),
        # )
        if channel_name not in metrics:
            metrics[channel_name] = dict()

        metrics[channel_name]["mse"] = float(fm.mse(real_values, y_pred))
        metrics[channel_name]["mae"] = float(fm.mae(real_values, y_pred))
        metrics[channel_name]["mape"] = float(fm.mape(real_values, y_pred))
        metrics[channel_name]["smape"] = float(fm.smape(real_values, y_pred))
        metrics[channel_name]["timing"] = float(fin - beg)

    dumps_file(os.path.join(output_path, f"metrics.json"), metrics)


dd = dict()
i = 0

for experiment_path in natsort.natsorted(glob.glob("/mnt/cloud/20220201_experiments/*")):

    model_path = "/dev/null"
    subset = []
    for channel_path in natsort.natsorted(glob.glob(os.path.join(experiment_path, "*"))):
        model_path = os.path.join(channel_path, "model.pt")
        if os.path.exists(model_path):
            subset.append(os.path.basename(channel_path))

    for channel_path in natsort.natsorted(glob.glob(os.path.join(experiment_path, "*"))):
        model_path = os.path.join(channel_path, "model.pt")
        if not os.path.exists(model_path):
            continue

        exp_params = dict()
        for item in os.path.basename(experiment_path).split(","):
            k, v = item.split("=")
            exp_params[k] = v

        output_path = os.path.join(experiment_path, f"_test_{os.path.basename(channel_path)}")
        os.makedirs(output_path, exist_ok=True)

        run_prediction(
            output_path,
            model_path,
            input_steps=int(exp_params["input_length"]),
            output_steps=int(exp_params["forecast"]),
            dataset_name=exp_params["dataset"],
            subset=set(subset),
            batch_size=64
        )
