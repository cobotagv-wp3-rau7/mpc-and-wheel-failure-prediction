import json
import time
import typing

import evaluation.forecasting_metrics as fm
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from cobot_ml import detectors
from cobot_ml.data import datasets, patchers
from cobot_ml.data.datasets import DatasetInputData
from cobot_ml.training import runners


def dumps_file(path, _object):
    with open(path, "w") as params_file:
        json.dump(_object, params_file, indent=4)


def plot_results(
        title: str,
        real_values: np.ndarray,
        predictions: np.ndarray,
) -> None:
    import plotly.express as px
    fig = px.line(pd.DataFrame(dict(
        real=real_values,
        predictions=predictions,
        error=np.abs(predictions - real_values))))
    fig.show()


def unravel_vector(vector: torch.Tensor) -> np.ndarray:
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


def run_prediction(model_path: str,
                   input_steps: int,
                   output_steps: int,
                   dataset_name: str,
                   channel_name: str,
                   batch_size: int):
    device = torch.device("cpu")
    model = torch.load(model_path, map_location=device)

    dataset_ = DatasetInputData.create(dataset_name)

    metrics = dict()

    name, cols, chdata = dataset_.channel(channel_name)
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

    plot_results(
        channel_name,
        real_values,
        y_pred,
    )
    if channel_name not in metrics:
        metrics[channel_name] = dict()

    metrics[channel_name]["mse"] = float(fm.mse(real_values, y_pred))
    metrics[channel_name]["mae"] = float(fm.mae(real_values, y_pred))
    metrics[channel_name]["mape"] = float(fm.mape(real_values, y_pred))
    metrics[channel_name]["smape"] = float(fm.smape(real_values, y_pred))
    metrics[channel_name]["timing"] = float(fin - beg)

    return metrics
    #dumps_file(os.path.join(output_path, f"metrics.json"), metrics)