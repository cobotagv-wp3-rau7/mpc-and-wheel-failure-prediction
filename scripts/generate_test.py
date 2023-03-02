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
        mode
) -> datasets.TensorPairsDataset:
    channel_values = detectors.pad_beginning(channel_values, input_steps)
    patches = patchers.patch_with_stride(
        channel_values, input_steps + output_steps, stride=1
    )
    patches = [torch.from_numpy(patch.astype(np.float32)) for patch in patches]
    if mode == "uni":
        X = [patch[:input_steps, [0]] for patch in patches]
    elif mode == "with_mpc":
        X = [patch[:input_steps, :] for patch in patches]
    elif mode == "wo_mpc":
        X = [patch[:input_steps, 1:] for patch in patches]
    y = [patch[input_steps:, 0] for patch in patches]
    return datasets.TensorPairsDataset(X, y)


def run_prediction(mode, output_path: str,
                   model_path: str,
                   input_steps: int,
                   output_steps: int,
                   dataset_name: str,
                   subset,
                   batch_size: int):
    print(model_path)
    device = torch.device("cuda:0")
    model = torch.load(model_path, map_location=device)

    dataset_ = DatasetInputData.create(dataset_name)

    metrics = dict()

    for channel_name in dataset_.channel_names():
        name, cols, chdata = dataset_.channel(channel_name)
        if name is None or name not in subset:
            continue
        ds = prepare_dataset(chdata, input_steps, output_steps, mode)

        beg = time.time()
        predictions_file_path = os.path.join(output_path, f"{channel_name}_predictions_complete.npy")
        if not os.path.exists(predictions_file_path):
            predictions = runners.run_inference(
                model,
                data.DataLoader(ds, batch_size=batch_size),
                device=device,
            )
            preds_np = predictions.detach().cpu().numpy()
            np.save(predictions_file_path, preds_np)
        else:
            preds_np = np.load(predictions_file_path)
            predictions = torch.from_numpy(preds_np)

        fin = time.time()

        first_aes = []
        first_ses = []
        first_mapes = []
        first_smapes = []

        last_aes = []
        last_ses = []
        last_mapes = []
        last_smapes = []

        all_aes = []
        all_ses = []
        all_mapes = []
        all_smapes = []

        for i in range(preds_np.shape[0]):
            tar_np = ds.targets[i].detach().cpu().numpy()
            preds_i = preds_np[i]

            all_aes.append(abs(preds_i - tar_np))
            all_ses.append((preds_i - tar_np) ** 2)
            all_mapes.append(abs((tar_np - preds_i) / (tar_np + 0.000001)))
            all_smapes.append(abs(tar_np - preds_i) / ((tar_np + preds_i) / 2))

            first_aes.append(abs(preds_i[0] - tar_np[0]))
            first_ses.append((preds_i[0] - tar_np[0]) ** 2)
            first_mapes.append(abs((tar_np[0] - preds_i[0]) / (tar_np[0] + 0.000001)))
            first_smapes.append(abs(tar_np[0] - preds_i[0]) / ((tar_np[0] + preds_i[0]) / 2))

            last_aes.append(abs(preds_i[-1] - tar_np[-1]))
            last_ses.append((preds_i[-1] - tar_np[-1]) ** 2)
            last_mapes.append(abs((tar_np[-1] - preds_i[-1]) / (tar_np[-1] + 0.000001)))
            last_smapes.append(abs(tar_np[-1] - preds_i[-1]) / ((tar_np[-1] + preds_i[-1]) / 2))

        if channel_name not in metrics:
            metrics[channel_name] = dict()

        metrics[channel_name]["all_mae"] = float(np.mean(all_aes))
        metrics[channel_name]["all_mse"] = float(np.mean(all_ses))
        metrics[channel_name]["all_mape"] = float(np.mean(all_mapes))
        metrics[channel_name]["all_smape"] = float(np.mean(all_smapes))

        metrics[channel_name]["first_mae"] = float(np.mean(first_aes))
        metrics[channel_name]["first_mse"] = float(np.mean(first_ses))
        metrics[channel_name]["first_mape"] = float(np.mean(first_mapes))
        metrics[channel_name]["first_smape"] = float(np.mean(first_smapes))

        metrics[channel_name]["last_mae"] = float(np.mean(last_aes))
        metrics[channel_name]["last_mse"] = float(np.mean(last_ses))
        metrics[channel_name]["last_mape"] = float(np.mean(last_mapes))
        metrics[channel_name]["last_smape"] = float(np.mean(last_smapes))

        y_pred = torch.cat([predictions[:-1, 0], predictions[-1, :]]).detach().cpu().numpy()[input_steps:]
        # y_pred = torch.cat([predictions[:-1, 0, 0], predictions[-1, :, 0]]).detach().cpu().numpy()[input_steps:]
        np.save(
            os.path.join(output_path, f"{channel_name}_predictions.npy"), y_pred
        )

        # real_values = [ds.targets[idx][0][0] for idx in range(len(ds.targets) - 1)]
        # real_values = np.array(torch.hstack([torch.tensor(real_values), ds.targets[-1][:,0]]))[input_steps:]
        real_values = np.array(ds.get_unraveled_targets())[input_steps:]

        plot_results(
            f"{channel_name}_predictions.png",
            real_values,
            y_pred,
            os.path.join(output_path, f"{channel_name}_predictions.png"),
        )


        metrics[channel_name]["mse"] = float(fm.mse(real_values, y_pred))
        metrics[channel_name]["mae"] = float(fm.mae(real_values, y_pred))
        metrics[channel_name]["mape"] = float(fm.mape(real_values, y_pred))
        metrics[channel_name]["smape"] = float(fm.smape(real_values, y_pred))
        metrics[channel_name]["timing"] = float(fin - beg)

    dumps_file(os.path.join(output_path, f"metrics.json"), metrics)


dd = dict()
i = 0

# name, cols, chdata = dataset_.channel(channel_name)

for exps, mode in [
    ("cobot_2023_multivariate_with_MPC", "with_mpc"),
    ("cobot_2023_multivariate_wo_MPC", "wo_mpc"),
    ("cobot_2023_univariate", "uni"),
    ("cobot_2023_weighted_multivariate_with_MPC", "with_mpc"),
    ("cobot_2023_weighted_multivariate_wo_MPC", "wo_mpc")
]:

    for experiment_path in natsort.natsorted(glob.glob(f"c:\\experiments\\{exps}\\*")):
        model_path = "/dev/null"
        subset = ["test"]
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
                mode,
                output_path,
                model_path,
                input_steps=int(exp_params["input_length"]),
                output_steps=int(exp_params["forecast"]),
                dataset_name=exp_params["dataset"],
                subset=set(subset),
                batch_size=64
            )
