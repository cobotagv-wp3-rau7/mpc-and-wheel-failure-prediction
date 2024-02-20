import json
import os
import sys

import torch
from torch import nn
from torch.utils import data

from cobot_ml.utilities import dumps_file
import cobot_ml.training.trainer as trn

sys.path.append(os.getcwd())

from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd


def the_dataset():
    selected_columns = [
        "FH.6000.[ENS] - Energy Signals.Momentary power consumption",
        "FH.6000.[ENS] - Energy Signals.Battery cell voltage",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - safety interlock",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - automatic permission",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - command on",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - executed",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - in progress",
        "FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.ActualSpeed_L",
        "FH.6000.[G2PAS] GROUP 2 - PIN ACTUATOR SIGNALS.Pin Up - safety interlock",
        "FH.6000.[G2PAS] GROUP 2 - PIN ACTUATOR SIGNALS.Pin Up - automatic permission",
        "FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - safety interlock",
        "FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - automatic permission",
        "FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission",
        "FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - command on",
        "FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.ActualSpeed_R",
        "FH.6000.[GS] GENERAL SIGNALS.Manual Mode active",
        "FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active",
        "FH.6000.[GS] GENERAL SIGNALS.PLC fault active",
        "FH.6000.[GS] GENERAL SIGNALS.PLC warning Active",
        "FH.6000.[LED] LED STATUS.LED RGB Strip 1 left - R",
        "FH.6000.[LED] LED STATUS.LED RGB Strip 2 right - R",
        "FH.6000.[LED] LED STATUS.LED RGB Strip 1 left - G",
        "FH.6000.[LED] LED STATUS.LED RGB Strip 2 right - G",
        "FH.6000.[LED] LED STATUS.LED RGB Strip 1 left - B",
        "FH.6000.[LED] LED STATUS.LED RGB Strip 2 right - B",
        "FH.6000.[LED] LED STATUS.LED status - active mode",
        "FH.6000.[NNCF]3105 - Go to destination result.Destination ID",
        "FH.6000.[NNCF]3105 - Go to destination result.Go to result",
        "FH.6000.[NNCF]3106 - Pause drive result.Pause result",
        "FH.6000.[NNCF]3107 - Resume drive result.Destination ID",
        "FH.6000.[NNCF]3107 - Resume drive result.Resume result",
        "FH.6000.[NNCF]3108 - Abort drive result.Abort result",
        "FH.6000.[NNS] - Natural Navigation Signals.Natural Navigation status",
        "FH.6000.[NNS] - Natural Navigation Signals.Error status",
        "FH.6000.[NNS] - Natural Navigation Signals.Natural Navigation state",
        "FH.6000.[NNS] - Natural Navigation Signals.X-coordinate",
        "FH.6000.[NNS] - Natural Navigation Signals.Y-coordinate",
        "FH.6000.[NNS] - Natural Navigation Signals.Heading",
        "FH.6000.[NNS] - Natural Navigation Signals.Position confidence",
        "FH.6000.[NNS] - Natural Navigation Signals.Speed",
        "FH.6000.[NNS] - Natural Navigation Signals.Going to ID",
        "FH.6000.[NNS] - Natural Navigation Signals.Target reached",
        "FH.6000.[NNS] - Natural Navigation Signals.Current segment",
        "FH.6000.[ODS] - Odometry Signals.Momentary frequency of left encoder pulses",
        "FH.6000.[ODS] - Odometry Signals.Momentary frequency of right encoder pulses",
        "FH.6000.[ODS] - Odometry Signals.Cumulative distance left",
        "FH.6000.[ODS] - Odometry Signals.Cumulative distance right",
        "FH.6000.[SS] SAFETY SIGNALS.Safety circuit closed",
        "FH.6000.[SS] SAFETY SIGNALS.Scanners muted",
        "FH.6000.[SS] SAFETY SIGNALS.Front bumper triggered",
        "FH.6000.[SS] SAFETY SIGNALS.Front scanner safety zone violated",
        "FH.6000.[SS] SAFETY SIGNALS.Rear scanner safety zone violated",
        "FH.6000.[SS] SAFETY SIGNALS.Front scanner warning zone violated",
        "FH.6000.[SS] SAFETY SIGNALS.Rear scanner warning zone violated",
        "FH.6000.[SS] SAFETY SIGNALS.Scanners active zones",
    ]

    # Read the CSV file
    df = pd.read_csv('C:\\projekty\\cobot_with_weight\\concatenated_202210.csv_changing_columns.csv')
    df = df.replace({True: 1, False: 0})

    # df['labels'] = df['payload_weight'].apply(lambda x: 0 if (max_payload_excl > x >= min_payload_incl) else 1)

    df = df.dropna()
    # labels = df['labels']
    # df = df.drop(['labels'], axis=1)
    payload_weight = df['payload_weight']
    df = df[selected_columns]

    return df, payload_weight


def split_dataset(bins, labels):
    the_data, payload_weight = the_dataset()
    scaler = StandardScaler()
    scaler.fit(the_data)

    range = pd.cut(payload_weight, bins=bins, labels=labels, include_lowest=True)
    range_dict = {}
    for range_label in labels:
        range_dict[range_label] = scaler.transform(the_data[range == range_label])

    return range_dict



def process(
        input_length: int,
        forecast_length: int,
        model: nn.Module,
        descr: str,
        base="C:\\projekty\\cobot_with_weight\\"
):
    number_of_epochs: int = 200
    base_lr: float = 0.01
    batch_size: int = 64
    valid_set_size: float = 0.1
    patience: int = 10000
    device = torch.device("cuda:0")

    #################################################################################
    # bins = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 440, 480, 500]
    bins = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 500]
    step = 40
    # bins = [0, 300, 340, 380, 420, 460, 500]
    labels = [f'{bins[i]}-{bins[i + 1]}' for i in range(len(bins) - 1)]

    experiment_subfolder_path = os.path.join(base, f"{descr}_{str(model)},input_length={input_length}")
    os.makedirs(experiment_subfolder_path, exist_ok=True)
    # if os.path.exists(os.path.join(experiment_subfolder_path, "model.pt")):
        # return experiment_subfolder_path

    print(experiment_subfolder_path)

    input_data = split_dataset(bins, labels)

    the_boundary = 200
    train_data = [input_data[f"{x}-{x+step}"] for x in range(0, the_boundary, 2*step)]
    normal_test_data = [input_data[f"{x}-{x+step}"] for x in range(step, the_boundary, 2*step)]

    normal_test_dataset = data.ConcatDataset([trn.prepare_dataset(ntd, input_length, forecast_length) for ntd in normal_test_data])
    train_dataset = data.ConcatDataset([trn.prepare_dataset(td, input_length, forecast_length) for td in train_data])


    valid_samples_count = int(len(train_dataset) * valid_set_size)
    train_subset, valid_subset = data.random_split(
        train_dataset,
        [len(train_dataset) - valid_samples_count, valid_samples_count],
        generator=torch.Generator().manual_seed(42),
    )
    #################################################################################

    best_model_state, train_logs = trn.perform_training(
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

    ##################
    def proceed_with_test(model, the_dataset):
        import cobot_ml.training.runners as rnrs
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

        the_loader = data.DataLoader(the_dataset, batch_size=batch_size)
        results = rnrs.run_inference(model, the_loader, device=device)
        if isinstance(the_dataset, data.ConcatDataset):
            targets = [item for ds in the_dataset.datasets for item in ds.targets]
        else:
            targets = the_dataset.targets


        res_np = results.cpu().numpy()
        tar_np = torch.stack(targets).numpy()

        assert res_np.shape == tar_np.shape, "The dimensions of predictions and target must be the same."
        f_res = {
            "first_mse": mean_squared_error(tar_np[:, 0], res_np[:, 0]),
            "first_mae": mean_absolute_error(tar_np[:, 0], res_np[:, 0]),
            "first_mape": mean_absolute_percentage_error(tar_np[:, 0], res_np[:, 0]),
            "last_mse": mean_squared_error(tar_np[:, -1], res_np[:, -1]),
            "last_mae": mean_absolute_error(tar_np[:, -1], res_np[:, -1]),
            "last_mape": mean_absolute_percentage_error(tar_np[:, -1], res_np[:, -1]),
            "mse": mean_squared_error(tar_np, res_np),
            "mae": mean_absolute_error(tar_np, res_np),
            "mape": mean_absolute_percentage_error(tar_np, res_np),
        }
        return res_np, tar_np, f_res


    y_pred, y_target, metrics = proceed_with_test(model, normal_test_dataset)
    print(f"normal test: {metrics}")
    for rng, fragment in input_data.items():
        rng_dataset = trn.prepare_dataset(fragment, input_length, forecast_length)
        y_pred, y_target, metrics = proceed_with_test(model, rng_dataset)
        print(f"{rng}: {metrics}")

    return experiment_subfolder_path


from cobot_ml import models

if __name__ == "__main__":
    forecast_length = 10
    history_length = 50
    features_count = 55
    # process(history_length, forecast_length, models.LSTM(features_count, n_layers=2, forecast_length=forecast_length), descr="normal_up_to_300")
    process(history_length, forecast_length, models.GRU(features_count, n_layers=2, forecast_length=forecast_length), descr="normal_up_to_200")

