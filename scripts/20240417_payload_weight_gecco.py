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
from cobot_ml.utils import copy_files_and_dependencies

correlations = [1.0, 0.26245605971190045, 0.2753337340256664, 0.2563690190869329, 0.2561615648386402,
                0.6050273980282915, 0.6050273980282915, 0.6050273980282915, 0.01077562437937579, 0.2563871369463908,
                0.2563871369463908, 0.2753337340256664, 0.2563871369463908, 0.2561615648386402, 0.2563871369463908,
                0.001984775126317298, 0.2561615648386402, 0.2563690190869329, 0.1913337186004618, 0.131743875323736,
                0.19321485319821968, 0.1853564381016196, 0.23155297143676873, 0.21639167762717104, 0.02591316584698986,
                0.08266515272386872, 0.06601434318023364, 0.12716330611827814, 0.14745296196936572, 0.12716330611827814,
                0.12716330611827814, 0.14745296196936572, 0.12716330611827814, 0.2564011654873815, 0.25631435313361683,
                0.4349717871281959, 0.2872823495815461, 0.12839496804116027, 0.2740284688440543, 0.10353700056110185,
                0.011646741561676242, 0.12716330611827814, 0.14745296196936572, 0.11620610572349993,
                0.03251919051059216, 0.0003411174703626511, 0.0154933267026157, 0.017665332438558678,
                0.2753337340256664, 0.2561615648386402, 0.2572350863160418, 0.26936309051247154, 0.26151691880571,
                0.3069194208149784, 0.3239747482884918, 0.2561615648386402]


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
        # "payload_weight"
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


def split_dataset(bins, labels, weights):
    the_data, payload_weight = the_dataset()
    scaler = StandardScaler()
    scaler.fit(the_data)

    cors = the_data.corrwith(the_data["FH.6000.[ENS] - Energy Signals.Momentary power consumption"],
                             method='spearman').abs().values

    range = pd.cut(payload_weight, bins=bins, labels=labels, include_lowest=True)
    range_dict = {}
    for range_label in labels:
        range_dict[range_label] = scaler.transform(the_data[range == range_label]) * weights

    return range_dict


def process(
        input_length: int,
        forecast_length: int,
        model: nn.Module,
        descr: str,
        weights,
        folder_prefix: str,
        the_boundary: int = 200,
        base="E:\\experiments_gecco2024_normal_test_nearest_to_008_longer",
):
    number_of_epochs: int = 50
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

    experiment_subfolder_path = os.path.join(base, f"{folder_prefix}_{descr}_{str(model)},input_length={input_length}")
    os.makedirs(experiment_subfolder_path, exist_ok=True)
    experiment_code_path = os.path.join(experiment_subfolder_path, "code")
    copy_files_and_dependencies(__file__, experiment_code_path)
    # if os.path.exists(os.path.join(experiment_subfolder_path, "model.pt")):
    # return experiment_subfolder_path

    print(experiment_subfolder_path)

    input_data = split_dataset(bins, labels, weights)

    train_data = [input_data[f"{x}-{x + step}"] for x in range(0, the_boundary, 2 * step)]
    normal_test_data = [input_data[f"{x}-{x + step}"] for x in range(step, the_boundary, 2 * step)]

    normal_test_dataset = data.ConcatDataset(
        [trn.prepare_dataset(ntd, input_length, forecast_length) for ntd in normal_test_data])
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
    np.save(os.path.join(experiment_subfolder_path, f"y_pred_normal_test.npy"), y_pred)
    np.save(os.path.join(experiment_subfolder_path, f"y_target_normal_test.npy"), y_target)
    metrics['name'] = "normal test"

    all_metrics = [metrics]
    print(f"normal test: {metrics}")
    for rng, fragment in input_data.items():
        rng_dataset = trn.prepare_dataset(fragment, input_length, forecast_length)
        y_pred, y_target, metrics = proceed_with_test(model, rng_dataset)
        np.save(os.path.join(experiment_subfolder_path, f"y_pred_{rng}.npy"), y_pred)
        np.save(os.path.join(experiment_subfolder_path, f"y_target_{rng}.npy"), y_target)
        metrics['name'] = rng
        all_metrics.append(metrics)
        print(f"{rng}: {metrics}")

    pd.DataFrame(all_metrics).to_csv(os.path.join(experiment_subfolder_path, "metrics.csv"))

    return experiment_subfolder_path, all_metrics


from cobot_ml import models

import hashlib
from deap import base, creator, tools, algorithms
import random
import multiprocessing


def evaluate(individual):
    model = models.LSTM(56, n_layers=2, forecast_length=10)
    hasher = hashlib.sha1()
    hasher.update(individual.tobytes())
    the_prefix = hasher.hexdigest()
    subfolder, all_metrics = process(50, 10, model, the_boundary=200,
                                     descr=f"fake_anomalies_with_MPC_with_weight_normal_up_to_200", weights=individual,
                                     folder_prefix=the_prefix)

    np.save(os.path.join(subfolder, "_weights.npy"), individual)

    fixed_normal_mse = 0.08
    normal_mse = all_metrics[0]['mse']

    the_normal_distance = abs(fixed_normal_mse - normal_mse)
    anomalous_mse = np.mean([m['mse'] for m in all_metrics[6:]])

    return the_normal_distance, anomalous_mse


def main():
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))  # min, max
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=56)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    # Zrównoleglanie oceny
    pool = multiprocessing.Pool(4)
    toolbox.register("map", pool.map)

    # Uruchomienie algorytmu
    population = toolbox.population(n=10)
    hof = tools.HallOfFame(1, similar=np.array_equal)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats,
                                              halloffame=hof)

    print(population)
    print(logbook)
    print([p.fitness.values for p in population])


if __name__ == "__main__":
    # evaluate(np.array(correlations))
   main()
