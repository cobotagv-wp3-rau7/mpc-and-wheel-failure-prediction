import numpy
import os
import numpy as np

from training_v_cobot import process
from cobot_ml.data.utilities import DsMode
from cobot_ml.models import LSTM
import cobot_ml.data.datasets as dss
from cobot_ml.data.datasets import DatasetInputData
from generate_test import run_prediction
from cobot_ml.utilities import dumps_file

import uuid
import csv

base_dir = "a:\\202303_experiments_baseline"


def fitness_func(solution, idx):
    global base_dir
    model = LSTM(features_count=len(solution), n_layers=2, forecast_length=10)
    os.chdir(os.path.join(os.path.dirname(__file__), "", ".."))

    dataset: dss.CoBot202210Data = DatasetInputData.create(dss.Datasets.CoBot202210)
    dataset.minmax()
    dataset.apply_weights(solution)

    folder = os.path.join(base_dir, f"{idx}_{str(uuid.uuid4())}")

    subfolder = process(
        input_length=5,
        forecast_length=10,
        model=model,
        dataset=dataset,
        dataset_name=dss.Datasets.CoBot202210,
        channel_name="train",
        ds_mode=DsMode.WITH_MPC,
        base=folder
    )
    dumps_file(os.path.join(subfolder, "weights.json"), solution)
    metrics = run_prediction(
        ds_mode=DsMode.WITH_MPC,
        output_path=subfolder,
        model_path=os.path.join(subfolder, "model.pt"),
        input_steps=5,
        output_steps=10,
        dataset=dataset,
        subset=["test"],
        batch_size=64,
        plot=True
    )
    fitness = metrics["test"]["all_mse"]
    with open(f"{base_dir}\\fitnesses.csv", 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([subfolder, idx, -fitness])
    return -fitness


if __name__ == '__main__':
    dataset = DatasetInputData.create(dss.Datasets.CoBot202210)

    weights = []
    for c in dataset.columns:
        c_type = dataset.metadata[c]["type"]
        if c_type == "bool":
            weights.append(1)
        else:
            weights.append(abs(dataset.mpc_correlations[c]))
    # weights = np.array(weights)
    weights = np.ones(shape=(len(dataset.columns),))

    fitness_func(weights, 0)
