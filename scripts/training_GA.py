import csv
import os
import random
import uuid

import numpy
import numpy as np
import pygad

import cobot_ml.data.datasets as dss
from cobot_ml.data.datasets import DatasetInputData
from cobot_ml.data.utilities import DsMode
from cobot_ml.models import LSTM
from cobot_ml.utilities import dumps_file
from generate_test import run_prediction
from training_v_cobot import process


def fitness_func(solution, idx):
    global base_dir
    model = LSTM(features_count=len(solution), n_layers=2, forecast_length=10)

    os.chdir(os.path.join(os.path.dirname(__file__), "", ".."))
    dataset: dss.CoBot202210Data = DatasetInputData.create(dss.Datasets.CoBot202210, weights=solution)
    dataset.minmax()
    dataset.apply_weights(solution)

    folder = os.path.join(base_dir, f"{idx}_{str(uuid.uuid4())}")

    subfolder = process(
        input_length=5,
        forecast_length=10,
        model=model,
        dataset_name=dss.Datasets.CoBot202210,
        dataset=dataset,
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
    with open(os.path.join(base_dir, "fitnesses.csv"), 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([subfolder, idx, -fitness])
    return -fitness


def on_generation(ga: pygad.GA):
    import json
    from json import JSONEncoder
    global base_dir
    tobedumped = {
        "best_solutions": ga.best_solutions,
        "best_solution_fitness": ga.best_solutions_fitness,
        "solutions": ga.solutions,
        "solutions_fitness": ga.solutions_fitness,
    }

    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)

    with open(f"{base_dir}\\results_{ga.generations_completed}.json", 'w') as jsondump:
        json.dump(tobedumped, jsondump, indent=3, cls=NumpyArrayEncoder)


def bounds_numeric_0_to_1_bools_0_or_1(dataset):
    var_bounds = []
    for c in dataset.columns:
        c_type = dataset.metadata[c]["type"]
        if c_type == "bool":
            var_bounds.append([0, 1])
        else:
            var_bounds.append({'low': 0.0, 'high': 1.0})
    return var_bounds


def bounds_numeric_0_to_1_bools_0_to_1(dataset):
    var_bounds = []
    for _ in dataset.columns:
        var_bounds.append({'low': 0.0, 'high': 1.0})
    return var_bounds


def prepare_around_pearson_var_bounds(around_size: float, flopping_bools: bool = True):
    dataset: dss.CoBot202210Data = DatasetInputData.create(dss.Datasets.CoBot202210)
    var_bounds = []
    for idx, c in enumerate(dataset.columns):
        c_type = dataset.metadata[c]["type"]
        if c_type == "bool" and flopping_bools:
            var_bounds.append([0, 1])
        else:
            corr = abs(dataset.mpc_correlations[idx])
            low = max(0.0, corr - around_size)
            high = min(1.0, corr + around_size)
            var_bounds.append({'low': low, 'high': high})
    return var_bounds


def init_population_numeric_pearson_bool_pearson(dataset, epsilon):
    initial_population = []
    for i in range(10):
        solution = []
        for idx, c in enumerate(dataset.columns):
            corr = abs(dataset.mpc_correlations[idx])
            deviation = 2 * epsilon * random.random()
            solution.append(abs(corr) - epsilon + deviation)
        initial_population.append(solution)
    return np.clip(np.array(initial_population), 0.0, 1.0)


def init_population_numeric_pearson_gaussian_0dot1_bool_pearson(dataset, epsilon):
    initial_population = []
    for i in range(10):
        solution = []
        for idx, c in enumerate(dataset.columns):
            corr = abs(dataset.mpc_correlations[idx])
            solution.append(random.gauss(abs(corr), epsilon))
        initial_population.append(solution)
    return np.clip(np.array(initial_population), 0.0, 1.0)


base_dir = "a:\\202303_experiments_start_Pearson_gaussian_0.2_numeric[0-1]_bools[0,1]_3"
# base_dir = "a:\\202303_experiments_start_Pearson_plusminus_0.2_numeric[0-1]_bools[0,1]_2"

if __name__ == '__main__':
    os.makedirs(base_dir, exist_ok=True)

    dataset = DatasetInputData.create(dss.Datasets.CoBot202210)

    # initial_population = init_population_numeric_pearson_bool_pearson(dataset, 0.2)
    initial_population = init_population_numeric_pearson_gaussian_0dot1_bool_pearson(dataset, 0.2)
    var_bounds = bounds_numeric_0_to_1_bools_0_or_1(dataset)

    ga_instance = pygad.GA(
        initial_population=initial_population,
        num_generations=50,
        sol_per_pop=10,
        fitness_func=fitness_func,
        num_genes=len(var_bounds),
        num_parents_mating=4,
        parallel_processing=["process", 6],
        save_solutions=True,
        save_best_solutions=True,
        gene_space=var_bounds,
        on_generation=on_generation,
        crossover_probability=0.8,
        mutation_probability=0.15,
    )

    ga_instance.run()
