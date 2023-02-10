import os
import pickle
import sys
import tempfile
import typing

import fire
import mlflow
import numpy as np
import torch

sys.path.append(os.getcwd())

from cobot_ml.data.datasets import DatasetInputData

DEFAULT_FFT_SIZE = 8192

def plot_results(
        title: str,
        real_values: np.ndarray,
        predictions: np.ndarray,
        save_path: str,
        anomaly_scores: np.ndarray,
        real_anomaly_ranges: typing.Iterable[typing.Tuple[int, int]] = None,
        predicted_anomaly_ranges: typing.Iterable[typing.Tuple[int, int]] = None,
) -> None:
    import plotly.express as px
    import pandas as pd
    fig = px.line(pd.DataFrame(
        dict(
            real=real_values[:, 0],
            predicted=predictions,
            anomaly_score=anomaly_scores
        )
    ))
    fig.write_html(f"{save_path}.html")

    # plt.figure(figsize=(11, 7))
    # plt.title(title)
    # plt.plot(real_values[:,0], label="Real values")
    # plt.plot(predictions, label="Predictions")
    # plt.plot(anomaly_scores, label="Anomaly_score")
    # for anomaly_ranges, color in [
    #     (real_anomaly_ranges, "r"),
    #     (predicted_anomaly_ranges, "b"),
    # ]:
    #     if anomaly_ranges is not None:
    #         *_, ymin, ymax = plt.axis()
    #         rectangle_height = sum([abs(x) for x in [ymin, ymax]])
    #         for anomaly_start_stop in anomaly_ranges:
    #             plt.gca().add_patch(
    #                 Rectangle(
    #                     (anomaly_start_stop[0], ymin),
    #                     anomaly_start_stop[1] - anomaly_start_stop[0],
    #                     rectangle_height,
    #                     alpha=0.5,
    #                     color=color,
    #                 )
    #             )
    # plt.legend(loc="upper left")
    # plt.savefig(save_path)
    # plt.close()


def unravel_vector(vector: torch.Tensor) -> np.ndarray:
    """
    Extract first value from each sample in predicted vector (but all values
    from the last sample).
    """
    return torch.cat([vector[:-1, 0], vector[-1, :]]).detach().cpu().numpy()


def find_child_run(child_runs: typing.List, name: str):
    """
    Search for a child run with given name and return it
    :param child_runs: List of child runs
    :param name: Name of the child run to return
    :return: A child run with given name
    """
    for run in child_runs:
        if run.data.tags["mlflow.runName"] == name:
            return run


ieee_battery_run_ids = [
    "d477934a105c49e08c7626f08290983e",
    "b80cb6ead41f4f95875f1e639f0aaf27",
    "2323cbc06ac04228acb9dbf9853699ab",
    "5a8e111831c04475aa859c749d974030",
    "835a6b6015514d729b2049c145b2713a",
    "b7474a9ce6cc4fc6a7fa5d0f7d189347",
    "4ebf8a11560f48a5b7e42d487465c844",
    "e7021822cf504289aeab7f7af9ec7948",
    "99326c40428d4b2c96693aef5354ca2b",
    "b51243ffb70741b29573ae9febb52888",
    "fe8f6376ff9c4f899dc70662485e6571",
    "fa3b4cc245c24aad8af34294a00b0d63",
    "36ae07fabce843bd812dac48937d5b44",
    "aaf0bf903aa24bc2a094155f4a1a0e88",
    "1752586084194822a7be0972f603c997",
    "e977b5cbe59a493092fff698bccb8048",
    "d433d9001cc34fddac61d733a0a3340a",
    "7d7c5f7ad8da4278be72bc2f19868a4e",
    "de373657b75146c8baf0c954b26e32db",
    "bb50d55f209f4730a62252dd7633ebdc",
    "806d296068d340f98a27ca0d077f5854",
    "f3e3f503571644c09b7d43d4ab8a0e8a",
    "2d8998833eab42e2990eefc9c5106141",
    "f55422f2bac5432ea1ab23c418aca854",
]

cobot_run_ids = [
    "38e8d5fa54a34be58cd63e05e464b131",
    "4623256d8d9d4d70b04576dc4709b68a",
    "36f7367464b949f38f134752235c0eb7",
    "72334c1b8a4b413d86e69030209f8809",
    "44f2250660a14d9483ff1c56b64c6c23",
    "5714655cd15a4f07af55302865e57e7f",
    "57c16d56cee64ababded20bf3d8a3f15",
    "ab9dc433ef7545c49c207362719ac91c",
    "867c99e839504037829ee64e55bc73a2",
    "cf64bb7abc164325a4a0800d6742894c",
    "f44a172da50f4e4e8428429ed16affb7",
    "22dd5a85ff7b4647a66065abbfdb8f3b",
    "8ac8e05938a54d91815d95ad008eb486",
    "2950cb52019440348bf22725d8a88482",
    "6b69be5da9d74d888111075239d0bd1c",
    "5244d57480754ba3ab9a6359f86f051a",
    "61c084ea88d84d86a824019f24172023",
    "4982974ca4154ea0a1990d82f809a9be",
    "d043047e566d42a4babba62d0ca69130",
    "c323e768a25140889b0346d34fac69e4",
    "c42dbaa477dc4b63858390076bc27362",
    "19ddd1289ed848c4b7d7ee932c12dada",
    "7edf0181e2de462f9f35634e784995b2",
    "3d57e8bec0194241b6e01cbd323ff492",
    "51846b1547864e83b8cbc4bf1371269f",
    "0e639ebdd28f45caaf56a8b12e122695",
    "f390f30a1dd74a4c85839f2a49cacb09",
    "f74213b2b8314f0588bb8d2c4c79e3a5",
    "6e6b85f14f0d487caf54b2ad2432c377",
]

['TripA01', 'TripA02', 'TripA03', 'TripA04', 'TripA05', 'TripA06', 'TripA07',
 'TripA08', 'TripA09', 'TripA10', 'TripA11', 'TripA12', 'TripA13', 'TripA14',
 'TripA15', 'TripA16', 'TripA17', 'TripA18', 'TripA19', 'TripA20', 'TripA22',
 'TripA23', 'TripA24', 'TripA25', 'TripA26', 'TripA27', 'TripA28', 'TripA29',
 'TripA30', 'TripA31', 'TripA32', 'TripB01', 'TripB02', 'TripB03', 'TripB04',
 'TripB05', 'TripB06', 'TripB07', 'TripB08', 'TripB09', 'TripB10', 'TripB11',
 'TripB12', 'TripB13', 'TripB14', 'TripB15', 'TripB16', 'TripB17', 'TripB18',
 'TripB19', 'TripB20', 'TripB21', 'TripB22', 'TripB23', 'TripB24', 'TripB25',
 'TripB26', 'TripB27', 'TripB28', 'TripB29', 'TripB30', 'TripB31', 'TripB32',
 'TripB33', 'TripB34', 'TripB35', 'TripB36', 'TripB37']



def find_runs(run_ids):
    runs = mlflow.search_runs(experiment_ids=[3, 4])
    runs = runs.query("run_id in @run_ids")
    return runs


def main(
        # parent_experiment_id: int,
        # parent_run_name: str,
        deviations: int = 7,
        k: int = 2,
        input_steps: int = 250,
        output_steps: int = 10,
        # experiment_dataset: str = "eagleone",
        experiment_dataset: str = "ieee_battery",
        batch_size: int = 64,
        tracking_uri: str = "http://localhost:5000",
        # experiment_name: str = "20220111 EagleOne Predictions",
        experiment_name: str = "20220111 IEEE Battery Predictions",
        run_name: str = "default_run",
        device: str = "cuda:0",
):
    # dataset = DatasetInputData.create("husky")
    # columns_map = dict()
    # for channel_name, columns, train_channel, test_channel, y_true in dataset.channels(whole=True):
    #     for c in columns:
    #         if c not in columns_map:
    #             columns_map[c] = []
    #         columns_map[c].append(channel_name)
    # return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    dataset = DatasetInputData.create(experiment_dataset)
    _runs = find_runs(ieee_battery_run_ids)
    for _, row in _runs.iterrows():
        _parent_run = mlflow.get_run(row["tags.mlflow.parentRunId"])
        _run_name = _parent_run.data.params['run_name']
        input_steps = int(_parent_run.data.params['input_steps'])
        train_series = row["tags.mlflow.runName"]
        _dir = row["params.dir"]
        print("")

        # mlflow.log_params(locals())
        device = torch.device(device)
        # parent_run = mlflow_client.search_runs(
        #     parent_experiment_id,
        #     filter_string=f"tags.mlflow.runName='{parent_run_name}'",
        # )
        # child_runs = mlflow_client.search_runs(
        #     parent_experiment_id,
        #     filter_string=f'tags.mlflow.parentRunId = "{parent_run[0].info.run_id}"',
        #     max_results=1500,
        # )
        for channel_name, columns, train_channel, test_channel, y_true in dataset.channels(whole=True):
            assert train_channel.shape[1] == test_channel.shape[1]
            run_name = f"{{input_steps={input_steps}, train: \"{train_series}\", input:\"{channel_name}\"}}"
            print(run_name)
            with mlflow.start_run(run_name=run_name) as channel_run:
                temp_dir = tempfile.mkdtemp(dir="/home/pawel/mlflow_artifacts")
                model_path = os.path.join(_dir, "model.pt")
                mlflow.log_param("model_path", model_path)
                model = torch.load(
                    model_path,
                    map_location=torch.device(device),
                )

                test_channel_orig = test_channel
                test_channel = test_channel[:, 1:]
                train_channel = train_channel[:, 1:]

                detector = None
                model_predictions = detector.predictor.predict_signal_with_model(test_channel)[:, 0]
                diffs = model_predictions - test_channel_orig[:, 0]
                MSE = np.mean(diffs ** 2)
                np.save(os.path.join(temp_dir, "real.npy"), test_channel_orig)
                np.save(os.path.join(temp_dir, "predictions.npy"), model_predictions)
                np.save(os.path.join(temp_dir, "error.npy"), diffs)

                anomaly_scores = detector.predict(test_channel, batch_size=batch_size)
                np.save(os.path.join(temp_dir, "anomaly_scores.npy"), anomaly_scores)
                anomalies = np.zeros_like(anomaly_scores)
                anomalies[anomaly_scores >= detector.threshold] = 1
                np.save(os.path.join(temp_dir, "anomalies.npy"), anomalies)

                plot_results(
                    "test_predictions.png",
                    test_channel_orig,
                    model_predictions,
                    os.path.join(temp_dir, "test_vis.png"),
                    anomaly_scores,
                )
                mlflow.log_metrics(
                    {"k": detector.k, "x0": detector.x0, "thr": detector.threshold, "MSE": MSE}
                )
                with open(os.path.join(temp_dir, "detector_pickle"), "wb") as f:
                    pickle.dump(detector, f)
                mlflow.log_param("path", temp_dir)
                # mlflow.log_artifacts(temp_dir)
                # shutil.rmtree(temp_dir)
        print("r")


if __name__ == "__main__":
    fire.Fire(main)
