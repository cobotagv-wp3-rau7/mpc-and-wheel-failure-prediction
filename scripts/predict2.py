import os
import sys
import tempfile
import typing

import fire
import mlflow
import numpy as np
import torch

sys.path.append(os.getcwd())

from cobot_ml.data.datasets2 import DatasetInputData
from cobot_ml.detectors import Predictor


def plot_results(
        title: str,
        real_values: np.ndarray,
        predictions: np.ndarray,
        save_path: str,
) -> None:
    import plotly.express as px
    import pandas as pd
    fig = px.line(pd.DataFrame(
        dict(
            real=real_values[:, 0],
            predicted=predictions,
        )
    ))
    fig.write_html(f"{save_path}.html")


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


runs = [
    # ["6", "lstm", "eagleone", "ee61962922c9473aa476a8787e1e2a74", "280"],
    # ["6", "lstm", "eagleone", "892b4ddc6ee94dfcb154ef6b2c895118", "260"],
    # ["6", "lstm", "eagleone", "e1f45f4708b64d42a96f07d433df4636", "240"],
    # ["6", "lstm", "eagleone", "3fc72d94e4af4b0094e88bd2eba98c71", "220"],
    # ["6", "lstm", "eagleone", "7c394f8d2fbf4a92a8397245e6cf3c21", "200"],
    # ["6", "lstm", "eagleone", "ab8582eb7845410c87ac06b5cfdd02f7", "180"],
    # ["6", "lstm", "eagleone", "0842084eabd44925bc5d38b76f39eda1", "160"],
    # ["6", "lstm", "eagleone", "8d9bd26969f6484fb4b16147f9e2c866", "140"],
    # ["6", "lstm", "eagleone", "48dbf092b34b4471a2deeeaab4d59eab", "120"],
    # ["6", "lstm", "eagleone", "d9706c7575f449648b772c5b895033d7", "100"],
    # ["6", "lstm", "eagleone", "37dacc40b7644e98bd66124ee6c591b2", "80"],
    # ["6", "lstm", "eagleone", "b928cb4e95f847a7bf0f94f075f67b15", "60"],
    # ["6", "lstm", "eagleone", "598cb6e18d8e43f29cb592c1e0283546", "40"],
    # ["6", "lstm", "eagleone", "5c4e4a12684048a689de099fee0c1f86", "20"],
    # ["1", "lstm", "husky", "de518b45d7424229b3a012e28533f29f", "160"],
    # ["1", "lstm", "husky", "ce9b825851754a9b9c911bb4f688aa9c", "140"],
    # ["1", "lstm", "husky", "8bbc493cc7de412eb506efab8172c488", "120"],
    # ["1", "lstm", "husky", "78ef7ea0f75f44c09ee8763d559a00de", "100"],
    # ["1", "lstm", "husky", "82cebdb59d8e4803a129cc9a2e988d92", "90"],
    # ["1", "lstm", "husky", "87625685da024d6c929331d121377f45", "70"],
    # ["1", "lstm", "husky", "cc90c5552fb9432e8f0509710f15462a", "50"],
    # ["1", "lstm", "husky", "bd417cf2cf6b4de4a053745805e90c7c", "30"],
    # ["1", "lstm", "husky", "82ad8f51e5a742b68f6b2cccfcc3ff62", "10"],
    # ["0", "lstm", "ieee_battery", "4160aeecb74a4d31a2469a00aae14073", "40"],
    # ["0", "lstm", "ieee_battery", "f65a7ad587b3499e81facac589afa99a", "35"],
    # ["0", "lstm", "ieee_battery", "75a9d601165045f2a49f8517f37f25af", "30"],
    # ["0", "lstm", "ieee_battery", "004d191ed1b5462e8857faede648e841", "25"],
    # ["0", "lstm", "ieee_battery", "be17c0baa3f746e3b18835d9da987dd8", "20"],
    # ["0", "lstm", "ieee_battery", "37ca35c214b44c6697f1cd6b9e7b92bb", "15"],
    # ["0", "lstm", "ieee_battery", "f68fe849293a4d969395d220774203fd", "10"],
    # ["0", "lstm", "ieee_battery", "16d3ac74c9b143758bae14720b050892", "5"],
    # ["5", "gru", "eagleone", "40d16ccf885d4d748464508d85b94272", "280"],
    # ["5", "gru", "eagleone", "236e79965db046799e96ca010fdb03b9", "260"],
    # ["5", "gru", "eagleone", "26ff1a1a73bd4b78bb0716803fddd3a4", "240"],
    # ["5", "gru", "eagleone", "efdc14dc38a846aba9a50bb577f6db41", "220"],
    # ["5", "gru", "eagleone", "81f5f70ee4a94277b7229d9636aa6cf9", "200"],
    # ["5", "gru", "eagleone", "9e736822f775464b93b9d63c4162187d", "180"],
    # ["5", "gru", "eagleone", "4c36637b881c42f9bd67b88fd96e0023", "160"],
    # ["5", "gru", "eagleone", "cefca513bfdb4ae3964760030a2ff3c9", "140"],
    # ["5", "gru", "eagleone", "b1d51518f82240c5b28eaa8422db7fe4", "120"],
    # ["5", "gru", "eagleone", "005a53d0a8054588b3547ab0dbbb42bf", "100"],
    # ["5", "gru", "eagleone", "90460063dc284c0a8c5f26f8d1642df1", "90"],
    # ["5", "gru", "eagleone", "aa13642f49254a8b8511e1bc5191d635", "80"],
    # ["5", "gru", "eagleone", "885e68d5dda646ca8308782f036f37a0", "70"],
    # ["5", "gru", "eagleone", "4664abab75c545a1b2339b8d9fe8756d", "60"],
    # ["5", "gru", "eagleone", "d470aca05fb744e1936d0a8b4ac527a7", "50"],
    # ["5", "gru", "eagleone", "4e5f77488dce42b8b6a8555ae7f00638", "40"],
    # ["5", "gru", "eagleone", "6046cc32e7c7416db47c90783eeac451", "30"],
    # ["5", "gru", "eagleone", "dfdb56ec384e43caa366445eeb753121", "20"],
    # ["5", "gru", "eagleone", "c5c1bb6821a24082a9b8421fea01c67b", "10"],
    # ["2", "gru", "husky", "f13ec3a255f1430fae65ea2567200fd7", "100"],
    # ["2", "gru", "husky", "0e99ce622d9d49b492903852e2b5499f", "90"],
    # ["2", "gru", "husky", "88eb7bbac9de4fa68720d0b939d5c771", "80"],
    # ["2", "gru", "husky", "456ddf8c724e4a25be1b6991574046d5", "70"],
    ["2", "gru", "husky", "7bd5e0612ceb4c1db130169ff3e1ecba", "60"],
    ["2", "gru", "husky", "081131091c2049e998a8da29021fb2a2", "50"],
    ["2", "gru", "husky", "198a624ed38d41448b59fdb1d0119bf1", "40"],
    ["2", "gru", "husky", "e00d637b7ba943268ca480f3733f22cd", "30"],
    ["2", "gru", "husky", "03a59c7b6c574746af88525ae6d7d0e6", "20"],
    ["2", "gru", "husky", "1b68b7c881b34c11a7c1ae9a97f6a000", "10"],
    ["2", "gru", "husky", "4e9bf33ba58c41aeb7a424e6cf5d927f", "160"],
    ["2", "gru", "husky", "65a6a237dc5b4b59943216d052d96ac2", "140"],
    ["2", "gru", "husky", "ea58237c02804a7abd215e4927f986de", "120"],
    ["3", "gru", "ieee_battery", "d98c625be6a140ba93af8a3c296b3c3e", "25"],
    ["3", "gru", "ieee_battery", "6ae7e640bc934c98899bf7b0d435b16f", "20"],
    ["3", "gru", "ieee_battery", "97567c351731488381334ad49ed37866", "15"],
    ["3", "gru", "ieee_battery", "42a5da378b02450aa23aaa1cdaec75e8", "10"],
    ["3", "gru", "ieee_battery", "27afe53d524c4f0895b0d722ecc14998", "5"],
]

from math import isnan


def main(
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "20220118 predictions",
        device: str = "cuda:0",
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    for exp_id, network, dataset_name, run_id, ls in runs:
        dataset = DatasetInputData.create(dataset_name)

        the_run = mlflow.get_run(run_id)
        run_name_templ = f"{network}_{dataset_name}_{ls}"
        input_steps = int(ls)
        model_file = os.path.join(the_run.data.params["dir"], "model.pt")
        device = torch.device(device)

        # print(run_name_templ)

        MSES = []

        for channel_name, columns, the_data, scaler in dataset.channels():
            if channel_name == the_run.data.tags["mlflow.runName"]:
                continue
            run_name = f"{run_name_templ}_{channel_name}"
            # print(run_name)
            with mlflow.start_run(run_name=run_name) as channel_run:
                temp_dir = tempfile.mkdtemp(dir="/home/pawel/mlflow_artifacts")
                mlflow.log_param("model_path", model_file)
                model = torch.load(
                    model_file,
                    map_location=torch.device(device),
                )

                the_data = scaler.transform(the_data)
                test_channel_orig = the_data
                test_channel = the_data[:, 1:]

                predictor = Predictor(model, input_window_size=input_steps)

                model_predictions = predictor.predict_signal_with_model(test_channel)[:, 0]
                diffs = model_predictions - test_channel_orig[:, 0]
                MSE = np.mean(diffs ** 2)
                mlflow.log_param("MSE", MSE)
                if not isnan(MSE):
                    MSES.append(MSE)

                np.save(os.path.join(temp_dir, "real.npy"), test_channel_orig)
                np.save(os.path.join(temp_dir, "predictions.npy"), model_predictions)
                np.save(os.path.join(temp_dir, "error.npy"), diffs)

                plot_results(
                    "test_predictions.png",
                    test_channel_orig,
                    model_predictions,
                    os.path.join(temp_dir, "test_vis.png"),
                )
                mlflow.log_param("path", temp_dir)
                mlflow.log_artifacts(temp_dir)
        print(f"{run_name_templ} AVG MSE={np.average(MSES)}")


if __name__ == "__main__":
    fire.Fire(main)
