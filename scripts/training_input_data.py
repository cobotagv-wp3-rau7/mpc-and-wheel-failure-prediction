import cobot_ml.data.datasets as dss
from cobot_ml.data.datasets import Datasets
import cobot_ml.models as models


def first_stage():
    for input_length in range(10, 200, 20):
        for output_length in [10]:
            for dataset, subset in [
                # (Datasets.EagleOne, None),
                (Datasets.Formica20220104, None),
                # (Datasets.Husky, dss.Husky.SUBSET),
                # (Datasets.IEEEBattery, dss.IEEEBattery.SUBSET)
            ]:
                for model, params in [
                    (models.LSTM, {"n_layers": 1}),
                    (models.BiLSTM, {"n_layers": 1}),
                    (models.GRU, {"n_layers": 1}),
                    (models.BiGRU, {"n_layers": 1}),

                    (models.LSTM, {"n_layers": 2}),
                    (models.BiLSTM, {"n_layers": 2}),
                    (models.GRU, {"n_layers": 2}),
                    (models.BiGRU, {"n_layers": 2}),
                ]:
                    yield input_length, output_length, dataset, subset, model, params


###################################################################3
def formica_2nd_stage():
    for input_length in [50, 90, 150, 170]:
        for dataset, subset in [
            (Datasets.Formica_005, None),
            (Datasets.Formica_01, None),
            (Datasets.Formica_02, None),
            (Datasets.Formica_04, None),
        ]:
            for model, params in [
                (models.LSTM, {"n_layers": 2}),
            ]:
                yield input_length, 10, dataset, subset, model, params


###################################################################3
def husky_2nd_stage():
    for input_length in [10, 30, 50, 90]:
        for dataset, subset in [
            (Datasets.Husky_005, dss.Husky.SUBSET),
            (Datasets.Husky_01, dss.Husky.SUBSET),
            (Datasets.Husky_02, dss.Husky.SUBSET),
            (Datasets.Husky_04, dss.Husky.SUBSET),
        ]:
            for model, params in [
                (models.LSTM, {"n_layers": 2}),
            ]:
                yield input_length, 10, dataset, subset, model, params


###################################################
def ieee_battery_2nd_stage():
    for dataset, subset in [
        (Datasets.IEEEBattery_005, dss.IEEEBattery.SUBSET),
        (Datasets.IEEEBattery_03, dss.IEEEBattery.SUBSET),
        (Datasets.IEEEBattery_04, dss.IEEEBattery.SUBSET),
    ]:
        for model, params, input_length in [
            (models.BiLSTM, {"n_layers": 1}, 70),
            (models.BiLSTM, {"n_layers": 2}, 10),
            (models.LSTM, {"n_layers": 1}, 70),
            (models.LSTM, {"n_layers": 2}, 10),
        ]:
            yield input_length, 10, dataset, subset, model, params
