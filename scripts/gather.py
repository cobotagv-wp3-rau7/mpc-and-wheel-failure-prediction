import pandas
import json
import os
import glob
import numpy as np
from cobot_ml.data.datasets import Datasets
import torch

dd = dict()

i = 0
for f in glob.glob("/mnt/cloud/20220201_experiments/*"):

    ff = dict()
    for item in os.path.basename(f).split(","):
        k, v = item.split("=")
        ff[k] = v
    if ff["dataset"] == Datasets.EagleOne:
        continue

    train_losses = []
    val_losses = []
    avg_times = []
    epoch_counts = []
    best_epochs = []
    channel_count = 0

    for channel in glob.glob(os.path.join(f, "*")):
        if not os.path.exists(os.path.join(channel, "summary.json")):
            continue
        with open(os.path.join(channel, "summary.json")) as sf:
            summary = json.load(sf)
        with open(os.path.join(channel, "losses.json")) as lf:
            losses = json.load(lf)

        train_losses.append(summary["train_loss"])
        val_losses.append(summary["valid_loss"])
        avg_times.append(summary["avg_epoch_time"])
        epoch_counts.append(losses["epoch_count"])
        best_epochs.append(losses["best_epoch"])
        channel_count += 1

        print(channel)

    ff["avg_train_loss"] = np.mean(train_losses)
    ff["avg_valid_loss"] = np.mean(val_losses)
    ff["avg_epoch_time"] = np.mean(avg_times)

    ff["avg_epoch_count"] = np.mean(epoch_counts)
    ff["avg_best_epoch"] = np.mean(best_epochs)
    ff["channel_count"] = channel_count

    pred_timing = []
    for tchannel in glob.glob(os.path.join(f, "_test_*")):
        if not os.path.exists(os.path.join(tchannel, "metrics.json")):
            continue
        working_on_channel = os.path.basename(tchannel)[6:]

        m_mse = []
        m_mae = []
        m_mape = []
        m_smape = []
        with open(os.path.join(tchannel, "metrics.json")) as mf:
            chmetrics = json.load(mf)
            for k, v in chmetrics.items():
                pred_timing.append(float(v["timing"]))
                if k != working_on_channel:
                    m_mse.append(float(v["mse"]))
                    m_mae.append(float(v["mae"]))
                    m_mape.append(float(v["mape"]))
                    m_smape.append(float(v["smape"]))
        ff[f"test_mse_cross_{working_on_channel}"] = np.mean(m_mse)
        ff[f"test_mae_cross_{working_on_channel}"] = np.mean(m_mae)
        ff[f"test_mape_cross_{working_on_channel}"] = np.mean(m_mape)
        ff[f"test_smape_cross_{working_on_channel}"] = np.mean(m_smape)

    all_metrics_file = os.path.join(f, "_test_dump_from_simulation_5", "metrics.json") \
        if ff["dataset"] == Datasets.Formica20220104 \
        else os.path.join(f, "_test", "metrics.json")

    all_metr_timing = []
    if os.path.exists(all_metrics_file):
        m_mse = []
        m_mae = []
        m_mape = []
        m_smape = []
        with open(all_metrics_file) as amf:
            chmetrics = json.load(amf)
        considered_channel = "dump_from_simulation_5" if ff["dataset"].lower().startswith("formica") \
            else list(chmetrics.keys())[0]

        model_path = os.path.join(f, considered_channel, "model.pt")
        device = torch.device("cuda:0")
        model = torch.load(model_path, map_location=device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())

        with open(all_metrics_file) as mf:
            chmetrics = json.load(mf)
            for ch, v in chmetrics.items():
                if "timing" in v:
                    all_metr_timing.append(float(v["timing"]))
                if ch != considered_channel:
                    m_mse.append(float(v["mse"]))
                    m_mae.append(float(v["mae"]))
                    m_mape.append(float(v["mape"]))
                    m_smape.append(float(v["smape"]))

        if len(pred_timing) != 0:
            ff["avg_pred_time"] = np.mean(pred_timing)
        else:
            ff["avg_pred_time"] = np.mean(all_metr_timing)

        ff[f"test_mse_all"] = np.nanmean(m_mse)
        ff[f"test_mae_all"] = np.nanmean(m_mae)
        ff[f"test_mape_all"] = np.nanmean(m_mape)
        ff[f"test_smape_all"] = np.nanmean(m_smape)
        ff["model_param_count"] = pytorch_total_params

    dd[i] = ff
    i += 1

df = pandas.DataFrame(dd).transpose()
df.sort_values(["dataset", "model", "layers", "input_length"], inplace=True)
df.to_csv("/mnt/cloud/summary_2nd_stage.csv")
