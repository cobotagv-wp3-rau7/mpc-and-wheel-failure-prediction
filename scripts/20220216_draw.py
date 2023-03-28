import numpy as np
import pandas as pd
import plotly.express as px
import cobot_ml.data.datasets as ds
import os


items = [
    ("all features", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=30,dataset=husky"),
    # (">= 0.05", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=30,dataset=Husky_005"),
    ("corr >= 0.1", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=30,dataset=Husky_01"),
    # (">= 0.2", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=30,dataset=Husky_02"),
    ("corr >= 0.4", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=30,dataset=Husky_04"),

    # ("all features", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=170,dataset=formica20220104"),
    # ("corr >= 0.05", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=170,dataset=Formica_005"),
    # ("corr >= 0.1", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=170,dataset=Formica_01"),
    # ("corr >= 0.2", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=170,dataset=Formica_02"),
    # ("corr >= 0.4", "/mnt/cloud/20220201_experiments/model=LSTM,layers=2,forecast=10,input_length=170,dataset=Formica_04"),
]

_ds = ds.DatasetInputData.create(ds.Datasets.CoBot202210)
# chan = f"dump_from_simulation_{i}"
chan = f"test"
name, cols, data = _ds.channel(chan)

mpc = data[:, 0]
weight = data[:, 75] + data[:, 76] + data[:, 77] + data[:, 78]

plotdata = dict()
plotdata["actual signal"] = data

from matplotlib import pyplot as plt
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['text.usetex'] = True
# plt.rcParams['text.usetex'] = True






# plt.plot(plotdata["actual signal"], color="red", label="actual signal")
# plt.plot(plotdata["all features"], color="green", label="all features", linestyle='dashed', )
# plt.plot(plotdata["corr >= 0.1"], color="blue", label="corr >= 0.1", linestyle='dashed', )
# plt.plot(plotdata["corr >= 0.4"], color="fuchsia", label="corr >= 0.4", linestyle='dashed', )
# ax1 = plt.subplot()
# plt.legend()
# plt.plot(mpc, 'b')
# plt.xlabel("t [s]")
# plt.ylabel("momentary power consumption [W]", color='b')
#
# ax2 = ax1.twinx()
# plt.plot(weight, '--r')  # solid green
# plt.ylabel("payload weight [kg]", color='r')
#
# plt.show()



# weighted_multivariate_with_MPC
# weighted_multivariate_with_MPC
# weighted_multivariate_wo_MPC
# weighted_multivariate_with_MPC
# weighted_multivariate_wo_MPC

for shift, preds, descr in [
    (10, "c:/experiments/cobot_2023_weighted_multivariate_with_MPC/model=GRU,layers=2,forecast=10,input_length=10,dataset=CoBot202210/_test_train/test_predictions_complete.npy", "GRU"),
    (64, "c:/experiments/cobot_2023_weighted_multivariate_with_MPC/model=SCINet,forecast=10,input_length=64,dataset=CoBot202210/_test_train/test_predictions_complete.npy", "SCINet"),
    (10, "c:/experiments/cobot_2023_weighted_multivariate_wo_MPC/model=LSTM,layers=2,forecast=10,input_length=10,dataset=CoBot202210/_test_train/test_predictions_complete.npy", "LSTM"),
]:
    vals = np.load(preds)
    vals_1st = vals[:, 0]
    vals_10th = vals[:, 9]

    # vals = np.insert(np.load(), 0, np.zeros((10,)))
    # vals_1st = np.insert(vals_1st, 0, np.zeros((shift,)))
    vals_10th = np.insert(vals_10th, 0, np.zeros((10,)))

    left = 2200
    right = 2500
    plt.plot(mpc[left:right], 'b', label="actual MPC")
    plt.plot(vals_1st[left:right], 'r', label=f"{descr} predictions - nearest prediction")
    # plt.plot(scinet, 'g', label="SCINet predictions")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("momentary power consumption [W]")
    plt.show()

    plt.plot(mpc[left:right], 'b', label="actual MPC")
    plt.plot(vals_10th[left:right], 'r', label=f"{descr} predictions - 10 s forward")
    # plt.plot(scinet, 'g', label="SCINet predictions")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("momentary power consumption [W]")
    plt.show()
