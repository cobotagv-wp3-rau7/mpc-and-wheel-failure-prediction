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



lstm = np.load("c:/experiments/cobot_2023_weighted_multivariate_with_MPC/model=LSTM,layers=2,forecast=10,input_length=5,dataset=CoBot202210/_test_train/test_predictions.npy")
# scinet = np.load("c:/experiments/cobot_2023_weighted_multivariate_with_MPC/model=SCINet,forecast=10,input_length=64,dataset=CoBot202210/_test_train/test_predictions.npy")


plt.plot(mpc, 'b', label="actual mpc")
plt.plot(lstm, 'r', label="LSTM predictions")
# plt.plot(scinet, 'g', label="SCINet predictions")
plt.legend()
plt.xlabel("t [s]")
plt.ylabel("momentary power consumption [W]")
plt.show()
