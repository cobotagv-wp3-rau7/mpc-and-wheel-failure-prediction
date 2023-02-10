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

_ds = ds.DatasetInputData.create("husky")
# chan = f"dump_from_simulation_{i}"
chan = f"111"
name, cols, data = _ds.channel(chan)

data = data[30:, 0]


plotdata = dict()
plotdata["actual signal"] = data

for setname, item in items:
    # if setname == "all features":
    #     preds = np.load(os.path.join(item, "_test_dump_from_simulation_5", f"{chan}_predictions.npy"))
    # else:
    preds = np.load(os.path.join(item, "_test", f"{chan}_predictions.npy"))
    plotdata[setname] = preds

import matplotlib

from matplotlib import pyplot as plt
# matplotlib.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['mathtext.fontset'] = 'cm'
# matplotlib.rcParams['text.usetex'] = True
# plt.rcParams['text.usetex'] = True






plt.plot(plotdata["actual signal"], color="red", label="actual signal")
plt.plot(plotdata["all features"], color="green", label="all features", linestyle='dashed', )
plt.plot(plotdata["corr >= 0.1"], color="blue", label="corr >= 0.1", linestyle='dashed', )
plt.plot(plotdata["corr >= 0.4"], color="fuchsia", label="corr >= 0.4", linestyle='dashed', )
plt.legend()
# plt.plot(x, x + 0, '')  # solid green
# plt.plot(x, x + 1, '--c') # dashed cyan
# plt.plot(x, x + 2, '-.k') # dashdot black
# plt.plot(x, x + 3, ':r');  # dotted red

plt.xlabel("sequence index")
plt.ylabel("power consumption (standarized)")
plt.show()

# fig = px.line(pd.DataFrame(plotdata),
#               labels={
#                   "index": "sequence index",
#                   "value": "power consumption"
#               },
#               color_discrete_sequence=["red", "green", "blue", "fuchsia"]
#               )
# fig.show()
# fig.write_html(f"/home/pawel/Downloads/formica_forecasts.html")

