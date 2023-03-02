import glob
import json
import os
import csv

with open(os.path.join("c:\\experiments\\metrics_2.csv"), 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    first_written = False

    for exps in [
        "cobot_2023_multivariate_with_MPC",
        "cobot_2023_multivariate_wo_MPC",
        "cobot_2023_univariate",
        "cobot_2023_weighted_multivariate_with_MPC",
        "cobot_2023_weighted_multivariate_wo_MPC"
    ]:
        base = os.path.join("c:\\experiments\\", exps)
        for dir in glob.glob(os.path.join(base, "*")):
            metrics_path = os.path.join(dir, "_test_train", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as mf:
                    metrics: dict = json.load(mf)
                    cols = ["experiment", "model"]
                    vals = [exps, os.path.basename(dir)]
                    for k, v in metrics.items():
                        for kk, vv in v.items():
                            cols.append(f"{k}_{kk}")
                            vals.append(metrics[k][kk])
                    if not first_written:
                        csvwriter.writerow(cols)
                        first_written = True
                    csvwriter.writerow(vals)
