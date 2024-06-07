import os
import pandas as pd


def read_metrics(directory):
    results = dict()
    # Przechodzimy przez wszystkie podkatalogi w podanym katalogu
    for subdir, dd, files in os.walk(directory):
        for file in files:
            if file == "metrics.csv":
                file_path = os.path.join(subdir, file)
                # Wczytujemy plik CSV
                df = pd.read_csv(file_path)
                # Znajdujemy MSE dla 'normal test'
                normal_test_mse = df[df['name'] == 'normal test']['mse'].values[0]
                # Obliczamy średnie MSE dla wierszy od 200-240 wzwyż
                range_mse = df.iloc[6:]['mse'].mean()
                print(range_mse)
                # Zapisujemy wyniki w słowniku
                xx = os.path.basename(subdir)
                results[xx] = (normal_test_mse, range_mse)
    return results


# Function to identify the Pareto front
def identify_pareto(data):
    pareto_front = []
    for sol, (fit1, fit2) in data.items():
        is_pareto = True
        for other_sol, (other_fit1, other_fit2) in data.items():
            if (fit1 > other_fit1 and fit2 <= other_fit2) or (fit1 >= other_fit1 and fit2 < other_fit2):
                is_pareto = False
                break
        if is_pareto:
            pareto_front.append((fit1, fit2))
    return pareto_front


# Ścieżka do katalogu głównego
import numpy as np
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

results = {}
for directory_path in ['experiments_gecco2024_normal_test_nearest_to_008_longer',
                       'experiments_gecco2024_normal_test_nearest_to_004_longer'
                       'experiments_gecco2024_normal_test_nearest_to_004',
                       'experiments_gecco2024_normal_test_nearest_to_002',
                       'experiments_gecco2024_2',
                       'experiments_gecco2024_1'
                       ]:
    results.update(read_metrics(os.path.join("e:", directory_path)))

results = {sol: (fit1, fit2) for sol, (fit1, fit2) in results.items() if fit1 < 0.1}

pareto_front = identify_pareto(results)

all = np.array([(fit1, fit2) for sol, (fit1, fit2) in results.items()])

baselines = np.array([(0.0922, 0.2743), (0.0843, 0.2470)])


# Plot the Pareto front
pareto_front = np.array(pareto_front)
plt.scatter(all[:, 0], all[:, 1], color='gray', label="Solutions")
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label="Pareto front")
plt.scatter(baselines[:, 0], baselines[:, 1], color='blue', label='Baselines')

plt.xlabel('MSE(TN)')
plt.ylabel('MSE(TA)')
# plt.title(directory_path)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join("e:\\experiments_charts", f"all.png"))
plt.show()
