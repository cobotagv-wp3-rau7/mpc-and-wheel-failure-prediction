from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC

from cobot_ml.data.datasets import CoBot20230708ForSVM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

selected_columns = [
    'FH.6000.[NNS] - Natural Navigation Signals.Difference heading average correction',
    'Difference heading average correction RM300',
    'FH.6000.[NNS] - Natural Navigation Signals.Distance average correction',
    'Distance average correction RM300',
]


def the_dataset(selected_columns):
    ds = CoBot20230708ForSVM(None, None)
    df = ds.the_data.all_the_data.dropna()
    df['labels'] = df['WHEEL_DIAMETER'].apply(lambda x: 0 if x == 52.9 else 1)
    df['labels_one_class'] = df['WHEEL_DIAMETER'].apply(lambda x: 1 if x == 52.9 else -1)

    df['Difference heading average correction RM300'] = \
        df['FH.6000.[NNS] - Natural Navigation Signals.Difference heading average correction'].rolling(
            window=300).mean()

    df['Distance average correction RM300'] = \
        df['FH.6000.[NNS] - Natural Navigation Signals.Distance average correction'].rolling(window=300).mean()

    scaler = StandardScaler()
    df = df.dropna()
    labels = df['labels']
    labels_one_class = df['labels_one_class']
    df = df.drop(['WHEEL_DIAMETER', 'labels', 'labels_one_class'], axis=1)
    df = scaler.fit_transform(df[selected_columns])

    # df = np.column_stack((df, (df[:,0]+df[:,1])/2))
    return scaler, df, labels.to_numpy(), labels_one_class.to_numpy()


scaler, X, y, y_one_class = the_dataset(selected_columns)

np.random.seed(37)
n_samples = 50000
indices = np.sort(np.random.choice(y.shape[0], n_samples, replace=False))

X = X[indices, :]
y = y[indices]
y_one_class = y_one_class[indices]

X_normal = X[y == 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.8, shuffle=True)


def show_dataset():
    import plotly.express as px
    df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 2], 'y': y})
    fig = px.scatter(df, x='X1', y='X2', color='y', color_continuous_scale='RdYlGn', opacity=0.7)
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene=dict(xaxis_title='cor1', yaxis_title='cor2'))
    fig.show()


def one_class(X, X_normal):
    # for kernel in ['sigmoid', 'rbf', 'linear']:
    #     for gamma in [0.01, 0.1, 0.2, 0.5]:
    #         for nu in [0.2, 0.25, 0.3, 0.35, 0.4]:

    kernel = 'sigmoid'
    gamma = 0.1
    nu = 0.3

    clf = OneClassSVM(gamma=gamma, nu=nu, kernel=kernel)
    clf.fit(X_normal)
    y_pred = clf.predict(X)
    print(f"kernel: {kernel}, gamma: {gamma}, nu: {nu}, accuracy: {accuracy_score(y_one_class, y_pred)}")

    # Ocena modelu
    print(classification_report(y_one_class, y_pred))
    print("Accuracy Score:", accuracy_score(y_one_class, y_pred))

    zgodne = y_one_class == y_pred
    niezgodne = ~zgodne

    plt.figure(figsize=(10, 6))

    # plt.scatter(range(y_one_class.shape[0]), y_one_class, color='gray', label='Labels', s=1)
    plt.scatter(range(y_one_class.shape[0]), X[:, 1], color='gray', label='corr1 RollAVG(300)', s=0.5)
    plt.scatter(range(y_one_class.shape[0]), X[:, 3], color='gray', label='corr2 RollAVG(300)', s=0.5)
    # plt.scatter(range(y_one_class.shape[0]), X[:, 4], color='black', label='avg(corr1,corr2)', s=0.5)
    plt.scatter(np.where(niezgodne), y_pred[niezgodne], color='red', label='Niezgodne', s=1)
    plt.scatter(np.where(zgodne), y_pred[zgodne], color='green', label='Zgodne', s=1)

    # Dodaj etykiety i legendę
    plt.title('Porównanie labels i y_pred')
    plt.xlabel('Indeks')
    plt.ylabel('Wartości')
    plt.legend()

    # Pokaż wykres
    plt.show()


def two_class(X_train, y_train, X_test, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    for clf in [
        # KNeighborsClassifier(),
        GaussianNB(),
        # DecisionTreeClassifier(),
        # RandomForestClassifier(),
        SVC()
    ]:
        # for kernel in ['sigmoid', 'rbf', 'linear']:
        #     for gamma in [0.01, 0.1, 0.2, 0.5]:
        #         for nu in [0.2, 0.25, 0.3, 0.35, 0.4]:

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("***********************")
        print(f"classifier: {clf}, accuracy: {accuracy_score(y_test, y_pred)}")

        # Ocena modelu
        print(classification_report(y_test, y_pred))

        # zgodne = y_test == y_pred
        # niezgodne = ~zgodne
        #
        # plt.figure(figsize=(10, 6))
        #
        # plt.scatter(range(y_test.shape[0]), y_test, color='gray', label='Labels', s=0.5)
        # plt.scatter(range(y_test.shape[0]), X_test[:, 1], color='gray', label='corr1 RollAVG(300)', s=0.5)
        # plt.scatter(range(y_test.shape[0]), X_test[:, 3], color='gray', label='corr2 RollAVG(300)', s=0.5)
        # # plt.scatter(range(y_one_class.shape[0]), X[:, 4], color='black', label='avg(corr1,corr2)', s=0.5)
        # plt.scatter(np.where(niezgodne), y_pred[niezgodne], color='red', label='Niezgodne', s=1)
        # plt.scatter(np.where(zgodne), y_pred[zgodne], color='green', label='Zgodne', s=1)
        #
        # # Dodaj etykiety i legendę
        # plt.title('Porównanie labels i y_pred')
        # plt.xlabel('Indeks')
        # plt.ylabel('Wartości')
        # plt.legend()
        #
        # # Pokaż wykres
        # plt.show()

if __name__ == "__main__":
    two_class(X_train, y_train, X_test, y_test)
