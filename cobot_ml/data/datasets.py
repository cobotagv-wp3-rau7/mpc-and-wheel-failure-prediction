import glob
import os
import typing
import warnings

import natsort
import numpy as np
import pandas as pd
import torch
from torch.utils import data

import cobot_ml.data.patchers as patchers


class TensorPairsDataset(data.Dataset):
    """
    Dataset holding pairs of input and target tensors.
    :param inputs: List of input tensors
    :param targets: List of target tensors
    """

    def __init__(
            self, inputs: typing.List[torch.Tensor], targets: typing.List[torch.Tensor]
    ):
        assert len(inputs) == len(
            targets
        ), "Sequences and targets should have the same length"
        if len(inputs) == 0:
            warnings.warn("Empty input provided, add data to dataset!")

        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        X = self.inputs[idx]
        y = self.targets[idx]
        return X, y

    def __add__(self, other):
        if not isinstance(other, TensorPairsDataset):
            raise TypeError(f"Unsupported operand type: {type(other)}")

        new_inputs = self.inputs + other.inputs
        new_targets = self.targets + other.targets

        return TensorPairsDataset(new_inputs, new_targets)

    def get_unraveled_targets(self) -> torch.Tensor:
        """
        Return first value from each sample (but all values from the last
        sample). Used to build 1D vector of real values.
        """
        real_values = [self.targets[idx][0] for idx in range(len(self.targets) - 1)]
        return torch.hstack([torch.tensor(real_values), self.targets[-1]])


class DatasetInputData:
    implementations = {}

    @staticmethod
    def create(dataset: str, *args, **kwargs):
        assert dataset in DatasetInputData.implementations, f"Unknown dataset {dataset}"
        impl = DatasetInputData.implementations[dataset]
        ds_path = impl[1] if len(args) == 0 else args[0]
        return impl[0](ds_path, kwargs)

    def channel_names(self):
        raise NotImplementedError()

    def channel(self, channel_name):
        raise NotImplementedError()


def _files_as_pattern(path: str, pattern: str):
    files = glob.glob(os.path.join(path, pattern))
    return natsort.natsorted(files, key=lambda f: os.path.basename(f))


def _vector_length(df: pd.DataFrame, vector_name: str) -> pd.Series:
    return np.sqrt(
        df[f"{vector_name}_x"] ** 2 +
        df[f"{vector_name}_y"] ** 2 +
        df[f"{vector_name}_z"] ** 2
    )


class Datasets:
    CoBot202210 = "CoBot202210"
    CoBot20230708 = "CoBot20230708"


################################

base_path_datasets_ = "c:\\datasets\\"


class CoBot20230708Data:
    LABEL_COLUMN = "WHEEL_DIAMETER"

    def __init__(self):
        # base_dir = 'c:\\projekty\\cobot_july_august\\'
        # sciezka_csv = base_dir + 'concatenated_with_wheels_change_changing_columns.csv'
        base_dir = 'c:\\projekty\\cobot_july_august2\\'
        sciezka_csv = base_dir + 'concatenated_with_wheel_diameter_changing_columns.csv'

        ignored_columns = [
            'timestamp', 'isoTimestamp', 'FH.6000.[TS] TIME STAMP.Time stamp',
            "FH.6000.[ENS] - Energy Signals.State Of Charge",
            "FH.6000.[ENS] - Energy Signals.Battery cell voltage",
            "FH.6000.[ODS] - Odometry Signals.Cumulative distance right",
            "FH.6000.[ENS] - Energy Signals.Momentary current consuption mA",
            "FH.6000.[ODS] - Odometry Signals.Cumulative distance left",
            "FH.6000.[ENS] - Energy Signals.Cumulative energy consumption Wh",
        ]

        all_the_data = pd.read_csv(sciezka_csv).iloc[2:]
        all_the_data = all_the_data.drop(columns=ignored_columns)

        # move the label column to 0-th index
        self.columns = all_the_data.columns.tolist()
        self.columns.remove(CoBot20230708Data.LABEL_COLUMN)
        self.columns.insert(0, CoBot20230708Data.LABEL_COLUMN)
        self.all_the_data = all_the_data[self.columns]

        self.column_stats = {}
        for column in self.all_the_data.columns:
            column_data = self.all_the_data[column]
            if column_data.dtype != bool:
                self.column_stats[column] = {
                    'mean': column_data.mean(),
                    'std': column_data.std(),
                    'min': column_data.min(),
                    'max': column_data.max(),
                }


class CoBot20230708(DatasetInputData):

    def ranges(self, indices, total_length, range_length):
        beginning_from_indices = []
        all_the_rest = []
        previous_index = -range_length
        for index in indices:
            all_the_rest.append((previous_index + range_length, index))
            beginning_from_indices.append((index, index+range_length))
            previous_index = index
        all_the_rest.append((previous_index + range_length, total_length))
        return beginning_from_indices, all_the_rest

    def __init__(self, path, kwargs):
        self.the_data = CoBot20230708Data()
        all_the_data = self.the_data.all_the_data.copy(deep=True)
        for col, stats in self.the_data.column_stats.items():
            _range = stats['max'] - stats['min']
            if _range != 0:
                all_the_data[col] = (all_the_data[col] - stats['min']) / _range

        slice_length = 2000
        begin_indices = [3000, 12000, 21000, 30000, 39000, 48000, 57000, 66000]
        beginning_ranges, all_the_rest = self.ranges(begin_indices, len(all_the_data), slice_length)

        train_dfs = [all_the_data.iloc[rb:re] for rb, re in beginning_ranges]
        test_dfs = [all_the_data.iloc[rb:re] for rb, re in all_the_rest]

        self.train_data = pd.concat(train_dfs, ignore_index=True)
        self.test_data = pd.concat(test_dfs, ignore_index=True)

        print(f"Loaded Cobot20230708: len(train)={len(self.train_data)}, len(test)={len(self.test_data)}")

    def channel_names(self):
        return ["train", "test"]

    def channel(self, channel_name):
        if channel_name == "train":
            return channel_name, self.the_data.columns, self.train_data.to_numpy()
        elif channel_name == "test":
            return channel_name, self.the_data.columns, self.test_data.to_numpy()
        else:
            raise f"Invalid channel name {channel_name}. Only 'train' and 'test' allowed"


    def prepare_dataset(
            self,
            channel_name: str,
            input_steps: int,
            output_steps: int,
    ) -> TensorPairsDataset:
        if channel_name == "train":
            the_data = self.train_data.to_numpy()
        elif channel_name == "test":
            the_data = self.test_data.to_numpy()
        else:
            raise f"Invalid channel name {channel_name}. Only 'train' and 'test' allowed"

        patches = patchers.patch_with_stride(the_data, (input_steps + output_steps), stride=1)

        patches = [torch.from_numpy(patch.astype(np.float32)) for patch in patches]
        X = [patch[:input_steps, 1:] for patch in patches]
        y = [patch[input_steps:, 0] for patch in patches]
        return TensorPairsDataset(X, y)




DatasetInputData.implementations[Datasets.CoBot20230708] = (CoBot20230708, "nope")



class CoBot20230708ForSVM(DatasetInputData):

    def __init__(self, path, kwargs):
        self.the_data = CoBot20230708Data()
        all_the_data = self.the_data.all_the_data.copy(deep=True)
        for col, stats in self.the_data.column_stats.items():
            _range = stats['max'] - stats['min']
            if _range != 0:
                all_the_data[col] = (all_the_data[col] - stats['min']) / _range

        bool_columns = all_the_data.select_dtypes(include='bool').columns
        all_the_data[bool_columns] = all_the_data[bool_columns].replace({True: 1, False: 0})
        all_the_data[bool_columns] = all_the_data[bool_columns].astype(int)

        train_test_split_point = 8000
        self.train_data = all_the_data.iloc[:train_test_split_point]
        self.test_data = all_the_data.iloc[train_test_split_point:]

        print(f"Loaded Cobot20230708: len(train)={len(self.train_data)}, len(test)={len(self.test_data)}")

    def channel_names(self):
        return ["train", "test"]

    def channel(self, channel_name):
        if channel_name == "train":
            return channel_name, self.the_data.columns, self.train_data.to_numpy()
        elif channel_name == "test":
            return channel_name, self.the_data.columns, self.test_data.to_numpy()
        else:
            raise f"Invalid channel name {channel_name}. Only 'train' and 'test' allowed"



######
def prepare_dataset(
        channel_values: np.ndarray,
        input_steps: int,
        output_steps: int,
) -> TensorPairsDataset:
    def tensor_from(input: np.ndarray):
        return torch.from_numpy(input.astype(np.float32))

    patches = patchers.patch(channel_values, input_steps + output_steps, step=1)
    X, y = [(tensor_from(patch[:input_steps, 1:]), tensor_from(patch[input_steps:, 0])) for patch in patches]
    return TensorPairsDataset(X, y)


if __name__ == "__main__":
    dd = CoBot20230708(None, None)
    print('f')
