import glob
import json
import os
import typing
import warnings

import natsort
import numpy as np
import pandas as pd
import torch
from scipy import signal
from sklearn import preprocessing as preprocess
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


class EagleOne(DatasetInputData):

    def __init__(self, path: str, kwargs):
        self.path = path

    def channel(self, channel_name):
        channel_file = os.path.join(self.path, f"{channel_name}.csv")
        input_data = pd.read_csv(channel_file)

        ENERGY_COL = "AGV_EagleOne.6003.[MPC] Momentary power consumption.Momentary power consumption 3"

        RPM1_COL = "AGV_EagleOne.6000.[MS] MOTORS’ SIGNALS.Motor 1 - RPM"
        RPM2_COL = "AGV_EagleOne.6000.[MS] MOTORS’ SIGNALS.Motor 2 - RPM"

        input_cols = [
            ENERGY_COL,
            RPM1_COL,
            RPM2_COL,
            "AGV_EagleOne.6000.[PAS] Pin Actuator Signals.Actuator 1 - downward[SByte]",
            "AGV_EagleOne.6000.[PAS] Pin Actuator Signals.Actuator 1 - upward[SByte]",
        ]

        # cols = input_data.columns.tolist()
        # input_data = input_data[[cols[112]] + cols[2:107] + cols[113:]]
        input_data = input_data[input_cols]

        input_data[ENERGY_COL] = signal.detrend(input_data[ENERGY_COL].to_numpy())
        input_data[RPM1_COL] = abs(input_data[RPM1_COL])
        input_data[RPM2_COL] = abs(input_data[RPM2_COL])

        whole_data = input_data.to_numpy()
        scaler = preprocess.StandardScaler()
        whole_data = scaler.fit_transform(whole_data)

        return channel_name, input_data.columns.tolist(), whole_data

    def channel_names(self):
        return [os.path.splitext(os.path.basename(channel_file))[0] for channel_file in
                _files_as_pattern(self.path, "dump_simulation_?.csv")]


class Formica20220104(DatasetInputData):
    COLUMNS = [
        'FH.6000.[ENC] - Energy Signals.Momentary power consumption',

        'FH.6000.[ENC] - Energy Signals.Battery cell voltage',
        'FH.6000.[AI] - ALARM INFORMATION.Alarm Information - Safety - Circuit Opened',
        'FH.6000.[AI] - ALARM INFORMATION.Alarm Information - Safety - Front Scanner Protective Zone Active',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - executed',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - in progress (#)',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - executed',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - in progress (#)',
        'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - command on',
        'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission',
        'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - safety interlock',
        'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - command on',
        'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission',
        'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - safety interlock',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Down - executed',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Down - in progress (#)',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - automatic permission (#)',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - executed',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - in progress (#)',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - safety interlock',
        'FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active', 'FH.6000.[GS] GENERAL SIGNALS.Manual Mode active',
        'FH.6000.[GS] GENERAL SIGNALS.PLC fault active',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) - Red (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Blue (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Green (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Blue (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Green (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Red (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LEG RGB - External control active (Forbot has no LED so this signal is inactive)',
        'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Automatic Mode Active',
        'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Manual Mode Active',
        'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - Safety - Scanners Safety Zones Muted',
        'FH.6000.[ODS] - Odometry Signals.Cumulative distance left',
        'FH.6000.[ODS] - Odometry Signals.Cumulative distance right',
        'FH.6000.[ODS] - Odometry Signals.Momentary frequency of left encoder pulses',
        'FH.6000.[ODS] - Odometry Signals.Momentary frequency of right encoder pulses',
        'FH.6000.[SS] SAFETY SIGNALS.AGV velocity active zone',
        'FH.6000.[SS] SAFETY SIGNALS.Safety circuit closed',
        'FH.6000.[SS] SAFETY SIGNALS.Scanners active zones',
        'FH.6000.[WI] - WARNING INFORMATION.Warning Information - General Control - Fans Inactive',
        'FH.6000.[WI] - WARNING INFORMATION.Warning Information - Safety - Front Scanner Warning Zone Active',
        'FH.6000.[WI] - WARNING INFORMATION.Warning Information - Safety - Rear Scanner Warning Zone Active',
    ]

    def __init__(self, path: str, kwargs):
        self.path = path

    def columns(self):
        return Formica20220104.COLUMNS

    def channel_names(self):
        return [os.path.splitext(os.path.basename(channel_file))[0] for channel_file in
                _files_as_pattern(self.path, "dump_from_simulation_?.csv")]

    def channel(self, channel_name):
        channel_file = os.path.join(self.path, f"{channel_name}.csv")

        input_data = pd.read_csv(channel_file)

        input_data = input_data[self.columns()].astype(dtype=np.float32)

        whole_data = input_data.to_numpy()
        scaler = preprocess.StandardScaler()
        whole_data = scaler.fit_transform(whole_data)

        return channel_name, input_data.columns.tolist(), whole_data


class IEEEBattery(DatasetInputData):
    SCALER_RANGE = (-1, 1)

    SUBSET = ['TripA01', 'TripA08', 'TripA15', 'TripA23', 'TripA30', 'TripB05', 'TripB12', 'TripB19', 'TripB26',
              'TripB33']

    COLUMNS = [
        'MPC', 'Velocity [km/h]', 'Elevation [m]', 'Throttle [%]', 'Motor Torque [Nm]',
        'Longitudinal Acceleration [m/s^2]', 'Regenerative Braking Signal ', 'Battery Temperature [°C]',
        'max. Battery Temperature [°C]', 'SoC [%]', 'displayed SoC [%]', 'min. SoC [%]', 'max. SoC [%)',
        'Heating Power CAN [kW]', 'Requested Heating Power [W]', 'AirCon Power [kW]', 'Heater Signal',
        'Ambient Temperature [°C]', 'Requested Coolant Temperature [°C]',
        'Heat Exchanger Temperature [°C]', 'Cabin Temperature Sensor [°C]'
    ]

    def __init__(self, path: str, kwargs):
        self.path = path

    def channel_names(self):
        return [os.path.splitext(os.path.basename(channel_file))[0] for channel_file in
                _files_as_pattern(self.path, "Trip???.csv")]

    def columns(self):
        return IEEEBattery.COLUMNS

    def channel(self, channel_name):
        channel_file = os.path.join(self.path, f"{channel_name}.csv")

        VOLTAGE_COL = 'Battery Voltage [V]'
        CURRENT_COL = 'Battery Current [A]'

        input_data = pd.read_csv(channel_file, sep=';', encoding='latin1')

        input_cols = ["MPC"] + input_data.columns.tolist()
        input_cols.remove('Time [s]')
        input_cols.remove(VOLTAGE_COL)
        input_cols.remove(CURRENT_COL)

        input_data["MPC"] = abs(input_data[VOLTAGE_COL]) * abs(input_data[CURRENT_COL])

        input_cols = [c for c in self.columns() if c in input_cols]
        if len(input_cols) != len(self.columns()):
            return None, None, None

        input_data = input_data[input_cols]

        whole_data = input_data.to_numpy()
        scaler = preprocess.StandardScaler()
        whole_data = scaler.fit_transform(whole_data)

        return channel_name, input_cols, whole_data


def _vector_length(df: pd.DataFrame, vector_name: str) -> pd.Series:
    return np.sqrt(
        df[f"{vector_name}_x"] ** 2 +
        df[f"{vector_name}_y"] ** 2 +
        df[f"{vector_name}_z"] ** 2
    )


class Husky(DatasetInputData):
    VELOCITY_COL = "velocity"
    ANGULAR_COL = "angular"
    LIN_ACC_COL = "linear_acceleration"
    COLUMNS = [
        "power",
        "position_x", "position_y", "position_z",
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "velocity_x", "velocity_y", "velocity_z", VELOCITY_COL,
        "angular_x", "angular_y", "angular_z", ANGULAR_COL,
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z", LIN_ACC_COL,
        "temperature", "humidity",
    ]

    SUBSET = [
        "1", "10", "30",  # A
        "40", "50", "70",  # B
        "80", "90", "100",  # C
        "109", "111", "113",  # D
    ]

    def __init__(self, path: str, kwargs):
        self.path = path

    def channel_names(self):
        return [os.path.splitext(os.path.basename(channel_file))[0] for channel_file in
                _files_as_pattern(self.path, "*.csv")]

    def columns(self):
        return Husky.COLUMNS

    def channel(self, channel_name):
        channel_file = os.path.join(self.path, f"{channel_name}.csv")

        input_data = pd.read_csv(channel_file, sep=',', encoding='latin1')
        input_data["power"] = abs(input_data["battery_voltage_husky"]) * abs(input_data["battery_current_husky"])
        input_data[self.VELOCITY_COL] = _vector_length(input_data, self.VELOCITY_COL)
        input_data[self.ANGULAR_COL] = _vector_length(input_data, self.ANGULAR_COL)
        input_data[self.LIN_ACC_COL] = _vector_length(input_data, self.LIN_ACC_COL)
        input_cols = input_data.columns.tolist()

        input_cols = [c for c in self.columns() if c in input_cols]
        if len(input_cols) != len(self.columns()):
            return None, None

        input_data = input_data[input_cols]

        whole_data = input_data.to_numpy()
        scaler = preprocess.StandardScaler()
        whole_data = scaler.fit_transform(whole_data)

        return channel_name, input_cols, whole_data


class Datasets:
    EagleOne = "eagleone"
    Formica20220104 = "formica20220104"
    Formica_005 = "Formica_005"
    Formica_01 = "Formica_01"
    Formica_02 = "Formica_02"
    Formica_04 = "Formica_04"

    IEEEBattery = "ieee_battery"
    IEEEBattery_005 = "IEEEBattery_005"
    IEEEBattery_03 = "IEEEBattery_03"
    IEEEBattery_04 = "IEEEBattery_04"

    Husky = "husky"
    Husky_005 = "Husky_005"
    Husky_01 = "Husky_01"
    Husky_02 = "Husky_02"
    Husky_04 = "Husky_04"

    CoBot202210 = "CoBot202210"
    CoBot20230708 = "CoBot20230708"


class Formica_005(Formica20220104):
    def columns(self):
        return [
            'FH.6000.[ENC] - Energy Signals.Momentary power consumption',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - in progress (#)',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - in progress (#)',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Down - executed',
            'FH.6000.[ODS] - Odometry Signals.Momentary frequency of left encoder pulses',
            'FH.6000.[ODS] - Odometry Signals.Momentary frequency of right encoder pulses',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - executed',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - command on',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - command on',
            'FH.6000.[SS] SAFETY SIGNALS.AGV velocity active zone',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - executed',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - executed',
            'FH.6000.[WI] - WARNING INFORMATION.Warning Information - General Control - Fans Inactive',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance right',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance left',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[SS] SAFETY SIGNALS.Scanners active zones',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - automatic permission (#)',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - safety interlock',
            'FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active',
            'FH.6000.[GS] GENERAL SIGNALS.Manual Mode active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Automatic Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Manual Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - Safety - Scanners Safety Zones Muted',
        ]


class Formica_01(Formica20220104):
    def columns(self):
        return [
            'FH.6000.[ENC] - Energy Signals.Momentary power consumption',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - in progress (#)',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Down - executed',
            'FH.6000.[ODS] - Odometry Signals.Momentary frequency of left encoder pulses',
            'FH.6000.[ODS] - Odometry Signals.Momentary frequency of right encoder pulses',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - executed',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - command on',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - command on',
            'FH.6000.[SS] SAFETY SIGNALS.AGV velocity active zone',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - executed',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - executed',
            'FH.6000.[WI] - WARNING INFORMATION.Warning Information - General Control - Fans Inactive',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance right',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance left',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[SS] SAFETY SIGNALS.Scanners active zones',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - automatic permission (#)',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - safety interlock',
            'FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active',
            'FH.6000.[GS] GENERAL SIGNALS.Manual Mode active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Automatic Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Manual Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - Safety - Scanners Safety Zones Muted',
        ]


class Formica_02(Formica20220104):
    def columns(self):
        return [
            'FH.6000.[ENC] - Energy Signals.Momentary power consumption',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - command on',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - command on',
            'FH.6000.[SS] SAFETY SIGNALS.AGV velocity active zone',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - executed',
            'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - executed',
            'FH.6000.[WI] - WARNING INFORMATION.Warning Information - General Control - Fans Inactive',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance right',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance left',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[SS] SAFETY SIGNALS.Scanners active zones',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - automatic permission (#)',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - safety interlock',
            'FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active',
            'FH.6000.[GS] GENERAL SIGNALS.Manual Mode active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Automatic Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Manual Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - Safety - Scanners Safety Zones Muted',
        ]


class Formica_04(Formica20220104):
    def columns(self):
        return [
            'FH.6000.[ENC] - Energy Signals.Momentary power consumption',
            'FH.6000.[WI] - WARNING INFORMATION.Warning Information - General Control - Fans Inactive',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance right',
            'FH.6000.[ODS] - Odometry Signals.Cumulative distance left',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Blue (Forbot has no LED so this signal is inactive)',
            'FH.6000.[SS] SAFETY SIGNALS.Scanners active zones',
            'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission',
            'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - automatic permission (#)',
            'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - safety interlock',
            'FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active',
            'FH.6000.[GS] GENERAL SIGNALS.Manual Mode active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Automatic Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Manual Mode Active',
            'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - Safety - Scanners Safety Zones Muted',
        ]


################################

class Husky_005(Husky):
    def columns(self):
        return [
            'power',
            'temperature',
            'velocity_x',
            'orientation_z',
            'velocity',
            'velocity_y',
            'velocity_z',
            'position_z',
            'angular_y',
            'angular_z',
            'orientation_w',
            'orientation_x',
            'linear_acceleration_y',
            'orientation_y',
            'linear_acceleration_x',
            'angular',
        ]


class Husky_01(Husky):
    def columns(self):
        return [
            'power',
            'position_z',
            'angular_y',
            'angular_z',
            'orientation_w',
            'orientation_x',
            'linear_acceleration_y',
            'orientation_y',
            'linear_acceleration_x',
            'angular',
        ]


class Husky_02(Husky):
    def columns(self):
        return [
            'power',
            'linear_acceleration_y',
            'orientation_y',
            'linear_acceleration_x',
            'angular',
        ]


class Husky_04(Husky):
    def columns(self):
        return [
            'power',
            'angular',
        ]


################################
class IEEEBattery_005(IEEEBattery):
    def columns(self):
        return [
            'MPC',
            'Ambient Temperature [°C]',
            'Heating Power CAN [kW]',
            'Battery Temperature [°C]',
            'max. Battery Temperature [°C]',
            'Elevation [m]',
            'Velocity [km/h]',
            'Longitudinal Acceleration [m/s^2]',
            'Throttle [%]',
            'Motor Torque [Nm]',
        ]


class IEEEBattery_03(IEEEBattery):
    def columns(self):
        return [
            'MPC',
            'Velocity [km/h]',
            'Longitudinal Acceleration [m/s^2]',
            'Throttle [%]',
            'Motor Torque [Nm]',
        ]


class IEEEBattery_04(IEEEBattery):
    def columns(self):
        return [
            'MPC',
            'Longitudinal Acceleration [m/s^2]',
            'Throttle [%]',
            'Motor Torque [Nm]',
        ]


base_path_datasets_ = "c:\\datasets\\"


DatasetInputData.implementations[Datasets.EagleOne] = (EagleOne, os.path.join(base_path_datasets_, "cobot1"))
DatasetInputData.implementations[Datasets.Formica20220104] = (
    Formica20220104, os.path.join(base_path_datasets_, "cobot2"))
DatasetInputData.implementations[Datasets.IEEEBattery] = (
    IEEEBattery, os.path.join(base_path_datasets_, "battery_ieee"))
DatasetInputData.implementations[Datasets.Husky] = (Husky, os.path.join(base_path_datasets_, "husky", "trials"))

DatasetInputData.implementations[Datasets.Formica_005] = (Formica_005, os.path.join(base_path_datasets_, "cobot2"))
DatasetInputData.implementations[Datasets.Formica_01] = (Formica_01, os.path.join(base_path_datasets_, "cobot2"))
DatasetInputData.implementations[Datasets.Formica_02] = (Formica_02, os.path.join(base_path_datasets_, "cobot2"))
DatasetInputData.implementations[Datasets.Formica_04] = (Formica_04, os.path.join(base_path_datasets_, "cobot2"))

DatasetInputData.implementations[Datasets.IEEEBattery_005] = (
    IEEEBattery_005, os.path.join(base_path_datasets_, "battery_ieee"))
DatasetInputData.implementations[Datasets.IEEEBattery_03] = (
    IEEEBattery_03, os.path.join(base_path_datasets_, "battery_ieee"))
DatasetInputData.implementations[Datasets.IEEEBattery_04] = (
    IEEEBattery_04, os.path.join(base_path_datasets_, "battery_ieee"))

DatasetInputData.implementations[Datasets.Husky_005] = (Husky_005, os.path.join(base_path_datasets_, "husky", "trials"))
DatasetInputData.implementations[Datasets.Husky_01] = (Husky_01, os.path.join(base_path_datasets_, "husky", "trials"))
DatasetInputData.implementations[Datasets.Husky_02] = (Husky_02, os.path.join(base_path_datasets_, "husky", "trials"))
DatasetInputData.implementations[Datasets.Husky_04] = (Husky_04, os.path.join(base_path_datasets_, "husky", "trials"))


class CoBot202210Data(DatasetInputData):
    MPC_COLUMN = "FH.6000.[ENS] - Energy Signals.Momentary power consumption"

    @staticmethod
    def _add_computed_columns(input_data):
        input_data["[COMPUTED] - DRIVE ACC L"] = input_data["FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.ActualSpeed_L"].diff().fillna(0)
        input_data["[COMPUTED] - DRIVE ACC R"] = input_data["FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.ActualSpeed_R"].diff().fillna(0)
        input_data["[COMPUTED] - Natural Navigation ACC"] = input_data["FH.6000.[NNS] - Natural Navigation Signals.Speed"].diff().fillna(0)
        input_data["[COMPUTED] - Odometry SPEED L"] = input_data["FH.6000.[ODS] - Odometry Signals.Cumulative distance left"].diff().fillna(0)
        input_data["[COMPUTED] - Odometry SPEED R"] = input_data["FH.6000.[ODS] - Odometry Signals.Cumulative distance right"].diff().fillna(0)
        return input_data

    def __init__(self, path, kwargs):
        self.path = path
        with open(os.path.join(self.path, "metadata.json")) as f:
            self.metadata = json.load(f)
        self.columns = [k for k, v in self.metadata.items() if k != "timestamp"]
        self.columns.remove(CoBot202210Data.MPC_COLUMN)
        self.columns.insert(0, CoBot202210Data.MPC_COLUMN)

        self.run_1 = self._add_computed_columns(pd.read_csv(os.path.join(self.path, "data_1.csv")))[self.columns]
        self.run_2_3_4 = self._add_computed_columns(pd.read_csv(os.path.join(self.path, "data_2_3_4.csv")))[self.columns]
        self.run_5_6 = self._add_computed_columns(pd.read_csv(os.path.join(self.path, "data_5_6.csv")))[self.columns]

        all_the_data = pd.concat([self.run_1, self.run_2_3_4, self.run_5_6])
        self.mpc_correlations = all_the_data.corr().iloc[0, :]
        self._minmax = dict()
        for c in self.columns:
            md = self.metadata[c]
            if md["type"] != "bool":
                self._minmax[c] = (all_the_data[c].min(), all_the_data[c].max())

        run1_len = len(self.run_1)
        self.train_data = pd.concat([self.run_1.iloc[:int(0.8 * run1_len), :], self.run_2_3_4], ignore_index=True)
        self.test_data = pd.concat([self.run_1.iloc[int(0.8 * run1_len):, :], self.run_5_6], ignore_index=True)

        print(f"Loaded Cobot202210: len(train)={len(self.train_data)}, len(test)={len(self.test_data)}")

    def minmax(self):
        for c in self.columns:
            md = self.metadata[c]
            if md["type"] != "bool":
                _min = self._minmax[c][0]
                _max = self._minmax[c][1]
                if (_max - _min) > 0:
                    self.train_data[c] = (self.train_data[c] - _min) / (_max - _min)
                    self.test_data[c] = (self.test_data[c] - _min) / (_max - _min)
                else:
                    self.train_data[c] = 0.5
                    self.test_data[c] = 0.5
        self.original_train_data = self.train_data.copy(deep=True)
        self.original_test_data = self.test_data.copy(deep=True)

    def apply_weights(self, weights):
        for i in range(len(weights)):
            self.test_data.iloc[:, i] = weights[i] * self.test_data.iloc[:, i]
            self.train_data.iloc[:, i] = weights[i] * self.train_data.iloc[:, i]

    def standardize(self):
        for c in self.columns:
            md = self.metadata[c]
            if md["type"] != "bool":
                self.train_data[c] = (self.train_data[c] - md["mean"]) / md["std"]
                self.test_data[c] = (self.test_data[c] - md["mean"]) / md["std"]

    def channel_names(self):
        return ["train", "test"]

    def channel(self, channel_name):
        if channel_name == "train":
            return channel_name, self.columns, self.train_data.to_numpy(), self.original_train_data.to_numpy()
        elif channel_name == "test":
            return channel_name, self.columns, self.test_data.to_numpy(), self.original_test_data.to_numpy()
        else:
            raise f"Invalid channel name {channel_name}. Only 'train' and 'test' allowed"



class CoBot20230708(DatasetInputData):
    LABEL_COLUMN = "WHEEL_CHANGE"

    def __init__(self, path, kwargs):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        base_dir = 'c:\\projekty\\cobot_july_august\\'
        sciezka_csv = base_dir + 'concatenated_with_wheels_change_changing_columns.csv'

        # Wczytaj plik CSV
        all_the_data = pd.read_csv(sciezka_csv)
        all_the_data = all_the_data.drop(columns=['timestamp', 'isoTimestamp', 'FH.6000.[TS] TIME STAMP.Time stamp'])

        self.columns = all_the_data.columns.tolist()
        self.columns.remove(CoBot20230708.LABEL_COLUMN)
        self.columns.insert(0, CoBot20230708.LABEL_COLUMN)
        all_the_data = all_the_data[self.columns]

        # Znajdź kolumny, które nie są boolean
        kolumny_do_standaryzacji = [col for col in all_the_data.columns if not np.issubdtype(all_the_data[col].dtype, np.bool)]

        # Standaryzuj dane tylko dla kolumn nie będących boolean
        scaler = StandardScaler()
        all_the_data[kolumny_do_standaryzacji] = scaler.fit_transform(all_the_data[kolumny_do_standaryzacji])

        self.train_data = all_the_data.iloc[:(len(all_the_data) // 2), :]
        self.test_data = all_the_data.iloc[(len(all_the_data) // 2):, :]
        print(f"Loaded Cobot20230708: len(train)={len(self.train_data)}, len(test)={len(self.test_data)}")

    def channel_names(self):
        return ["train", "test"]

    def channel(self, channel_name):
        if channel_name == "train":
            return channel_name, self.columns, self.train_data.to_numpy()
        elif channel_name == "test":
            return channel_name, self.columns, self.test_data.to_numpy()
        else:
            raise f"Invalid channel name {channel_name}. Only 'train' and 'test' allowed"

DatasetInputData.implementations[Datasets.CoBot20230708] = (CoBot20230708, "nope")


class FormicaNew(DatasetInputData):
    COLUMNS = [
        'FH.6000.[ENC] - Energy Signals.Momentary power consumption',

        'FH.6000.[ENC] - Energy Signals.Battery cell voltage',
        'FH.6000.[AI] - ALARM INFORMATION.Alarm Information - Safety - Circuit Opened',
        'FH.6000.[AI] - ALARM INFORMATION.Alarm Information - Safety - Front Scanner Protective Zone Active',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - executed',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Lock - in progress (#)',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - executed',
        'FH.6000.[G1BS] GROUP 1 - BRAKES SIGNALS.Brake Release - in progress (#)',
        'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - command on',
        'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - manual permission',
        'FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.Left drive activate - safety interlock',
        'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - command on',
        'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - manual permission',
        'FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.Right drive activate - safety interlock',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Down - executed',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Down - in progress (#)',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - automatic permission (#)',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - executed',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - in progress (#)',
        'FH.6000.[G3LPS] GROUP 3 - LIGTING PLATE SIGNALS.Lifting plate Up - safety interlock',
        'FH.6000.[GS] GENERAL SIGNALS.Automatic Mode active', 'FH.6000.[GS] GENERAL SIGNALS.Manual Mode active',
        'FH.6000.[GS] GENERAL SIGNALS.PLC fault active',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) - Red (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Blue (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 1 (left) – Green (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Blue (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Green (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LED RGB Strip 2 (right) – Red (Forbot has no LED so this signal is inactive)',
        'FH.6000.[LED] LED STATUS.LEG RGB - External control active (Forbot has no LED so this signal is inactive)',
        'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Automatic Mode Active',
        'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - General Control - Manual Mode Active',
        'FH.6000.[MI] - MESSAGE INFORMATION.Message Information - Safety - Scanners Safety Zones Muted',
        'FH.6000.[ODS] - Odometry Signals.Cumulative distance left',
        'FH.6000.[ODS] - Odometry Signals.Cumulative distance right',
        'FH.6000.[ODS] - Odometry Signals.Momentary frequency of left encoder pulses',
        'FH.6000.[ODS] - Odometry Signals.Momentary frequency of right encoder pulses',
        'FH.6000.[SS] SAFETY SIGNALS.AGV velocity active zone',
        'FH.6000.[SS] SAFETY SIGNALS.Safety circuit closed',
        'FH.6000.[SS] SAFETY SIGNALS.Scanners active zones',
        'FH.6000.[WI] - WARNING INFORMATION.Warning Information - General Control - Fans Inactive',
        'FH.6000.[WI] - WARNING INFORMATION.Warning Information - Safety - Front Scanner Warning Zone Active',
        'FH.6000.[WI] - WARNING INFORMATION.Warning Information - Safety - Rear Scanner Warning Zone Active',
    ]

    def __init__(self, path: str, kwargs):
        self.path = path

    def columns(self):
        return FormicaNew.COLUMNS

    def _channel_names(self):
        return [os.path.splitext(os.path.basename(channel_file))[0] for channel_file in
                _files_as_pattern(self.path, "dump_from_simulation_?.csv")]

    def _channel(self, channel_name):
        channel_file = os.path.join(self.path, f"{channel_name}.csv")

        input_data = pd.read_csv(channel_file)

        input_data = input_data[self.columns()].astype(dtype=np.float32)

        whole_data = input_data.to_numpy()

        return channel_name, input_data.columns.tolist(), whole_data


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


if __name__ == "!!__main__":
    base = "c:\\datasets\\cobot_newest\\10\\"
    all_data = None
    for csv_file in [
        # "dump_from_simulation_new_2_20221010_10_10_00.csv", "dump_from_simulation_new_3_20221010_15_05_00.csv", "dump_from_simulation_new_4_20221011_08_45_00.csv"
        "dump_from_simulation_new_5_20221011_11_04_00.csv", "dump_from_simulation_new_6_20221011_11_40_00.csv"
    ]:
        input_data = pd.read_csv(os.path.join(base, csv_file))
        if all_data is None:
            all_data = input_data
        else:
            all_data = pd.concat([all_data, input_data])
    all_data.to_csv(os.path.join(base, "data_5_6.csv"), index=None)
    exit()

if __name__ == "!!!__main__":

    def columns_present_everywhere_and_changing(path):
        files = _files_as_pattern(path, "data*.csv")
        columns = dict()
        for channel_file in files:
            channel_name = os.path.splitext(os.path.basename(channel_file))[0]
            input_data = pd.read_csv(channel_file)

            input_data["[COMPUTED] - DRIVE ACC L"] = input_data["FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.ActualSpeed_L"].diff().fillna(0)
            input_data["[COMPUTED] - DRIVE ACC R"] = input_data["FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.ActualSpeed_R"].diff().fillna(0)
            input_data["[COMPUTED] - Natural Navigation ACC"] = input_data["FH.6000.[NNS] - Natural Navigation Signals.Speed"].diff().fillna(0)
            input_data["[COMPUTED] - Odometry SPEED L"] = input_data["FH.6000.[ODS] - Odometry Signals.Cumulative distance left"].diff().fillna(0)
            input_data["[COMPUTED] - Odometry SPEED R"] = input_data["FH.6000.[ODS] - Odometry Signals.Cumulative distance right"].diff().fillna(0)

            for col in input_data.columns.tolist():
                key = (col, input_data.dtypes[col])
                if key not in columns:
                    columns[key] = (0, 0)
                cnt, unique = columns[key]
                unique = max(unique, input_data[col].nunique())
                columns[key] = (cnt + 1, unique)
        result = {k[0]: str(k[1]) for k, (cnt, uniq) in columns.items() if cnt == len(files) and uniq > 1}
        result.pop("isoTimestamp")
        result.pop("FH.6000.[TS] TIME STAMP.Time stamp")
        return result


    for base_path in [
        # "c:\\datasets\\cobot_newest\\08\\",
        "c:\\datasets\\cobot_newest\\10\\"]:
        columns = columns_present_everywhere_and_changing(base_path)

        with open(os.path.join(base_path, "metadata.json"), "w") as f:
            all_the_data = []
            for csv_file in _files_as_pattern(base_path, "data*.csv"):
                input_data = pd.read_csv(csv_file)

                print(f"{csv_file}: {len(input_data)} rows")

                input_data["[COMPUTED] - DRIVE ACC L"] = input_data["FH.6000.[G1LDS] GROUP 1 - LEFT DRIVE SIGNALS.ActualSpeed_L"].diff().fillna(0)
                input_data["[COMPUTED] - DRIVE ACC R"] = input_data["FH.6000.[G2RDS] GROUP 2 - RIGHT DRIVE SIGNALS.ActualSpeed_R"].diff().fillna(0)
                input_data["[COMPUTED] - Natural Navigation ACC"] = input_data["FH.6000.[NNS] - Natural Navigation Signals.Speed"].diff().fillna(0)
                input_data["[COMPUTED] - Odometry SPEED L"] = input_data["FH.6000.[ODS] - Odometry Signals.Cumulative distance left"].diff().fillna(0)
                input_data["[COMPUTED] - Odometry SPEED R"] = input_data["FH.6000.[ODS] - Odometry Signals.Cumulative distance right"].diff().fillna(0)

                print(4*input_data['FH.6000.[WS] WEIGHT STATUSES.Weight statuses - front left strain gauge weight'].unique())

                import plotly.express as px
                import pandas as pd

                input_data['FH.6000.[ENS] - Energy Signals.Battery cell voltage'] /= 100
                input_data['FH.6000.[WS] WEIGHT STATUSES.Weight statuses - front left strain gauge weight'] *= 4

                fig = px.line(input_data[['FH.6000.[WS] WEIGHT STATUSES.Weight statuses - front left strain gauge weight',
                                          'FH.6000.[ENS] - Energy Signals.Momentary power consumption',
                                          'FH.6000.[ENS] - Energy Signals.Battery cell voltage']])
                fig.write_html(f"{csv_file}.html")

                all_the_data.append(input_data[columns.keys()])
            all_the_data = pd.concat(all_the_data, ignore_index=True)

            metadata_result = dict()
            for col, _type in columns.items():
                if _type == "bool":
                    metadata_result[col] = {"type": _type}
                else:
                    std = all_the_data[col].std()
                    mean = all_the_data[col].mean()
                    metadata_result[col] = {"type": _type, "std": std, "mean": mean}
                    all_the_data[col] = (all_the_data[col] - mean) / std

            json.dump(metadata_result, f, indent=4)

            all_the_data.to_csv(os.path.join(base_path, "_data_all_normalized.csv"), index=False)

            numeric_columns = [k for k, v in columns.items() if v not in ["bool"]]

            all_the_data.corr()
            # print(all_the_data)

            ew = pd.ExcelWriter(os.path.join(base_path, "__corr.xlsx"), engine='xlsxwriter')
            for method in ["pearson", "kendall", "spearman"]:
                corr = all_the_data.corr(method=method)
                corr.to_excel(ew, sheet_name=method)
            ew.close()


