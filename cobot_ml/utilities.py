import numpy as np


def get_windows_limits_idxs(y: np.ndarray):
    """
    Returns tuple containing indices of windows beginning and end
    :param y: Array with labels
    :return tuple with 2 lists, first one containing indices of windows start
    and the second one with indices of windows stop.
    """
    anomalies_start = []
    anomalies_stop = []
    previous = False
    for idx, is_anomaly in enumerate(y):
        if not previous and is_anomaly:  # 0 --> 1
            anomalies_start.append(idx)
        elif previous and not is_anomaly:  # 1 --> 0
            anomalies_stop.append(idx - 1)
        previous = is_anomaly
    if previous:
        anomalies_stop.append(len(y) - 1)
    return anomalies_start, anomalies_stop


def create_01_mask_in_ranges(desired_mask_length: int, ranges):
    """
    Creates a 1D 0-1 mask with 1-s in ranges and 0 elsewhere.
    :param desired_mask_length: length of mask to be created
    :param ranges: pairs of range boundaries
    :return 1D int ndarray
    """
    result = np.zeros(shape=(desired_mask_length,), dtype=int)
    for _range in ranges:
        result[_range[0]: _range[1]] = 1
    return result
