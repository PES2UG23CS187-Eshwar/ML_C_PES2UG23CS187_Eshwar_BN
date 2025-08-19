import numpy as np

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    """
    target_col = data[:, -1]
    values, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()

    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return float(np.round(entropy, 4))


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the weighted average entropy of a specific attribute.
    """
    values, counts = np.unique(data[:, attribute], return_counts=True)
    total = len(data)
    avg_info = 0.0

    for v, count in zip(values, counts):
        subset = data[data[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (count / total) * subset_entropy

    return float(np.round(avg_info, 4))


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Information gain = Dataset entropy - Attribute’s avg info.
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    gain = dataset_entropy - avg_info
    return float(np.round(gain, 4))


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Compute information gain for all attributes and select the best one.
    Returns (dictionary_of_gains, index_of_best_attribute).
    """
    num_attributes = data.shape[1] - 1  # exclude target column
    gains = {}

    for i in range(num_attributes):
        gains[i] = get_information_gain(data, i)

    best_attr = max(gains, key=gains.get)
    return gains, best_attr