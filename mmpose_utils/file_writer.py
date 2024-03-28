import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """
    Helper class to handel json data.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_json(data, file_path):
    """Saves data dictionaries

    Args: the location and the name of json file to be saved.
    """

    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=NpEncoder)


def load_json(file_path):
    """Loads data dictionaries

    Args:
        file_path: A string indicating the location and the name of json file to be loaded.

    Returns:
        A dictionary containing the dataset annotations.
    """

    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
