import pickle as pkl
import os


def load_data(pickle_file):
    if not os.path.isfile(pickle_file):
        raise FileNotFoundError(f"File '{pickle_file}' does not exist.")

    try:
        with open(pickle_file, 'rb') as f:
            data = pkl.load(f)
        return data
    except pkl.UnpicklingError as e:
        raise pkl.UnpicklingError(f"Could not unpickle file '{pickle_file}': {e}")
    except Exception as e:
        raise Exception(f"Error loading file '{pickle_file}': {e}")