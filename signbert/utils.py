import json
import gc
import torch

def my_import(name):
    """
    Dynamically import a module or object in Python.

    Given the full name of a module or an object within a module (as a string),
    this function imports it and returns the module or object.

    Parameters:
    name (str): The full name of the module or object to import, e.g., 'numpy.array'.

    Returns:
    module/object: The imported module or object.
    """
    # Split the full name by dots to separate the module and object names
    components = name.split('.')
    # Import the top-level module
    mod = __import__(components[0])
    # Traverse through the module hierarchy to get to the desired object
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod

def read_json(fpath):
    with open(fpath, 'r') as fid:
        data = json.load(fid)
    return data

def read_txt_as_list(fpath):
    with open(fpath, 'r') as fid:
        data = fid.read().split('\n')
    return data

def dict_to_json_file(dict, fpath):
    with open(fpath, 'w') as fid:
        json.dump(dict, fid)


def _num_active_cuda_tensors():
    """
    Returns all tensors initialized on cuda devices
    """
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.device.type == "cuda":
                count += 1
        except:
            pass
    return count