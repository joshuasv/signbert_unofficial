import json
import gc
import torch

def my_import(name):
    components = name.split('.')
    mod = __import__(components[0])
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