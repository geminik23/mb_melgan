import torch.nn as nn

def weights_init_norm(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1: 
        nn.init.norm_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.norm_(m.weight)

def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1: 
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.xavier_uniform_(m.weight)

def to_device(obj, device):
    if isinstance(obj, list):
        return [to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(to_device(list(obj), device))
    elif isinstance(obj, dict):
        retval = dict()
        for key, value in obj.items():
            retval[to_device(key, device)] = to_device(value, device)
        return retval 
    elif hasattr(obj, "to"): 
        return obj.to(device)
    else: 
        return obj