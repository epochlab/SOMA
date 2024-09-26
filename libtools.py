#!/usr/bin/env python3

import yaml
import torch

def device_mapper(f:str=None) -> torch.device:
    if f != None: torch.device(f)
    elif torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def load_profile(element:str):
    with open('profiles.yml') as f:
        data =  yaml.safe_load(f)['elements']
    p_dict = {element['name']: element for element in data}
    return p_dict.get(element)