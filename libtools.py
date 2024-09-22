#!/usr/bin/env python3

import torch

def device_mapper() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")