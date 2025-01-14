import os
import random

import numpy as np
import torch

from .log import getLogger
LOGGER = getLogger(__name__)


def worker_init_function(worker_id):
    seed = torch.utils.data.get_worker_info().seed
    np_seed = seed
    if np_seed > 2**32 - 1:
        np_seed = seed % (2**32 - 1) - 526 + int(worker_id)
    np.random.seed(np_seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_randstate_magic_numbers(device=None):
    """Use these to check that randstate advances the same accross runs"""
    np_int = np.random.randint(0, int(1e6))
    random_int = random.randint(0, int(1e6))
    torch_cpu_int = torch.randint(int(1e6), (1,), device="cpu").item()
    if device is not None:
        torch_device_int = torch.randint(int(1e6), (1,), device=device).item()
    else:
        torch_device_int = None
    return (random_int, np_int, torch_cpu_int, torch_device_int)


class PytorchRNGState(torch.nn.Module):
    """Class to save/restore PRNG states that masquarades as nn.Module for checkpointing"""

    __RANDOM_PRNG_STATE__ = "__random_prng_state__"
    __NUMPY_PRNG_STATE__ = "__numpy_prng_state__"
    __TORCH_PRNG_STATE__ = "__torch_prng_state__"
    __CUDA_PRNG_STATE__ = "__cuda_prng_state__"

    def __init__(self, seed=42):
        super(PytorchRNGState, self).__init__()
        self.register_buffer("initial_seed", torch.tensor(seed, dtype=torch.long), persistent=True)
        self.register_buffer("already_seeded", torch.tensor(False, dtype=torch.bool), persistent=True)

    @property
    def device(self):
        return self.initial_seed.device

    def seed_everything(self):
        if torch.all(self.already_seeded):  # sticky for checkpointing; do only once
            return
        else:
            seed = int(self.initial_seed.item())
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            self.already_seeded = torch.logical_not(self.already_seeded)  # keep it as buffer i.e. tensor
            LOGGER.info(f"Seed set to {seed}")

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super(PytorchRNGState, self).state_dict(destination, prefix, keep_vars)
        state_dict[self.__RANDOM_PRNG_STATE__] = random.getstate()
        state_dict[self.__NUMPY_PRNG_STATE__] = np.random.get_state()
        state_dict[self.__TORCH_PRNG_STATE__] = torch.random.get_rng_state()
        if torch.cuda.is_available() and "cuda" in str(self.device):
            cuda_state = torch.cuda.get_rng_state(self.device)
            state_dict[self.__CUDA_PRNG_STATE__] = cuda_state
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        LOGGER.debug(f"({self.device}) [CUDA={torch.cuda.is_available()}] state_dict: {state_dict.keys()}")
        random.setstate(state_dict.pop(self.__RANDOM_PRNG_STATE__))
        np.random.set_state(state_dict.pop(self.__NUMPY_PRNG_STATE__))
        torch.set_rng_state(state_dict.pop(self.__TORCH_PRNG_STATE__))
        LOGGER.debug(f"Restored state to python process and ")
        if strict:
            if torch.cuda.is_available() and "cuda" in str(self.device) and self.__CUDA_PRNG_STATE__ not in state_dict:
                raise RuntimeError(f"Error in restoring CUDA PRNG state: state missing")
            if self.__CUDA_PRNG_STATE__ in state_dict and not torch.cuda.is_available():
                raise RuntimeError(f"Error in restoring CUDA PRNG state: CUDA not available")
            if self.__CUDA_PRNG_STATE__ in state_dict and "cuda" not in str(self.device):
                raise RuntimeError(f"Error in restoring CUDA PRNG state: not on CUDA device")
        if self.__CUDA_PRNG_STATE__ in state_dict and torch.cuda.is_available() and "cuda" in str(self.device):
            torch.cuda.set_rng_state(state_dict.pop(self.__CUDA_PRNG_STATE__), self.device)
        return super(PytorchRNGState, self).load_state_dict(state_dict, strict)
