"""Picking the torch device, shared by the VLM and the detectors."""

import platform

import torch

def mps_usable() -> bool:
    # torch 2.15's MPS kernels crash outright on macOS 14 (an uncatchable NSException), so require macOS 15+
    if not torch.backends.mps.is_available():
        return False
    major = platform.mac_ver()[0].split(".")[0]
    return major.isdigit() and int(major) >= 15


def pick_device(name: str = "auto") -> torch.device:
    """auto: cuda if available, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_usable():
        return torch.device("mps")
    return torch.device("cpu")
