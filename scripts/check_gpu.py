"""Verify an actual CUDA computation, not just that a driver is installed."""
import json
import platform
from importlib.metadata import version

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit(
            'CUDA is unavailable. Run scripts/setup_vlm.ps1 and check nvidia-smi.'
        )
    sample = torch.randn(128, 128, device='cuda')
    result = sample @ sample.T
    torch.cuda.synchronize()
    if not torch.isfinite(result).all().item():
        raise SystemExit('CUDA matrix multiplication returned invalid values.')
    props = torch.cuda.get_device_properties(0)
    info = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_cuda_runtime': torch.version.cuda,
        'gpu': props.name,
        'vram_gib': round(props.total_memory / 2**30, 2),
        'compute_capability': list(torch.cuda.get_device_capability(0)),
        'bfloat16_supported': torch.cuda.is_bf16_supported(),
        'cuda_matrix_multiplication': 'passed',
    }
    for package in ('torchvision', 'transformers', 'accelerate', 'bitsandbytes'):
        info[package] = version(package)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
