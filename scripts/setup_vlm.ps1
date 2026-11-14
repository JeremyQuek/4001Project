$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }
    $env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache/pip'
    & ./.venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & ./.venv/Scripts/python.exe -m pip install -r requirements-vlm.txt
    if ($LASTEXITCODE -ne 0) { throw 'VLM dependencies could not be installed.' }
    & ./.venv/Scripts/python.exe scripts/check_gpu.py
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed.' }
}
finally { Pop-Location }
