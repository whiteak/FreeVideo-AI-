# Installs ai-toolkit (github.com/ostris/ai-toolkit) for LoRA Studio's "Train on this PC" (MiniMax H3 LoRAs).
# Fork addition. Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\install-lora-trainer.ps1 [-Target C:\Projects\ai-toolkit]
# It needs git and Python 3.10+ (3.12 recommended). It does not touch Eclipse's own Python.
# After it finishes: accept the MiniMax H3 licence at https://huggingface.co/Comfy-Org/MiniMax-H3 and run
#   C:\Projects\ai-toolkit\venv\Scripts\hf.exe auth login
# if the download asks for it. LoRA Studio finds C:\Projects\ai-toolkit on its own; for another folder set it
# in LoRA Studio > Train > Trainer setup.
param(
    [string]$Target = 'C:\Projects\ai-toolkit',
    [string]$Torch = 'torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0',
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cu130'
)
$ErrorActionPreference = 'Stop'

function Find-Python {
    foreach ($candidate in @(@('py', '-3.12'), @('py', '-3.11'), @('python'))) {
        try {
            $version = & $candidate[0] $candidate[1..9] -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -and [version]$version -ge [version]'3.10') { return ,$candidate }
        } catch { }
    }
    $fork = Join-Path $PSScriptRoot '..\envs\unified\python.exe'
    if (Test-Path $fork) { return ,@((Resolve-Path $fork).Path) }
    throw 'Python 3.10 or newer was not found. Install Python 3.12 from python.org (tick "Add to PATH") and run this again.'
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'git was not found. Install Git for Windows and run this again.' }

if (Test-Path (Join-Path $Target 'run.py')) {
    Write-Host "Updating ai-toolkit in $Target"
    git -C $Target pull --ff-only
} else {
    Write-Host "Cloning ai-toolkit into $Target"
    New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
    git clone https://github.com/ostris/ai-toolkit.git $Target
}

$venvPython = Join-Path $Target 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    $python = Find-Python
    Write-Host "Creating the trainer's own Python environment with $($python -join ' ')"
    & $python[0] $python[1..9] -m venv (Join-Path $Target 'venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}

Write-Host 'Installing PyTorch (CUDA 13, for RTX 50-series and older cards)'
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --no-cache-dir $Torch.Split(' ') --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch did not install.' }
Write-Host 'Installing ai-toolkit requirements'
& $venvPython -m pip install -r (Join-Path $Target 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'The ai-toolkit requirements did not install.' }

& $venvPython -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if (-not (Test-Path (Join-Path $Target 'extensions_built_in\diffusion_models\minimax_h3'))) {
    Write-Warning 'This ai-toolkit has no MiniMax H3 support yet. Run this script again later to update it.'
}
Write-Host ''
Write-Host "Done. ai-toolkit is in $Target."
Write-Host 'Next: accept the licence at https://huggingface.co/Comfy-Org/MiniMax-H3, then log in with a Read token:'
Write-Host "  $(Join-Path $Target 'venv\Scripts\hf.exe') auth login"
Write-Host "Then open LoRA Studio (the LoRA entry in Eclipse's top bar) > Train."
