param(
    [string]$ClientPath = "C:\Dev\untitled\service-assistant\client",
    [double]$MinRecall = 0.70,
    [int]$Epochs = 24,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$trainingRoot = $PSScriptRoot
$clientRoot = [System.IO.Path]::GetFullPath($ClientPath)
$clientWakeword = Join-Path $clientRoot "wakeword"
$checkpoint = Join-Path $trainingRoot "checkpoints\fikso_cnn.pt"
$clientCheckpoint = Join-Path $clientWakeword "checkpoints\fikso_cnn.pt"
$calibration = Join-Path $trainingRoot "results\calibration.json"
$exporter = Join-Path $clientWakeword "export_android_model.py"
$typescriptModule = Join-Path $clientRoot "modules\wake-word\index.ts"
$kotlinModule = Join-Path $clientRoot "modules\wake-word\android\src\main\java\expo\modules\wakeword\WakeWordModule.kt"
$androidAsset = Join-Path $clientRoot "modules\wake-word\android\src\main\assets\fikso_cnn.bin"

foreach ($requiredPath in @($exporter, $typescriptModule, $kotlinModule)) {
    if (-not (Test-Path $requiredPath)) {
        throw "Missing required file: $requiredPath"
    }
}

function Test-Python {
    param(
        [string]$Command,
        [string[]]$Prefix = @()
    )
    try {
        & $Command @Prefix -c "import sys, numpy, torch; assert sys.version_info >= (3, 10)" 2>$null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$pythonCommand = $null
$pythonPrefix = @()
$venvPython = Join-Path $trainingRoot ".venv\Scripts\python.exe"
$codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if ((Test-Path $venvPython) -and (Test-Python $venvPython)) {
    $pythonCommand = $venvPython
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-Python "python")) {
    $pythonCommand = "python"
} elseif ((Test-Path $codexPython) -and (Test-Python $codexPython)) {
    $pythonCommand = $codexPython
} elseif ((Get-Command py -ErrorAction SilentlyContinue) -and (Test-Python "py" @("-3"))) {
    $pythonCommand = "py"
    $pythonPrefix = @("-3")
} else {
    throw "No Python 3 interpreter with numpy and torch found. Install dependencies with: python -m pip install -r requirements.txt"
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $pythonCommand @pythonPrefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $Arguments"
    }
}

function Replace-Required {
    param(
        [string]$Path,
        [string]$Pattern,
        [string]$Replacement
    )
    $content = [System.IO.File]::ReadAllText($Path)
    if (-not [regex]::IsMatch($content, $Pattern)) {
        throw "Expected fragment not found in file: $Path"
    }
    $updated = [regex]::Replace($content, $Pattern, $Replacement)
    if ($updated -ne $content) {
        [System.IO.File]::WriteAllText($Path, $updated, [System.Text.UTF8Encoding]::new($false))
    }
}

Push-Location $trainingRoot
try {
    if (-not $SkipTraining) {
        Write-Host "==> Training model"
        Invoke-Python "train.py" "--epochs" "$Epochs"
    } elseif (-not (Test-Path $checkpoint)) {
        throw "Checkpoint not found: $checkpoint"
    }

    Write-Host "==> Calibrating streaming settings"
    $minRecallText = $MinRecall.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
    Invoke-Python "calibrate.py" "--min-recall" $minRecallText

    $calibrationResult = Get-Content $calibration -Raw | ConvertFrom-Json
    $threshold = [double]$calibrationResult.selected.threshold
    $requiredHits = [int]$calibrationResult.selected.required_hits
    $thresholdText = $threshold.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)

    Write-Host "==> Copying checkpoint"
    New-Item -ItemType Directory -Path (Split-Path $clientCheckpoint) -Force | Out-Null
    Copy-Item $checkpoint $clientCheckpoint -Force

    Write-Host "==> Exporting Android asset"
    Push-Location $clientWakeword
    try {
        Invoke-Python "export_android_model.py"
    } finally {
        Pop-Location
    }

    Write-Host "==> Synchronizing detection settings"
    Replace-Required $typescriptModule '(?m)^(\s*threshold\s*=\s*)[0-9.]+' ('${1}' + $thresholdText)
    Replace-Required $typescriptModule '(?m)^(\s*requiredHits\s*=\s*)\d+' ('${1}' + $requiredHits)
    Replace-Required $kotlinModule '(?m)^private const val MAX_STREAMING_THRESHOLD = [0-9.]+f$' 'private const val MAX_STREAMING_THRESHOLD = 1.0f'

    Write-Host ""
    Write-Host "Done."
    Write-Host "Checkpoint: $clientCheckpoint"
    Write-Host "Android asset: $androidAsset"
    Write-Host "Streaming: threshold=$thresholdText requiredHits=$requiredHits"
    Write-Host "Rebuild and install the Android application."
} finally {
    Pop-Location
}
