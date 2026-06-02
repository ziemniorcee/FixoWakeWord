param(
    [int]$Port = 8765
)

$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
if (-not (Test-Path $adb)) {
    throw "Nie znaleziono adb: $adb. Zainstaluj Android SDK Platform Tools."
}

& $adb start-server | Out-Null
$devices = @(& $adb devices | Select-Object -Skip 1 | Where-Object { $_ -match "\tdevice$" })
if ($devices.Count -ne 1) {
    throw "Podłącz dokładnie jeden tablet z włączonym debugowaniem USB. Wykryte urządzenia: $($devices.Count)"
}

& $adb reverse "tcp:$Port" "tcp:$Port"
if ($LASTEXITCODE -ne 0) {
    throw "Nie udało się skonfigurować adb reverse."
}

$url = "http://localhost:$Port"
Start-Job -ScriptBlock {
    param($Adb, $Url)
    Start-Sleep -Milliseconds 800
    & $Adb shell am start -a android.intent.action.VIEW -d $Url | Out-Null
} -ArgumentList $adb, $url | Out-Null
Write-Host "Panel otworzył się na tablecie: http://localhost:$Port"
Write-Host "Przy pierwszym nagraniu zaakceptuj dostęp do mikrofonu."

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$codexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (Test-Path $venvPython) {
    & $venvPython (Join-Path $PSScriptRoot "tablet_recorder.py") --port $Port
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python (Join-Path $PSScriptRoot "tablet_recorder.py") --port $Port
} elseif (Test-Path $codexPython) {
    & $codexPython (Join-Path $PSScriptRoot "tablet_recorder.py") --port $Port
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 (Join-Path $PSScriptRoot "tablet_recorder.py") --port $Port
} else {
    throw "Nie znaleziono Pythona. Zainstaluj Python 3 lub utwórz .venv w katalogu projektu."
}
