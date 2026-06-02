param(
    [int]$Port = 8765
)

$pidFile = Join-Path $PSScriptRoot ".tablet_recorder.pid"
if (Test-Path $pidFile) {
    $serverPid = [int](Get-Content $pidFile)
    $connection = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.OwningProcess -eq $serverPid }
    if ($connection) {
        Stop-Process -Id $serverPid -Force
        Write-Host "Zatrzymano serwer nagrywania (PID $serverPid)."
    } else {
        Write-Host "Usunięto nieaktywny plik PID."
    }
    Remove-Item $pidFile -Force
} else {
    Write-Host "Serwer nagrywania nie był uruchomiony."
}

$adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
if (Test-Path $adb) {
    & $adb reverse --remove "tcp:$Port" | Out-Null
}
