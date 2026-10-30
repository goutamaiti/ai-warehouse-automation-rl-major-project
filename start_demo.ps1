# Starts the whole demo: FastAPI backend + React portal, on fixed ports.
#
#   Right-click -> "Run with PowerShell", or:
#     .\start_demo.ps1            start the demo
#     .\start_demo.ps1 -Verify    prove everything works first (about 5 minutes)
#
# Fixed ports are deliberate: other copies of this project may be running on the
# usual Vite ports (5173, 5180), so the demo uses 5190, and the backend is told to
# accept that origin - otherwise the browser blocks the API call and the portal
# silently falls back to "Replaying recorded episodes".

param([switch]$Verify)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$apiPort  = 8000
$uiPort   = 5190
$uiOrigin = "http://localhost:$uiPort"

Write-Host ""
Write-Host "  Warehouse RL - demo launcher" -ForegroundColor White
Write-Host "  ----------------------------"

# --- preflight -------------------------------------------------------------
try {
    python -c "import torch, stable_baselines3, fastapi, gymnasium" 2>$null
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "  [ok] python packages" -ForegroundColor Green
} catch {
    Write-Host "  [!!] missing python packages - run:  python -m pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "$root\frontend\node_modules\vite")) {
    Write-Host "  [!!] frontend not installed - run:  npm install --prefix frontend" -ForegroundColor Red
    exit 1
}
Write-Host "  [ok] frontend packages" -ForegroundColor Green

# A stale backend or portal on these ports is the most common way a demo goes
# wrong: the browser talks to an old process running old code.
foreach ($port in @($apiPort, $uiPort)) {
    $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        $owner = ($busy | Select-Object -First 1).OwningProcess
        Write-Host "  [!!] port $port is already in use (pid $owner). Stop it first:  Stop-Process -Id $owner" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [ok] ports $apiPort and $uiPort are free" -ForegroundColor Green

if ($Verify) {
    Write-Host "  [..] verifying the Live floor (about 1 minute)..."
    python scripts\verify_live_floor.py | Select-String "VERDICT|scenarios passed"
    if ($LASTEXITCODE -ne 0) { Write-Host "  [!!] Live-floor verification FAILED - do not demo until fixed" -ForegroundColor Red; exit 1 }
    Write-Host "  [..] verifying single-robot runs and all 540 published episodes (a few minutes)..."
    python scripts\verify_single_robot.py | Select-String "TOTAL|PROBLEM"
    if ($LASTEXITCODE -ne 0) { Write-Host "  [!!] single-robot verification FAILED - do not demo until fixed" -ForegroundColor Red; exit 1 }
    Write-Host "  [ok] verification passed" -ForegroundColor Green
}

# point the dashboard at the local API
Set-Content -Path "$root\frontend\.env.local" -Value "VITE_API_BASE=http://127.0.0.1:$apiPort" -Encoding utf8

# --- backend ---------------------------------------------------------------
$api = Start-Process -PassThru -WindowStyle Minimized powershell -ArgumentList @(
    "-NoProfile", "-Command",
    "Set-Location '$root'; `$env:ALLOWED_ORIGINS='$uiOrigin,http://127.0.0.1:$uiPort'; " +
    "python -m uvicorn backend.main:app --host 127.0.0.1 --port $apiPort"
)
Write-Host "  [..] backend starting on http://127.0.0.1:$apiPort (pid $($api.Id))"

$ready = $false
foreach ($i in 1..40) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-RestMethod "http://127.0.0.1:$apiPort/api/health" -TimeoutSec 2
        if ($r.status -eq "ok") { $ready = $true; break }
    } catch { }
}
if ($ready) {
    Write-Host "  [ok] backend healthy - controllers: $(($r.controllers | ForEach-Object { $_.name }) -join ', ')" -ForegroundColor Green
} else {
    Write-Host "  [!!] backend did not come up - check the minimized window" -ForegroundColor Red
}

# --- frontend --------------------------------------------------------------
$ui = Start-Process -PassThru -WindowStyle Minimized powershell -ArgumentList @(
    "-NoProfile", "-Command",
    "Set-Location '$root'; npm run dev --prefix frontend -- --port $uiPort --strictPort"
)
Write-Host "  [..] portal starting on $uiOrigin (pid $($ui.Id))"
Start-Sleep -Seconds 6

Start-Process $uiOrigin
Write-Host ""
Write-Host "  Portal : $uiOrigin" -ForegroundColor White
Write-Host "  API    : http://127.0.0.1:$apiPort/docs" -ForegroundColor White
Write-Host ""
Write-Host "  The badge at the top right must read 'Backend connected'." -ForegroundColor Yellow
Write-Host "  Demo script: docs/demo-script.md" -ForegroundColor DarkGray
Write-Host "  To stop:  Stop-Process -Id $($api.Id),$($ui.Id)" -ForegroundColor DarkGray
Write-Host ""
