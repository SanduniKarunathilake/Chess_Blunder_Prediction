# start_backend.ps1
# Starts the Flask backend server for Chess Analysis Tools
# Usage: .\start_backend.ps1

$venv = Join-Path $PSScriptRoot "ML\.venv\Scripts\python.exe"
$app  = Join-Path $PSScriptRoot "backend\app.py"

if (-not (Test-Path $venv)) {
    Write-Error "Virtual environment not found at: $venv"
    exit 1
}

if (-not (Test-Path $app)) {
    Write-Error "Backend app not found at: $app"
    exit 1
}

Write-Host "[*] Starting Chess Analysis Tools backend..."
Write-Host "[*] API will be available at http://127.0.0.1:5000"
Write-Host "[*] Press Ctrl+C to stop."
Write-Host ""

Set-Location (Join-Path $PSScriptRoot "backend")
& $venv $app
