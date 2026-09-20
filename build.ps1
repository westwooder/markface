# Build a self-contained markface bundle. Run from the project root.
#   powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $py -m pip install -q --upgrade pip
& $py -m pip install -q -r requirements.txt -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "Fetching models..." -ForegroundColor Cyan
& $py tools\fetch_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model download failed. In mainland China try:" -ForegroundColor Yellow
    Write-Host "  .venv\Scripts\python.exe tools\fetch_models.py --mirror hf-mirror.com"
    throw "models missing"
}

if (Test-Path "$root\dist\markface") {
    Write-Host "Cleaning previous build..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force "$root\dist\markface"
}

Write-Host "Running PyInstaller..." -ForegroundColor Cyan
& $py -m PyInstaller --noconfirm --clean markface.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$exe = Join-Path $root "dist\markface\markface.exe"
if (-not (Test-Path $exe)) { throw "expected $exe to exist" }

$size = (Get-ChildItem -Recurse "$root\dist\markface" | Measure-Object -Sum Length).Sum / 1GB
Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host ("  Output: dist\markface\markface.exe")
Write-Host ("  Bundle size: {0:N2} GB" -f $size)
Write-Host ""
Write-Host "Ship the whole dist\markface folder. The target machine needs no Python."
