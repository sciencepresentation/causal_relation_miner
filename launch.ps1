<#
 Causal Relationship Extractor - PowerShell Launcher (auto-setup)
 Creates a venv if missing, installs requirements, and runs Streamlit.
#>

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Causal Relationship Extractor" -ForegroundColor Green
Write-Host "  Starting Streamlit App..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Move to the folder where this script resides
Set-Location -Path $PSScriptRoot

$venvDir = Join-Path $PSScriptRoot "myenv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

# Create venv if missing
if (-not (Test-Path $venvPython)) {
	Write-Host "[INFO] Creating Python virtual environment at: $venvDir" -ForegroundColor Cyan
	# Prefer 'py' launcher on Windows if available
	$pyLauncher = (Get-Command py -ErrorAction SilentlyContinue)
	if ($pyLauncher) {
		& py -3 -m venv $venvDir
	} else {
		& python -m venv $venvDir
	}
}

if (-not (Test-Path $venvPython)) {
	Write-Host "[ERROR] Could not find or create venv Python at: $venvPython" -ForegroundColor Red
	Write-Host "        Please install Python 3.9+ and try again." -ForegroundColor Red
	Read-Host "Press Enter to exit"
	exit 1
}

Write-Host "[INFO] Upgrading pip..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

if (Test-Path (Join-Path $PSScriptRoot 'requirements.txt')) {
	Write-Host "[INFO] Installing required packages (this can take a few minutes the first time)..." -ForegroundColor Cyan
	& $venvPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
}

# Launch Streamlit app
Write-Host "[INFO] Starting Streamlit app..." -ForegroundColor Cyan
& $venvPython -m streamlit run app.py

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
