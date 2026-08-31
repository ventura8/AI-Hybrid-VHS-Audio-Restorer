# === AI Hybrid VHS Audio Restorer Installer ===
# Installs runtime dependencies only via Poetry (verbose mode), plus local FFmpeg.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InformationPreference = "Continue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

Set-Location -Path $PSScriptRoot

Write-Information "=== Setting up Hybrid AI Audio Environment (Poetry Runtime Mode) ==="

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

# 1. Check for Python
try {
    $pyVersion = python --version 2>&1
    Write-Information "Found Python: $pyVersion"
}
catch {
    throw "Python not found in PATH. Please install Python 3.12 manually and re-run this script."
}

# 2. Create Virtual Environment
Write-Information "`nStep 1: Setting up Python virtual environment (.venv)..."
[version]$minVersion = "3.12.0"
[version]$maxVersionExclusive = "3.13.0"
$resolvedPyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve Python interpreter version."
}

[version]$currentVersion = $resolvedPyVersion
if ($currentVersion -lt $minVersion -or $currentVersion -ge $maxVersionExclusive) {
    throw "Python version must be >= 3.12.0 and < 3.13.0. Found $currentVersion"
}

if (-not (Test-Path "$PSScriptRoot\.venv\Scripts\python.exe")) {
    Invoke-CheckedCommand "python" @("-m", "venv", ".venv")
    Write-Information "Created .venv virtual environment."
}
else {
    Write-Information "Virtual environment already exists."
}

$VenvPy = "$PSScriptRoot\.venv\Scripts\python.exe"
$VenvScripts = "$PSScriptRoot\.venv\Scripts"

if (-not (Test-Path $VenvPy)) {
    throw "Virtual environment interpreter not found at $VenvPy"
}

$resolvedPyVersion = & $VenvPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to resolve virtual environment interpreter version."
}
[version]$currentVersion = $resolvedPyVersion
if ($currentVersion -lt $minVersion -or $currentVersion -ge $maxVersionExclusive) {
    throw "Virtual environment Python version must be >= 3.12.0 and < 3.13.0. Found $currentVersion"
}

# 3. Install FFmpeg Setup (Portable & Enforced Local)
Write-Information "`nStep 2: Checking local FFmpeg..."
$localFFmpegPath = "$VenvScripts\ffmpeg.exe"

if (-not (Test-Path $localFFmpegPath)) {
    Write-Information "Local FFmpeg not found. Downloading full portable build..."

    $url = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.1/ffmpeg-9.0.1-essentials_build.zip"
    $urlFallback = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.1-essentials_build.zip"
    $expectedSha256 = "fec81ae03971d9dd4be3ebe02e263bd2ec1d789483f931bdba5f5715e65da2e9"
    $zip = "$PSScriptRoot\ffmpeg.zip"
    $temp = "$PSScriptRoot\temp_ffmpeg"

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -UserAgent "Mozilla/5.0"
    }
    catch {
        Write-Warning "Primary FFmpeg source failed. Trying pinned fallback..."
        Invoke-WebRequest -Uri $urlFallback -OutFile $zip -UseBasicParsing -UserAgent "Mozilla/5.0"
    }

    $actualSha256 = ""
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    $fileStream = [System.IO.File]::OpenRead($zip)
    try {
        $hashBytes = $hasher.ComputeHash($fileStream)
        $actualSha256 = [System.BitConverter]::ToString($hashBytes).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $fileStream.Dispose()
        $hasher.Dispose()
    }

    if ($actualSha256 -ne $expectedSha256) {
        throw "FFmpeg archive checksum mismatch. Expected $expectedSha256 but got $actualSha256"
    }

    Write-Information "Extracting FFmpeg..."
    Expand-Archive -Path $zip -DestinationPath $temp -Force

    $bin = Get-ChildItem -Path $temp -Recurse -Filter "ffmpeg.exe" | Select-Object -ExpandProperty DirectoryName -First 1
    if ([string]::IsNullOrWhiteSpace($bin)) {
        throw "FFmpeg extraction failed: ffmpeg.exe was not found in the extracted archive."
    }
    Copy-Item "$bin\ffmpeg.exe" $VenvScripts -Force
    Copy-Item "$bin\ffprobe.exe" $VenvScripts -Force

    Remove-Item $zip -Force
    Remove-Item $temp -Recurse -Force

    Write-Information "FFmpeg installed to .venv\Scripts."
}
else {
    Write-Information "Local FFmpeg already installed in .venv."
}

# 4. Install runtime dependencies with Poetry (verbose)
Write-Information "`nStep 3: Installing runtime dependencies with Poetry..."
Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "--upgrade", "pip")
Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "poetry==2.4.1")

# Prevent Poetry from inheriting an unrelated active environment (for example, an external "venv").
if ($env:VIRTUAL_ENV) {
    Write-Information "Detected active environment at '$($env:VIRTUAL_ENV)'. Clearing it for installer-scoped Poetry commands."
    Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
}

$env:POETRY_VIRTUALENVS_IN_PROJECT = "true"
$env:POETRY_VIRTUALENVS_CREATE = "false"
$env:POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON = "false"

Invoke-CheckedCommand $VenvPy @("-m", "poetry", "config", "--local", "virtualenvs.in-project", "true")
Invoke-CheckedCommand $VenvPy @("-m", "poetry", "config", "--local", "virtualenvs.create", "false")

if (-not (Test-Path "$PSScriptRoot\poetry.lock")) {
    Write-Information "poetry.lock not found. Generating lock file..."
    Invoke-CheckedCommand $VenvPy @("-m", "poetry", "lock", "-v", "--no-interaction")
}

$env:POETRY_REQUESTS_TIMEOUT = "600"
$env:PIP_DEFAULT_TIMEOUT = "600"

$maxAttempts = 3
$attempt = 1
while ($attempt -le $maxAttempts) {
    try {
        Write-Information "Running poetry install (attempt $attempt of $maxAttempts)..."
        Invoke-CheckedCommand $VenvPy @("-m", "poetry", "install", "-v", "--with", "ml", "--without", "dev", "--no-root", "--no-interaction")
        break
    }
    catch {
        if ($attempt -ge $maxAttempts) {
            throw $_
        }
        Write-Warning "Poetry install attempt $attempt failed ($($_.Exception.Message)). Retrying in 10 seconds..."
        Start-Sleep -Seconds 10
        $attempt++
    }
}

# Resemble-Enhance is a VCS install, so pip re-clones and rebuilds it on every
# run unless it is skipped explicitly. Prefer the installed copy: rebuild only
# when the recorded commit does not match the pinned ref (a version string alone
# does not change between revisions), or when FORCE_RESEMBLE_REBUILD=1.
$ResembleRepo = "https://github.com/daswer123/resemble-enhance-windows.git"
$ResembleRef = "270d8da4ea7c0efc960c52d605b75c0458b708d0"

function Get-InstalledResembleRef {
    param([string]$PythonExe)

    $probe = 'import json, importlib.metadata as metadata; dist = next((item for item in metadata.distributions() if str(item.metadata[''Name'']).lower() == ''resemble-enhance''), None); raw = dist.read_text(''direct_url.json'') if dist else None; print(json.loads(raw).get(''vcs_info'', {}).get(''commit_id'', '''') if raw else '''')'

    $result = & $PythonExe "-c" $probe 2>$null
    if ($LASTEXITCODE -ne 0) {
        return ""
    }
    return ($result | Select-Object -Last 1)
}

Write-Information "Checking Resemble-Enhance runtime package..."
$installedRef = Get-InstalledResembleRef -PythonExe $VenvPy
if ($env:FORCE_RESEMBLE_REBUILD -ne "1" -and $installedRef -eq $ResembleRef) {
    Write-Information "Resemble-Enhance already installed at pinned commit $($ResembleRef.Substring(0, 12)); skipping source build."
}
else {
    Write-Information "Installing Resemble-Enhance runtime package (building from source)..."
    Invoke-CheckedCommand $VenvPy @(
        "-m",
        "pip",
        "install",
        "-v",
        "git+${ResembleRepo}@${ResembleRef}",
        "--no-deps",
        "--force-reinstall"
    )
}

Write-Information "Applying runtime patches (DeepSpeed removal + Torchaudio fixes)..."
Invoke-CheckedCommand $VenvPy @("scripts/apply_patches.py")

# 5. Create directories
Write-Information "`nStep 4: Creating project structure..."
New-Item -ItemType Directory -Force -Path "input" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "temp_work" | Out-Null

# 6. Create launcher
Write-Information "Step 5: Creating launcher..."
$batContent = @"
@echo off
set "SCRIPT_DIR=%~dp0"
set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Virtual environment not found. Setting up environment automatically...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_dependencies.ps1"
)
if not exist "%PYTHON_EXE%" (
    echo Failed to configure environment.
    pause
    exit /b 1
)
"%PYTHON_EXE%" "%SCRIPT_DIR%restore_audio_hybrid.py" %*
pause
"@
Set-Content -Path "start.bat" -Value $batContent -Encoding Ascii

Write-Information "`n=== Installation Complete! ==="
Write-Information "1. Put your video files in the input folder."
Write-Information "2. Double-click start.bat to run the Hybrid AI Cleaner."
