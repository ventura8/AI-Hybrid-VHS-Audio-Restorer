param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InformationPreference = "Continue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot
$VenvPy = "$repoRoot\.venv\Scripts\python.exe"
$PoetryVersion = "2.4.1"

if (-not (Test-Path $VenvPy)) {
    throw "Virtual environment interpreter not found at $VenvPy. Run install_dependencies.ps1 first."
}

# Ensure Poetry commands are executed against this repository's .venv,
# even if the caller has another venv activated in the shell.
if ($env:VIRTUAL_ENV) {
    Write-Information "Detected active environment at '$($env:VIRTUAL_ENV)'. Clearing it for local pipeline commands."
    Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
}

$env:POETRY_VIRTUALENVS_IN_PROJECT = "true"
$env:POETRY_VIRTUALENVS_CREATE = "false"
$env:POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON = "false"

$PowerShellExe = $null
if (Get-Command "pwsh" -ErrorAction SilentlyContinue) {
    $PowerShellExe = "pwsh"
}
elseif (Get-Command "powershell" -ErrorAction SilentlyContinue) {
    $PowerShellExe = "powershell"
}
else {
    throw "Could not find a PowerShell executable (pwsh or powershell)."
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Information "==> $Name"
    & $Action
}

function Invoke-CheckedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Native tools can emit informational stderr lines; gate failures on exit code.
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Executable $($Arguments -join ' ')"
    }
}

function Invoke-PoetryCommand {
    param([string[]]$Arguments)

    Invoke-CheckedCommand $VenvPy (@("-m", "poetry") + $Arguments)
}

function Get-CoverageThreshold {
    $thresholdRaw = & $VenvPy -c "from tests.tooling.threshold_policy import get_coverage_threshold; print(f'{get_coverage_threshold():.2f}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve coverage threshold from tests.tooling.threshold_policy.get_coverage_threshold()."
    }

    $thresholdText = ($thresholdRaw | Out-String).Trim()
    $parsedThreshold = 0.0
    $isValidThreshold = [double]::TryParse(
        $thresholdText,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$parsedThreshold
    )
    if (-not $isValidThreshold) {
        throw "Coverage threshold is not numeric: '$thresholdText'"
    }

    return [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:F2}", $parsedThreshold)
}

function Initialize-Poetry {
    try {
        Invoke-PoetryCommand @("--version")
    }
    catch {
        Write-Information "Poetry was not found. Installing Poetry..."
        Invoke-CheckedCommand $VenvPy @("-m", "pip", "install", "poetry==$PoetryVersion")
        Invoke-PoetryCommand @("--version")
    }
}

$pipelineFailed = $false
$failureMessage = $null
$script:coverageThreshold = $null

try {
    Invoke-Step "Install Poetry" {
        Initialize-Poetry
    }

    Invoke-Step "Install dev dependencies" {
        Invoke-PoetryCommand @("install", "-v", "--only", "dev", "--no-root")
    }

    Invoke-Step "Resolve coverage threshold policy" {
        $script:coverageThreshold = Get-CoverageThreshold
        Write-Information "Using coverage threshold: $script:coverageThreshold"
    }

    Invoke-Step "Run PowerShell lint" {
        $lintCommand = @"
& '$repoRoot\.github\scripts\Invoke-PowerShellLint.ps1' -ScriptPaths @(
    '$repoRoot\install_dependencies.ps1',
    '$repoRoot\run_pipeline_locally.ps1',
    '$repoRoot\.github\scripts\Invoke-PowerShellLint.ps1'
)
"@
        Invoke-CheckedCommand $PowerShellExe @(
            "-NoProfile",
            "-Command",
            $lintCommand
        )
    }

    Invoke-Step "Run Ruff" {
        Invoke-PoetryCommand @(
            "run",
            "ruff",
            "check",
            "modules",
            "tests",
            "restore_audio_hybrid.py",
            "scripts/apply_patches.py"
        )
    }

    Invoke-Step "Run Flake8" {
        Invoke-PoetryCommand @("run", "flake8", "modules", "tests", "restore_audio_hybrid.py", "scripts/apply_patches.py")
    }

    Invoke-Step "Run Pylint" {
        Invoke-PoetryCommand @(
            "run",
            "pylint",
            "--errors-only",
            "modules",
            "tests",
            "restore_audio_hybrid.py",
            "scripts/apply_patches.py"
        )
    }

    Invoke-Step "Run tests with coverage" {
        Invoke-PoetryCommand @(
            "run",
            "pytest",
            "-o",
            "addopts=",
            "--cov=restore_audio_hybrid",
            "--cov=modules",
            "--cov-branch",
            "--cov-report=xml",
            "--cov-report=json",
            "--cov-report=term",
            "--cov-fail-under=$script:coverageThreshold",
            "tests/"
        )
    }

    Invoke-Step "Enforce strict per-file coverage" {
        Invoke-PoetryCommand @("run", "python", "tests/tooling/quality_gate.py", "coverage.json", "--threshold", $script:coverageThreshold)
    }
}
catch {
    $pipelineFailed = $true
    $currentFailureMessage = $_.Exception.Message
    if (-not $failureMessage) {
        $failureMessage = $currentFailureMessage
    }
    Write-Error $currentFailureMessage -ErrorAction Continue
}

try {
    Invoke-Step "Generate coverage badge and summary" {
        if (-not (Test-Path "coverage.xml")) {
            throw "coverage.xml was not generated by the test step."
        }

        Invoke-PoetryCommand @("run", "python", "tests/tooling/badge_report.py", "coverage.xml")
    }
}
catch {
    $pipelineFailed = $true
    $currentFailureMessage = $_.Exception.Message
    if (-not $failureMessage) {
        $failureMessage = $currentFailureMessage
    }
    Write-Error $currentFailureMessage -ErrorAction Continue
}

if ($pipelineFailed) {
    exit 1
}

Write-Information "Local pipeline completed successfully."
