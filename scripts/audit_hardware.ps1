[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    poetry run python scripts/audit_hardware.py
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    }
} finally {
    Pop-Location
}
