# === PR Review Tooling Installer ===
# Installs GitHub CLI and VS Code extensions used for PR comment workflows.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InformationPreference = "Continue"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Test-CommandAvailable {
    param([string]$Name)

    return [bool](Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Get-CombinedPathFromRegistry {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")

    if ([string]::IsNullOrWhiteSpace($machinePath) -and [string]::IsNullOrWhiteSpace($userPath)) {
        return $null
    }

    $combinedPath = @($machinePath, $userPath) -join ";"
    if ([string]::IsNullOrWhiteSpace($combinedPath)) {
        return $null
    }

    return $combinedPath
}

function Get-GitHubCliPath {
    if (Test-CommandAvailable "gh") {
        $ghCmd = Get-Command -Name "gh" -ErrorAction SilentlyContinue
        if ($ghCmd -and -not [string]::IsNullOrWhiteSpace($ghCmd.Source)) {
            return $ghCmd.Source
        }
    }

    $candidatePaths = @(
        (Join-Path $env:ProgramFiles "GitHub CLI\gh.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\GitHub CLI\gh.exe"),
        (Join-Path $env:USERPROFILE "scoop\shims\gh.exe")
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

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

function Install-GitHubCliIfMissing {
    $existingGhPath = Get-GitHubCliPath
    if ($existingGhPath) {
        Write-Information "GitHub CLI already installed."
        return
    }

    Write-Information "GitHub CLI not found. Installing..."

    if (Test-CommandAvailable "winget") {
        & winget @(
            "install",
            "--id",
            "GitHub.cli",
            "--exact",
            "--source",
            "winget",
            "--accept-package-agreements",
            "--accept-source-agreements"
        )

        if ($LASTEXITCODE -ne 0) {
            Write-Warning "winget install returned exit code $LASTEXITCODE. Checking whether GitHub CLI is already present."
            $wingetList = winget list --id GitHub.cli --exact 2>&1 | Out-String
            $wingetListExit = $LASTEXITCODE
            $looksInstalled = $wingetList -match "GitHub\.cli|GitHub CLI"
            if ($wingetListExit -ne 0 -and -not $looksInstalled) {
                throw "GitHub CLI installation failed and package was not detected via winget list."
            }
        }
    }
    elseif (Test-CommandAvailable "choco") {
        Invoke-CheckedCommand "choco" @("install", "gh", "-y")
    }
    elseif (Test-CommandAvailable "scoop") {
        Invoke-CheckedCommand "scoop" @("install", "gh")
    }
    else {
        throw "No supported package manager found. Install GitHub CLI manually: https://cli.github.com/"
    }

    $combinedPath = Get-CombinedPathFromRegistry
    if ($combinedPath) {
        $env:Path = $combinedPath
    }

    $ghPath = Get-GitHubCliPath
    if (-not $ghPath) {
        throw "GitHub CLI installation was attempted but 'gh' is still unavailable in PATH."
    }

    $ghFolder = Split-Path -Parent $ghPath
    if ($ghFolder -and -not ($env:Path -split ';' | Where-Object { $_ -eq $ghFolder })) {
        $env:Path = "$ghFolder;$env:Path"
    }

    Write-Information "GitHub CLI installation complete."
}

function Install-VsCodeExtensionIfMissing {
    param(
        [string]$ExtensionId,
        [string]$Description
    )

    if (-not (Test-CommandAvailable "code")) {
        Write-Warning "VS Code CLI ('code') not found. Skipping extension install for $ExtensionId ($Description)."
        return
    }

    $installedExtensions = code --list-extensions
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list VS Code extensions via 'code --list-extensions'."
    }

    $isInstalled = $installedExtensions | Where-Object {
        $_.Trim().ToLowerInvariant() -eq $ExtensionId.ToLowerInvariant()
    }

    if ($isInstalled) {
        Write-Information "VS Code extension already installed: $ExtensionId"
        return
    }

    Write-Information "Installing VS Code extension: $ExtensionId ($Description)"
    Invoke-CheckedCommand "code" @("--install-extension", $ExtensionId, "--force")
}

Write-Information "=== Installing PR Review Tooling ==="
Install-GitHubCliIfMissing
Install-VsCodeExtensionIfMissing -ExtensionId "eamodio.gitlens" -Description "GitLens"
Install-VsCodeExtensionIfMissing -ExtensionId "github.vscode-pull-request-github" -Description "GitHub Pull Requests"
Write-Information "PR review tooling installation step complete."
