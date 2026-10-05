<#
.SYNOPSIS
    Stores the SonarQube Cloud analysis token as the repository secret SONAR_TOKEN and re-runs CI.

.DESCRIPTION
    One command for the token step of the SonarQube Cloud setup: the token is typed (masked) or
    pasted at the prompt, never written to a file or a shell history, handed to `gh secret set`
    through its standard input, and the latest failed CI run of the current branch is re-run so
    the first analysis lands. Generate the token on sonarcloud.io (My Account > Security).

.PARAMETER Repository
    The GitHub repository (owner/name). Defaults to this project.

.EXAMPLE
    .\scripts\set_sonar_token.ps1
#>
[CmdletBinding()]
param(
    [string]$Repository = "ventura8/AI-Hybrid-VHS-Audio-Restorer"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is not installed or not on PATH."
}

$secure = Read-Host -Prompt "Paste the SonarQube Cloud token" -AsSecureString
$pointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $token = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
}
finally {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
}
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "No token given."
}

$token | gh secret set SONAR_TOKEN --repo $Repository
if ($LASTEXITCODE -ne 0) {
    throw "gh secret set failed with exit code $LASTEXITCODE."
}
$token = $null
Write-Output"SONAR_TOKEN stored in $Repository."

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
$runId = gh run list --repo $Repository --branch $branch --limit 1 --json databaseId,conclusion --jq '.[0] | select(.conclusion == "failure") | .databaseId'
if ([string]::IsNullOrWhiteSpace($runId)) {
    Write-Output"No failed CI run on $branch to re-run; the next push will analyse."
    exit 0
}
gh run rerun $runId --repo $Repository --failed
Write-Output"Re-running the failed jobs of run $runId on $branch."
