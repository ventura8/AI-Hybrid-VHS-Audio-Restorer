param(
    [Parameter(Mandatory = $true)]
    [string[]]$ScriptPaths
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not (Get-Module -ListAvailable -Name PSScriptAnalyzer)) {
    Write-Information "PSScriptAnalyzer was not found. Installing module..."
    $psGallery = Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue
    if ($psGallery -and $psGallery.InstallationPolicy -ne "Trusted") {
        Set-PSRepository -Name PSGallery -InstallationPolicy Trusted
    }
    Install-Module PSScriptAnalyzer -Scope CurrentUser -Force -Repository PSGallery
}

$issues = foreach ($scriptPath in $ScriptPaths) {
    Invoke-ScriptAnalyzer -Path $scriptPath -Severity Warning,Error -Recurse:$false
}

if ($issues) {
    $issues |
        Select-Object ScriptName, Line, Severity, RuleName, Message |
        Format-Table -AutoSize |
        Out-String |
        Write-Output
    throw "PowerShell lint failed. Resolve all PSScriptAnalyzer warnings/errors."
}
