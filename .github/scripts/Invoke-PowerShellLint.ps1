param(
    [Parameter(Mandatory = $true)]
    [string[]]$ScriptPaths
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-LatestInstalledPssaVersion {
    $installed = Get-Module -ListAvailable -Name PSScriptAnalyzer | Sort-Object Version -Descending | Select-Object -First 1
    if ($null -eq $installed) {
        return $null
    }
    return $installed.Version
}


$psGallery = Get-PSRepository -Name PSGallery -ErrorAction SilentlyContinue
if ($null -eq $psGallery) {
    throw "PSGallery repository is not registered."
}

# Validate the scheme and host rather than an exact URL string. Hosted runners
# register PSGallery with a trailing slash or the v3 endpoint depending on the
# PowerShellGet version present, which made a byte-exact match fail CI
# intermittently. Checking the host still rejects any third-party gallery.
$sourceUri = $null
$parsed = [System.Uri]::TryCreate($psGallery.SourceLocation, [System.UriKind]::Absolute, [ref]$sourceUri)
if (-not $parsed -or $sourceUri.Scheme -ne "https" -or $sourceUri.Host -ne "www.powershellgallery.com") {
    throw "PSGallery is not configured with an approved source location: $($psGallery.SourceLocation)"
}

$restoreInstallationPolicy = $false
$originalInstallationPolicy = $null
if ($psGallery.InstallationPolicy -ne "Trusted") {
    $originalInstallationPolicy = $psGallery.InstallationPolicy
    $restoreInstallationPolicy = $true
    Set-PSRepository -Name PSGallery -InstallationPolicy Trusted
}

$installedVersion = Get-LatestInstalledPssaVersion
$galleryVersion = $null

try {
    try {
        $galleryVersion = (Find-Module PSScriptAnalyzer -Repository PSGallery -ErrorAction Stop).Version
    }
    catch {
        Write-Warning "Could not query PSGallery for PSScriptAnalyzer version. Falling back to installed version if available."
    }

    $needsInstall = ($null -eq $installedVersion)
    $needsUpdate = ($null -ne $galleryVersion -and $null -ne $installedVersion -and $galleryVersion -gt $installedVersion)
    if ($needsInstall -or $needsUpdate) {
        if ($needsInstall) {
            Write-Information "PSScriptAnalyzer was not found. Installing latest module..."
        }
        else {
            Write-Information "Updating PSScriptAnalyzer from $installedVersion to $galleryVersion..."
        }

        Install-Module PSScriptAnalyzer -Scope CurrentUser -Force -AllowClobber -Repository PSGallery
        $installedVersion = Get-LatestInstalledPssaVersion
    }

    if ($null -eq $installedVersion) {
        throw "PSScriptAnalyzer installation failed."
    }

    Import-Module PSScriptAnalyzer -RequiredVersion $installedVersion -Force

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
}
finally {
    if ($restoreInstallationPolicy) {
        Set-PSRepository -Name PSGallery -InstallationPolicy $originalInstallationPolicy
    }
}
