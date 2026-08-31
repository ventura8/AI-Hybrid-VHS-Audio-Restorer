<#
.SYNOPSIS
    Builds a native Windows .exe installer for AI Hybrid VHS Audio Restorer.

.DESCRIPTION
    Packages the repository source into a self-extracting executable installer (.exe)
    using 7-Zip SFX or embedded assembly, configuring the launcher start.bat and automated setup.

.PARAMETER Tag
    Release tag (e.g., v1.1.0).

.PARAMETER OutputDir
    Target directory for the generated .exe installer.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "release-assets"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$InformationPreference = "Continue"

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location -Path $repoRoot

$targetDir = Join-Path $repoRoot $OutputDir
if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
}

$installerName = "AI-Hybrid-VHS-Audio-Restorer-$Tag-windows.exe"
$outputPath = Join-Path $targetDir $installerName

Write-Information "Creating Windows executable: $outputPath"

$stageDir = Join-Path $env:TEMP ("ai-vhs-restorer-" + [System.Guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $stageDir -Force | Out-Null

try {
    # 1. Export clean git tree
    $archiveZip = Join-Path $stageDir "source.zip"
    git archive --format=zip --output="$archiveZip" HEAD
    Expand-Archive -Path $archiveZip -DestinationPath (Join-Path $stageDir "app") -Force
    Remove-Item $archiveZip -Force

    $appDir = Join-Path $stageDir "app"

    Write-Information "Generating standalone Windows executable launcher..."
    $csharpCode = @"
using System;
using System.IO;
using System.IO.Compression;
using System.Diagnostics;
using System.Text;

namespace Launcher {
    class Program {
        static int Main(string[] args) {
            try {
                string appDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "AI-Hybrid-VHS-Audio-Restorer");
                Directory.CreateDirectory(appDir);

                string launcherBat = Path.Combine(appDir, "start.bat");
                string pythonExe = Path.Combine(appDir, ".venv", "Scripts", "python.exe");

                // Extract bundle payload if not present or on version update
                string versionFile = Path.Combine(appDir, ".installed_version");
                string currentVersion = "$Tag";
                bool needsExtract = !File.Exists(launcherBat) || !File.Exists(versionFile) || File.ReadAllText(versionFile).Trim() != currentVersion;

                if (needsExtract) {
                    Console.WriteLine("Extracting AI Hybrid VHS Audio Restorer ($Tag) to: " + appDir);
                    byte[] payload = Convert.FromBase64String("BASE64_PAYLOAD_PLACEHOLDER");
                    string tempZip = Path.Combine(Path.GetTempPath(), "ai_vhs_" + Guid.NewGuid().ToString("N") + ".zip");
                    File.WriteAllBytes(tempZip, payload);
                    ZipFile.ExtractToDirectory(tempZip, appDir);
                    File.Delete(tempZip);
                    File.WriteAllText(versionFile, currentVersion);
                }

                // Check environment
                if (!File.Exists(pythonExe)) {
                    Console.WriteLine("First-time setup: Initializing environment...");
                    string setupScript = Path.Combine(appDir, "install_dependencies.ps1");
                    ProcessStartInfo psiSetup = new ProcessStartInfo {
                        FileName = "powershell.exe",
                        Arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + setupScript + "\"",
                        WorkingDirectory = appDir,
                        UseShellExecute = false
                    };
                    Process pSetup = Process.Start(psiSetup);
                    pSetup.WaitForExit();
                    if (pSetup.ExitCode != 0) {
                        Console.Error.WriteLine("Environment initialization failed with code: " + pSetup.ExitCode);
                        return pSetup.ExitCode;
                    }
                }

                // Run main restoration pipeline with forwarded arguments
                string mainScript = Path.Combine(appDir, "restore_audio_hybrid.py");
                StringBuilder argBuilder = new StringBuilder();
                argBuilder.Append("\"").Append(mainScript).Append("\"");
                foreach (string arg in args) {
                    argBuilder.Append(" \"").Append(arg.Replace("\"", "\\\"")).Append("\"");
                }

                ProcessStartInfo psi = new ProcessStartInfo {
                    FileName = pythonExe,
                    Arguments = argBuilder.ToString(),
                    WorkingDirectory = appDir,
                    UseShellExecute = false
                };

                Process p = Process.Start(psi);
                p.WaitForExit();
                return p.ExitCode;
            } catch (Exception ex) {
                Console.Error.WriteLine("Execution error: " + ex.Message);
                return 1;
            }
        }
    }
}
"@
    $bundleZip = Join-Path $stageDir "bundle.zip"
    [System.IO.Compression.ZipFile]::CreateFromDirectory($appDir, $bundleZip, [System.IO.Compression.CompressionLevel]::Optimal, $false)
    $base64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($bundleZip))
    $csharpSource = $csharpCode.Replace("BASE64_PAYLOAD_PLACEHOLDER", $base64)
    $sourceFile = Join-Path $stageDir "Launcher.cs"
    Set-Content -Path $sourceFile -Value $csharpSource -Encoding utf8

    # Compile standalone executable using .NET CSC compiler (available on all Windows systems)
    $csc = "$env:WINDIR\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
    if (-not (Test-Path $csc)) {
        $csc = "$env:WINDIR\Microsoft.NET\Framework\v4.0.30319\csc.exe"
    }

    if (Test-Path $csc) {
        Write-Information "Compiling with .NET Framework compiler: $csc"
        & $csc /target:exe /platform:anycpu /out:"$outputPath" /r:System.IO.Compression.FileSystem.dll /r:System.IO.Compression.dll "$sourceFile" | Out-Null
    }
    else {
        # Fallback to Windows PowerShell 5.1 which supports OutputType ConsoleApplication
        Write-Information "Compiling with Windows PowerShell 5.1..."
        powershell.exe -NoProfile -Command "Add-Type -TypeDefinition (Get-Content '$sourceFile' -Raw) -OutputAssembly '$outputPath' -OutputType ConsoleApplication -ReferencedAssemblies @('System.IO.Compression.FileSystem.dll', 'System.IO.Compression.dll')"
    }

    if (-not (Test-Path $outputPath) -or (Get-Item $outputPath).Length -eq 0) {
        throw "Failed to generate $outputPath"
    }

    Write-Information "Successfully generated Windows executable: $outputPath"
}
finally {
    Remove-Item $stageDir -Recurse -Force -ErrorAction SilentlyContinue
}
