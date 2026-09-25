[CmdletBinding()]
param(
    # Private LEXIDE project (vendor drivers + generated Debug makefile).
    [string]$SourceProject = "$env:USERPROFILE\lexide\workspace_omega_v2\AcrylicPanCollector_lowlatency",
    # CMSIS Core include directory (ARM.CMSIS 5.9.0 pack).
    [string]$CmsisInclude = "$env:LOCALAPPDATA\Arm\Packs\ARM\CMSIS\5.9.0\CMSIS\Core\Include",
    # C file that replaces S_System/main.c (default: the LCD/switch demo).
    [string]$MainSource,
    [string]$Configuration = "Debug",
    [string]$StagingRoot
)

$ErrorActionPreference = "Stop"
$demoRoot = Split-Path -Parent $PSScriptRoot             # firmware/LcdSwitchDemo
$repoRoot = Split-Path -Parent (Split-Path -Parent $demoRoot)
$makeExe = "C:\LAPIS\LEXIDE\Utilities\Bin\make.exe"
if (-not $MainSource) { $MainSource = Join-Path $demoRoot "src\main_demo.c" }
if (-not (Test-Path -LiteralPath $MainSource -PathType Leaf)) { throw "Not found: $MainSource" }

foreach ($p in @($SourceProject, $CmsisInclude)) {
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { throw "Not found: $p" }
}
if (-not (Test-Path -LiteralPath $makeExe -PathType Leaf)) { throw "LEXIDE make.exe not found: $makeExe" }
if (-not $StagingRoot) {
    $StagingRoot = Join-Path $repoRoot (".local\firmware-build\" + (Get-Date -Format "yyyyMMdd_HHmmss") + "_lcdswitch")
}
$stagingFull = [System.IO.Path]::GetFullPath($StagingRoot)
if (Test-Path -LiteralPath $stagingFull) { throw "Staging path already exists: $stagingFull" }

$projectName = Split-Path -Leaf $SourceProject
$staged = Join-Path $stagingFull $projectName
New-Item -ItemType Directory -Path $staged -Force | Out-Null
Write-Host "Staging a disposable copy of $SourceProject"
Get-ChildItem -LiteralPath $SourceProject -Force | Copy-Item -Destination $staged -Recurse -Force

Copy-Item -LiteralPath $MainSource -Destination (Join-Path $staged "S_System\main.c") -Force

# The generated .res files and makefile carry absolute paths of whichever PC
# generated them (project root and CMSIS pack).  Re-point every project path at
# the staged copy and the CMSIS include at this PC's pack.
$buildDir = Join-Path $staged $Configuration
if (-not (Test-Path -LiteralPath (Join-Path $buildDir "makefile"))) { throw "Generated makefile not found in $buildDir" }
$projRe = '[A-Za-z]:[\\/]Users[\\/][^\\/"]+[\\/]lexide[\\/]workspace_omega_v2[\\/]' + [regex]::Escape($projectName)
$cmsisRe = '[A-Za-z]:/Users/[^/"]+/AppData/Local/Arm/Packs/ARM/CMSIS/[^/"]+/CMSIS/Core/Include'
$stagedFwd = $staged.Replace('\', '/')
$cmsisFwd = (Resolve-Path -LiteralPath $CmsisInclude).Path.Replace('\', '/')
# (-Include is silently ignored together with -LiteralPath in Windows PowerShell,
# which would rewrite binary objects too, so filter explicitly.)
$files = @(Get-ChildItem -LiteralPath $buildDir -Recurse -File |
    Where-Object { $_.Extension -in @(".res", ".mk") -or $_.Name -eq "makefile" })
foreach ($f in $files) {
    $t = [IO.File]::ReadAllText($f.FullName)
    $t2 = [regex]::Replace($t, $projRe, { param($m) if ($m.Value.Contains('\')) { $staged } else { $stagedFwd } })
    $t2 = [regex]::Replace($t2, $cmsisRe, { param($m) $cmsisFwd })
    if ($t2 -ne $t) { [IO.File]::WriteAllText($f.FullName, $t2, [Text.UTF8Encoding]::new($false)) }
}

# Objects copied between PCs can be stale or truncated; always rebuild everything.
Get-ChildItem -LiteralPath $buildDir -Recurse -File |
    Where-Object { $_.Extension -in @(".o", ".asm", ".ll", ".d") } | Remove-Item -Force
Remove-Item -Path (Join-Path $buildDir "*.hex"), (Join-Path $buildDir "*.elf"), (Join-Path $buildDir "*.map"), (Join-Path $buildDir "*.flash.bin") -Force -ErrorAction SilentlyContinue

$env:Path = "C:\LAPIS\LEXIDE\Bin;C:\LAPIS\LEXIDE\BuildTools\Ver.20260317\Bin;C:\LAPIS\LEXIDE\Utilities\Bin;" + $env:Path
Push-Location $buildDir
try {
    & $makeExe all -j
    if ($LASTEXITCODE -ne 0) { throw "Firmware build failed with exit code $LASTEXITCODE." }
} finally { Pop-Location }

$hex = Get-ChildItem -LiteralPath $buildDir -Filter "*.hex" | Select-Object -First 1
if (-not $hex) { throw "No HEX produced in $buildDir" }
Write-Host "Build succeeded: $($hex.FullName)"
Write-Host "Flash with: powershell -ExecutionPolicy Bypass -File firmware\LcdSwitchDemo\tools\flash.ps1 -FirmwareHex `"$($hex.FullName)`" -Execute"
