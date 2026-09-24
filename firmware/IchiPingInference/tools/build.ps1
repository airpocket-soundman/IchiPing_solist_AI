[CmdletBinding()]
param(
    # Private LEXIDE project that already has a generated Debug makefile with the
    # S_AcrylicPan overlay (see D:\GitHub\acrylic_pan\scripts\build-firmware.ps1).
    [string]$SourceProject = "C:\Users\yamas\lexide\workspace_omega_v2\AcrylicPanCollector_lowlatency",
    [string]$AcrylicOverlay = "D:\GitHub\acrylic_pan\firmware\AcrylicPanCollector",
    [string]$Configuration = "Debug",
    [string]$StagingRoot
)

$ErrorActionPreference = "Stop"
$overlayRoot = Split-Path -Parent $PSScriptRoot          # firmware/IchiPingInference
$repoRoot = Split-Path -Parent (Split-Path -Parent $overlayRoot)
$makeExe = "C:\LAPIS\LEXIDE\Utilities\Bin\make.exe"

foreach ($p in @($SourceProject, $AcrylicOverlay)) {
    if (-not (Test-Path -LiteralPath $p -PathType Container)) { throw "Not found: $p" }
}
if (-not (Test-Path -LiteralPath $makeExe -PathType Leaf)) { throw "LEXIDE make.exe not found: $makeExe" }
if (-not $StagingRoot) {
    $StagingRoot = Join-Path $repoRoot (".local\firmware-build\" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$stagingFull = [System.IO.Path]::GetFullPath($StagingRoot)
if (Test-Path -LiteralPath $stagingFull) { throw "Staging path already exists: $stagingFull" }

$projectName = Split-Path -Leaf $SourceProject
$staged = Join-Path $stagingFull $projectName
New-Item -ItemType Directory -Path $staged -Force | Out-Null
Write-Host "Staging a disposable copy of $SourceProject"
Get-ChildItem -LiteralPath $SourceProject -Force | Copy-Item -Destination $staged -Recurse -Force

# 1) acrylic_pan overlay (protocol, capture, KX134 inference, app, vendor main replacement)
$ovl = Join-Path $staged "S_AcrylicPan"
$dbgOvl = Join-Path $staged "$Configuration\S_AcrylicPan"
foreach ($d in @($ovl, $dbgOvl)) { if (-not (Test-Path -LiteralPath $d)) { throw "Missing overlay dir: $d" } }
Copy-Item -Path (Join-Path $AcrylicOverlay "include\*.h") -Destination $ovl -Force
Copy-Item -Path (Join-Path $AcrylicOverlay "generated\*.h") -Destination $ovl -Force
Copy-Item -Path (Join-Path $AcrylicOverlay "src\*.c") -Destination $ovl -Force
Copy-Item -LiteralPath (Join-Path $AcrylicOverlay "integration\apan_collector_app.c") -Destination $ovl -Force
Copy-Item -LiteralPath (Join-Path $AcrylicOverlay "integration\main_collector.c") -Destination (Join-Path $staged "S_System\main.c") -Force
Copy-Item -LiteralPath (Join-Path $AcrylicOverlay "tools\S_AcrylicPan.subdir.mk") -Destination (Join-Path $dbgOvl "subdir.mk") -Force
$posRes = Join-Path $dbgOvl "apan_position_inference.res"
if (-not (Test-Path -LiteralPath $posRes)) {
    $t = [IO.File]::ReadAllText((Join-Path $dbgOvl "apan_inference.res"))
    $t = $t.Replace('apan_inference.asm', 'apan_position_inference.asm').Replace('apan_inference.c', 'apan_position_inference.c')
    [IO.File]::WriteAllText($posRes, $t, [Text.UTF8Encoding]::new($false))
}

# 2) IchiPing overrides: 167x32x14 model, AI_INFER command, larger command frames
Copy-Item -Path (Join-Path $overlayRoot "include\*.h") -Destination $ovl -Force
Copy-Item -Path (Join-Path $overlayRoot "generated\ichiping_model.h") -Destination $ovl -Force
Copy-Item -Path (Join-Path $overlayRoot "src\*.c") -Destination $ovl -Force
Copy-Item -LiteralPath (Join-Path $overlayRoot "integration\apan_collector_app.c") -Destination $ovl -Force

# The .res response files carry absolute include paths of the source project;
# the overlay sources are identical in name, so they resolve to the staged copy
# only for relative paths.  Point them at the staged project instead.
Get-ChildItem -LiteralPath (Join-Path $staged $Configuration) -Filter "*.res" -Recurse | ForEach-Object {
    $t = [IO.File]::ReadAllText($_.FullName)
    $t2 = $t.Replace($SourceProject.Replace('\', '/'), $staged.Replace('\', '/')).Replace($SourceProject, $staged)
    if ($t2 -ne $t) { [IO.File]::WriteAllText($_.FullName, $t2, [Text.UTF8Encoding]::new($false)) }
}

$buildDir = Join-Path $staged $Configuration
if (-not (Test-Path -LiteralPath (Join-Path $buildDir "makefile"))) { throw "Generated makefile not found in $buildDir" }
# Force recompilation of the overlay objects so stale .o files are never linked.
Get-ChildItem -Path (Join-Path $dbgOvl "*") -Include "*.o", "*.asm", "*.ll" -File | Remove-Item -Force
Remove-Item -Path (Join-Path $buildDir "*.hex"), (Join-Path $buildDir "*.elf"), (Join-Path $buildDir "*.map") -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $buildDir "S_System\main.o"), (Join-Path $buildDir "S_System\main.asm") -Force -ErrorAction SilentlyContinue

$env:Path = "C:\LAPIS\LEXIDE\Bin;C:\LAPIS\LEXIDE\BuildTools\Ver.20260317\Bin;C:\LAPIS\LEXIDE\Utilities\Bin;" + $env:Path
Push-Location $buildDir
try {
    & $makeExe all -j
    if ($LASTEXITCODE -ne 0) { throw "Firmware build failed with exit code $LASTEXITCODE." }
} finally { Pop-Location }

$hex = Get-ChildItem -LiteralPath $buildDir -Filter "*.hex" | Select-Object -First 1
if (-not $hex) { throw "No HEX produced in $buildDir" }
Write-Host "Build succeeded: $($hex.FullName)"
Write-Host "Flash with: powershell -File D:\GitHub\acrylic_pan\scripts\flash-firmware.ps1 -FirmwareHex `"$($hex.FullName)`" -Execute"
