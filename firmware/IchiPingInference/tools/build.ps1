[CmdletBinding()]
param(
    # LEXIDE project (ROHM AIVibrationInference based) with a generated Debug makefile.
    # Only its vendor drivers/library are used; its application sources are replaced.
    [string]$SourceProject = "$env:USERPROFILE\lexide\workspace_omega_v2\AcrylicPanCollector_lowlatency",
    # ichi_main: UART request/response firmware.  ichi_pipeline_main: Stamp I2C PCM ->
    # on-chip N333 feature -> inference test (firmware/StampPipelineTest).
    # ichi_tft_test_main: ILI9341 TFT bring-up (colour bars, text, counter).
    # ichi_io_test_main: switch / EXEC states from the Stamp (I2C STATUS) shown on the TFT.
    # ichi_servo_test_main: PCA9685 + SG90 one-by-one sweep started / stopped with EXEC.
    [ValidateSet("ichi_main", "ichi_pipeline_main", "ichi_tft_test_main", "ichi_io_test_main", "ichi_servo_test_main")]
    [string]$Main = "ichi_main",
    # CMSIS Core include (ARM.CMSIS 5.9.0 pack) of this PC.
    [string]$CmsisInclude = "$env:LOCALAPPDATA\Arm\Packs\ARM\CMSIS\5.9.0\CMSIS\Core\Include",
    [string]$Configuration = "Debug",
    [string]$StagingRoot
)

# Builds the IchiPing firmware in a disposable copy of the vendor project:
#   S_IchiPing/        <- include/*.h, src/<sources>.c, generated/*.h
#   S_System/main.c    <- src/<Main>.c
#   Debug/S_IchiPing/  <- subdir.mk (tools/S_IchiPing.subdir.mk with the source list) + per-source .res
# The vendor project may have been generated on another PC: its absolute project
# and CMSIS paths in the .res files and makefile are re-pointed at this PC.
# The vendor project itself is never modified.  Do not run "make clean" (it deletes the .res files).

$ErrorActionPreference = "Stop"
$fwRoot = Split-Path -Parent $PSScriptRoot                  # firmware/IchiPingInference
$repoRoot = Split-Path -Parent (Split-Path -Parent $fwRoot)
$makeExe = "C:\LAPIS\LEXIDE\Utilities\Bin\make.exe"
if ($Main -eq "ichi_pipeline_main") { $sources = @("ichi_protocol", "ichi_inference", "ichi_feature", "ichi_stamp_link") }
elseif ($Main -eq "ichi_tft_test_main") { $sources = @("ichi_tft") }
elseif ($Main -eq "ichi_io_test_main") { $sources = @("ichi_tft", "ichi_stamp_link") }
elseif ($Main -eq "ichi_servo_test_main") { $sources = @("ichi_tft", "ichi_stamp_link", "ichi_servo") }
else { $sources = @("ichi_protocol", "ichi_inference", "ichi_app") }
if (-not (Test-Path -LiteralPath $CmsisInclude -PathType Container)) { throw "CMSIS include not found: $CmsisInclude" }

if (-not (Test-Path -LiteralPath $SourceProject -PathType Container)) { throw "Vendor project not found: $SourceProject" }
if (-not (Test-Path -LiteralPath $makeExe -PathType Leaf)) { throw "LEXIDE make.exe not found: $makeExe" }
if (-not $StagingRoot) { $StagingRoot = Join-Path $repoRoot (".local\firmware-build\" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
$stagingFull = [IO.Path]::GetFullPath($StagingRoot)
if (Test-Path -LiteralPath $stagingFull) { throw "Staging path already exists: $stagingFull" }

$staged = Join-Path $stagingFull (Split-Path -Leaf $SourceProject)
$buildDir = Join-Path $staged $Configuration
New-Item -ItemType Directory -Path $staged -Force | Out-Null
Write-Host "Staging a copy of $SourceProject"
Get-ChildItem -LiteralPath $SourceProject -Force | Copy-Item -Destination $staged -Recurse -Force
if (-not (Test-Path -LiteralPath (Join-Path $buildDir "makefile"))) { throw "Generated makefile not found in $buildDir" }

function Write-Utf8([string]$path, [string]$text) { [IO.File]::WriteAllText($path, $text, [Text.UTF8Encoding]::new($false)) }
$stagedFwd = $staged.Replace('\', '/')
$srcProjFwd = $SourceProject.Replace('\', '/')

# 1) Drop the previous application overlay (acrylic_pan) and install the IchiPing sources.
foreach ($d in @((Join-Path $staged "S_AcrylicPan"), (Join-Path $buildDir "S_AcrylicPan"))) {
    if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Recurse -Force }
}
$ichiDir = Join-Path $staged "S_IchiPing"
$ichiBuild = Join-Path $buildDir "S_IchiPing"
New-Item -ItemType Directory -Path $ichiDir, $ichiBuild -Force | Out-Null
Copy-Item -Path (Join-Path $fwRoot "include\*.h") -Destination $ichiDir -Force
Copy-Item -Path (Join-Path $fwRoot "generated\*.h") -Destination $ichiDir -Force
foreach ($s in $sources) { Copy-Item -LiteralPath (Join-Path $fwRoot "src\$s.c") -Destination $ichiDir -Force }
Copy-Item -LiteralPath (Join-Path $fwRoot "src\$Main.c") -Destination (Join-Path $staged "S_System\main.c") -Force
$mkLists = @{
    "C_SRCS" = { param($s) "../S_IchiPing/$s.c" }; "RESS" = { param($s) "./S_IchiPing/$s.res" }
    "RESS__QUOTED" = { param($s) "`"./S_IchiPing/$s.res`"" }; "ASMS" = { param($s) "./S_IchiPing/$s.asm" }
    "ASMS__QUOTED" = { param($s) "`"./S_IchiPing/$s.asm`"" }; "OBJS" = { param($s) "./S_IchiPing/$s.o" }
    "OBJS__QUOTED" = { param($s) "`"./S_IchiPing/$s.o`"" }
}
$mkText = [regex]::Replace([IO.File]::ReadAllText((Join-Path $PSScriptRoot "S_IchiPing.subdir.mk")),
    '(?m)^(C_SRCS|RESS|RESS__QUOTED|ASMS|ASMS__QUOTED|OBJS|OBJS__QUOTED) \+=.*$',
    { param($m) $v = $m.Groups[1].Value; "$v += " + (($sources | ForEach-Object { & $mkLists[$v] $_ }) -join " ") })
Write-Utf8 (Join-Path $ichiBuild "subdir.mk") $mkText

# 2) Compiler response files: point every .res at the staged project, and use
#    S_IchiPing instead of S_AcrylicPan as the application include directory.
$ichiInclude = "-I`"$stagedFwd/S_IchiPing`""
$projRe = '[A-Za-z]:[\\/]Users[\\/][^\\/"]+[\\/]lexide[\\/]workspace_omega_v2[\\/]' + [regex]::Escape((Split-Path -Leaf $SourceProject))
$cmsisRe = '[A-Za-z]:/Users/[^/"]+/AppData/Local/Arm/Packs/ARM/CMSIS/[^/"]+/CMSIS/Core/Include'
$cmsisFwd = (Resolve-Path -LiteralPath $CmsisInclude).Path.Replace('\', '/')
function Repoint([string]$text) {
    $text = [regex]::Replace($text, $projRe, { param($m) if ($m.Value.Contains('\')) { $staged } else { $stagedFwd } })
    return [regex]::Replace($text, $cmsisRe, { param($m) $cmsisFwd })
}
$mkPath = Join-Path $buildDir "makefile"
Write-Utf8 $mkPath (Repoint ([IO.File]::ReadAllText($mkPath)))
Get-ChildItem -LiteralPath $buildDir -Filter "*.res" -Recurse | ForEach-Object {
    $t = Repoint ([IO.File]::ReadAllText($_.FullName).Replace($srcProjFwd, $stagedFwd).Replace($SourceProject, $staged))
    $t = $t.Replace("$stagedFwd/S_AcrylicPan`"", "$stagedFwd/S_IchiPing`"")
    if (-not $t.Contains($ichiInclude)) { $t = $t.Replace("[option_lccarm]", "[option_lccarm]$ichiInclude ") }
    Write-Utf8 $_.FullName $t
}
$template = [IO.File]::ReadAllText((Join-Path $buildDir "S_System\main.res"))
foreach ($s in $sources) {
    $t = $template.Replace('[output_dir]"./S_System/"', '[output_dir]"./S_IchiPing/"')
    $t = $t.Replace('[output_filename]"main.asm"', "[output_filename]`"$s.asm`"")
    $t = $t.Replace('[file_c]"../S_System/main.c"', "[file_c]`"../S_IchiPing/$s.c`"")
    $t = $t.Replace('-o"S_System/"', '-o"S_IchiPing/"')
    Write-Utf8 (Join-Path $ichiBuild "$s.res") $t
}

# 3) makefile: build S_IchiPing instead of S_AcrylicPan.
$mk = Join-Path $buildDir "makefile"
$m = [IO.File]::ReadAllText($mk)
if ($m.Contains("-include S_AcrylicPan/subdir.mk")) { $m = $m.Replace("-include S_AcrylicPan/subdir.mk", "-include S_IchiPing/subdir.mk") }
elseif (-not $m.Contains("-include S_IchiPing/subdir.mk")) { $m = $m.Replace("-include RTE/subdir.mk", "-include S_IchiPing/subdir.mk`r`n-include RTE/subdir.mk") }
Write-Utf8 $mk $m

# 4) Build (stale objects of replaced sources are removed first).
Remove-Item -Path (Join-Path $buildDir "*.hex"), (Join-Path $buildDir "*.elf"), (Join-Path $buildDir "*.map") -Force -ErrorAction SilentlyContinue
Remove-Item -Path (Join-Path $buildDir "S_System\main.o"), (Join-Path $buildDir "S_System\main.asm") -Force -ErrorAction SilentlyContinue
$env:Path = "C:\LAPIS\LEXIDE\Bin;C:\LAPIS\LEXIDE\BuildTools\Ver.20260317\Bin;C:\LAPIS\LEXIDE\Utilities\Bin;" + $env:Path
Push-Location $buildDir
try {
    & $makeExe all -j
    if ($LASTEXITCODE -ne 0) { throw "Firmware build failed with exit code $LASTEXITCODE." }
} finally { Pop-Location }

$elf = Get-ChildItem -LiteralPath $buildDir -Filter "*.elf" | Select-Object -First 1
if (-not $elf) { throw "No ELF produced in $buildDir" }
Write-Host "Build succeeded: $($elf.FullName)"
Write-Host "Flash with: powershell -ExecutionPolicy Bypass -File $PSScriptRoot\flash.ps1 -Firmware `"$($elf.FullName)`" -Execute"
