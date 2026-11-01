[CmdletBinding()]
param(
    # LEXIDE project (ROHM AIVibrationInference based) with a generated Debug makefile.
    # Only its vendor drivers/library are used; its application sources are replaced.
    [string]$SourceProject = "$env:USERPROFILE\lexide\workspace_omega_v2\AcrylicPanCollector_lowlatency",
    # ichi_infer_main: the inference firmware (original IchiPing behaviour + on-site calibration).
    # ichi_survey_main: servos set each of the 32 states, Stamp measures live (firmware/StampMeasure),
    # on-chip feature + inference, results to TFT / UART (tools/survey_monitor.py).
    # ichi_collect_main: PC-driven servo states for data collection (tools/collect_session.py).
    [ValidateSet("ichi_infer_main", "ichi_survey_main", "ichi_collect_main")]
    [string]$Main = "ichi_infer_main",
    # CMSIS Core include (ARM.CMSIS 5.9.0 pack) of this PC.
    [string]$CmsisInclude = "$env:LOCALAPPDATA\Arm\Packs\ARM\CMSIS\5.9.0\CMSIS\Core\Include",
    [string]$Configuration = "Debug",
    [string]$StagingRoot,
    [string]$ModelGeneratedDir,
    [switch]$SurveyCalibrate,
    [switch]$SurveyExportOdl,
    [ValidateRange(1,20)][int]$SurveyRounds = 1,
    [string]$FeatureSource
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
if ($Main -eq "ichi_collect_main") { $sources = @("ichi_protocol", "ichi_stamp_link", "ichi_tft", "ichi_ui", "ichi_servo") }
else { $sources = @("ichi_protocol", "ichi_inference", "ichi_feature", "ichi_stamp_link", "ichi_tft", "ichi_ui", "ichi_servo") }
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
if ($ModelGeneratedDir) { Copy-Item -Path (Join-Path $ModelGeneratedDir '*.h') -Destination $ichiDir -Force }
function HeaderHash([string]$path) {
    # A checkout's LF/CRLF conversion must not invalidate the same ODL model.
    $canonical=[IO.File]::ReadAllText($path).Replace("`r`n","`n")
    # Existing deployed snapshots use CRLF feature tables, LF model/prior headers.
    # Preserve that identity while accepting either checkout line ending.
    if ((Split-Path -Leaf $path) -eq 'ichi_feature_tables.h') { $canonical=$canonical.Replace("`n","`r`n") }
    $bytes=[Text.Encoding]::UTF8.GetBytes($canonical)
    $sha=[Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','') } finally { $sha.Dispose() }
}
$identity = ((HeaderHash (Join-Path $ichiDir 'ichiping_model.h')) + (HeaderHash (Join-Path $ichiDir 'ichi_feature_tables.h')) + (HeaderHash (Join-Path $ichiDir 'ichi_calib_prior.h')))
[IO.File]::WriteAllText((Join-Path $ichiDir 'ichi_model_identity.h'), "#define ICHI_MODEL_IDENTITY `"$identity`"`n", [Text.UTF8Encoding]::new($false))
foreach ($s in $sources) { Copy-Item -LiteralPath (Join-Path $fwRoot "src\$s.c") -Destination $ichiDir -Force }
if ($FeatureSource) { Copy-Item -LiteralPath $FeatureSource -Destination (Join-Path $ichiDir 'ichi_feature.c') -Force }
if ($FeatureSource -and $SurveyExportOdl) {
    $featurePath=Join-Path $ichiDir 'ichi_feature.c'
    $featureText=[IO.File]::ReadAllText($featurePath)
    if ($featureText -notmatch 'bool IchiFeatureBaselineRead\(') {
        $featureText += "`nbool IchiFeatureBaselineRead(uint16_t offset, uint8_t *data, uint16_t count) { uint16_t i; if (offset > sizeof(baseline_q) || count > sizeof(baseline_q)-offset) return false; for(i=0;i<count;i++) data[i]=((const uint8_t *)(const void *)baseline_q)[offset+i]; return true; }`n"
        [IO.File]::WriteAllText($featurePath,$featureText,[Text.UTF8Encoding]::new($false))
    }
}
Copy-Item -LiteralPath (Join-Path $fwRoot "src\$Main.c") -Destination (Join-Path $staged "S_System\main.c") -Force
if ($SurveyExportOdl) {
    if ($Main -ne 'ichi_survey_main' -or -not $SurveyCalibrate) { throw 'SurveyExportOdl requires survey main and SurveyCalibrate' }
    $exportMain=Join-Path $staged 'S_System/main.c'
    [IO.File]::WriteAllText($exportMain,"#define SURVEY_EXPORT_ODL 1`n"+[IO.File]::ReadAllText($exportMain),[Text.UTF8Encoding]::new($false))
}
if ($SurveyCalibrate) {
    if ($Main -ne 'ichi_survey_main') { throw 'SurveyCalibrate requires ichi_survey_main' }
    $mainPath = Join-Path $staged 'S_System\main.c'
    [IO.File]::WriteAllText($mainPath, "#define SURVEY_CALIBRATE 1`n" + [IO.File]::ReadAllText($mainPath), [Text.UTF8Encoding]::new($false))
}
if ($Main -eq 'ichi_survey_main') {
    $mainPath = Join-Path $staged 'S_System\main.c'
    [IO.File]::WriteAllText($mainPath, "#define SURVEY_ROUNDS $SurveyRounds`n" + [IO.File]::ReadAllText($mainPath), [Text.UTF8Encoding]::new($false))
}
# The vendor project leaves UARTF1 at 9600 bps.  Collection relays 64 KiB PCM
# through this UART, so use the supported 115200-bps divisor for every IchiPing build.
$uart1Path = Join-Path $staged "S_Uartf\Uart1.c"
$uart1Text = [IO.File]::ReadAllText($uart1Path)
$uartPattern = '(?m)^\s*#define\s+UARTF1_PARAM_DLR\s*\([^\r\n]*\)'
if ([regex]::Matches($uart1Text, $uartPattern).Count -ne 1) { throw "Expected exactly one active UART divisor definition." }
$uart1Text = [regex]::Replace($uart1Text, $uartPattern, '#define UARTF1_PARAM_DLR ( 0x0019U )')
if ($uart1Text -notmatch '(?m)^#define UARTF1_PARAM_DLR \( 0x0019U \)\s*$') { throw "Could not set UARTF1 to 115200 bps." }
[IO.File]::WriteAllText($uart1Path, $uart1Text, [Text.UTF8Encoding]::new($false))
if ($Main -eq 'ichi_collect_main') {
    # 48 MHz / (16 * 6) = 500000, integer divisor, no fractional adjustment.
    $uart1Text = $uart1Text.Replace('#define UARTF1_PARAM_DLR ( 0x0019U )', '#define UARTF1_PARAM_DLR ( 0x0006U )')
    $uart1Text = [regex]::Replace($uart1Text, '(?m)^\s*#define\s+UARTF1_PARAM_CAJ[^\r\n]*', '#define UARTF1_PARAM_CAJ ( UARTF_RMV_DIS )')
    [IO.File]::WriteAllText($uart1Path, $uart1Text, [Text.UTF8Encoding]::new($false))
}
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
$buildToolsVer = (Get-ChildItem "C:\LAPIS\LEXIDE\BuildTools" -Directory -Filter "Ver.*" | Sort-Object Name -Descending | Select-Object -First 1).Name
function Repoint([string]$text) {
    $text = [regex]::Replace($text, $projRe, { param($m) if ($m.Value.Contains('\')) { $staged } else { $stagedFwd } })
    $text = [regex]::Replace($text, $cmsisRe, { param($m) $cmsisFwd })
    # the vendor project names the LEXIDE build tools version it was generated with; use the one installed here
    return [regex]::Replace($text, 'BuildTools([\\/])Ver\.\d+', { param($m) "BuildTools$($m.Groups[1].Value)$buildToolsVer" })
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
$buildToolsBin = Join-Path (Get-ChildItem "C:\LAPIS\LEXIDE\BuildTools" -Directory -Filter "Ver.*" | Sort-Object Name -Descending | Select-Object -First 1).FullName "Bin"   # newest installed LEXIDE build tools
$env:Path = "C:\LAPIS\LEXIDE\Bin;$buildToolsBin;C:\LAPIS\LEXIDE\Utilities\Bin;" + $env:Path
Push-Location $buildDir
try {
    & $makeExe all -j
    if ($LASTEXITCODE -ne 0) { throw "Firmware build failed with exit code $LASTEXITCODE." }
} finally { Pop-Location }

$elf = Get-ChildItem -LiteralPath $buildDir -Filter "*.elf" | Select-Object -First 1
if (-not $elf) { throw "No ELF produced in $buildDir" }
Write-Host "Build succeeded: $($elf.FullName)"
Write-Host "Flash with: powershell -ExecutionPolicy Bypass -File $PSScriptRoot\flash.ps1 -Firmware `"$($elf.FullName)`" -Execute"
