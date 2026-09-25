[CmdletBinding()]
param(
    # .elf from tools\build.ps1, or the flash-only .bin in prebuilt\ (no build environment needed)
    [Parameter(Mandatory = $true)]
    [string]$Firmware,
    [string]$OpenOcdExe = "C:\LAPIS\LEXIDE\gdb\openocd_arm.exe",
    [string]$ObjcopyExe = "C:\LAPIS\LEXIDE\BuildTools\Ver.20260317\Bin\llvm-objcopy.exe",
    [string]$InterfaceCfg = "C:\LAPIS\LEXIDE\Cfg\cmsis-dap.cfg",
    [string]$TargetCfg = "$env:LOCALAPPDATA\Arm\Packs\ROHM\ML63Q25x7_DFP\1.1.0\Cfg\ml63q25x7.cfg",
    [switch]$Execute
)

# Program the ML63Q2557 through MCU-Link (CMSIS-DAP) with the OpenOCD bundled in LEXIDE:
# erase -> write -> verify -> reset run.  Without -Execute only the inputs are checked.

$ErrorActionPreference = "Stop"
foreach ($path in @($Firmware, $OpenOcdExe, $InterfaceCfg, $TargetCfg)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file not found: $path" }
}
$firmwareFull = (Resolve-Path -LiteralPath $Firmware).Path

if ([IO.Path]::GetExtension($firmwareFull) -ieq ".bin") {
    $flashBinary = $firmwareFull
} else {
    if (-not (Test-Path -LiteralPath $ObjcopyExe -PathType Leaf)) { throw "llvm-objcopy not found: $ObjcopyExe" }
    # The HEX also carries .heap/.stack records at RAM addresses, so program a
    # flash-only image built from the ELF load sections.
    $flashBinary = [IO.Path]::ChangeExtension($firmwareFull, ".flash.bin")
    & $ObjcopyExe -O binary `
        --only-section=.text --only-section=.ARM.exidx --only-section=.copy.table `
        --only-section=.zero.table --only-section=.data `
        $firmwareFull $flashBinary
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $flashBinary)) { throw "Failed to create the flash-only binary." }
}

Write-Host "Probe : MCU-Link CMSIS-DAP"
Write-Host "Target: ML63Q25x7"
Write-Host "Image : $flashBinary"
if (-not $Execute) {
    Write-Host "Dry run only. Run again with -Execute to erase, program, verify, reset and start."
    return
}
$commands = "init; reset halt; flash write_image erase {$flashBinary} 0x10000000 bin; verify_image {$flashBinary} 0x10000000 bin; reset run; shutdown"
& $OpenOcdExe -f $InterfaceCfg -f $TargetCfg -c $commands
if ($LASTEXITCODE -ne 0) { throw "OpenOCD flashing failed with exit code $LASTEXITCODE." }
Write-Host "Firmware flashing and verification completed."
