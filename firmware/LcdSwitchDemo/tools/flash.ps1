[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FirmwareHex,
    [string]$OpenOcdExe = "C:\LAPIS\LEXIDE\gdb\openocd_arm.exe",
    [string]$ObjcopyExe = "C:\LAPIS\LEXIDE\BuildTools\Ver.20260317\Bin\llvm-objcopy.exe",
    [string]$InterfaceCfg = "C:\LAPIS\LEXIDE\Cfg\cmsis-dap.cfg",
    [string]$TargetCfg = "$env:LOCALAPPDATA\Arm\Packs\ROHM\ML63Q25x7_DFP\1.1.0\Cfg\ml63q25x7.cfg",
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$firmwareElf = [System.IO.Path]::ChangeExtension($FirmwareHex, ".elf")
$required = @($FirmwareHex, $firmwareElf, $OpenOcdExe, $ObjcopyExe, $InterfaceCfg, $TargetCfg)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

$hexFull = (Resolve-Path -LiteralPath $FirmwareHex).Path
$openOcdFull = (Resolve-Path -LiteralPath $OpenOcdExe).Path
$objcopyFull = (Resolve-Path -LiteralPath $ObjcopyExe).Path
$interfaceFull = (Resolve-Path -LiteralPath $InterfaceCfg).Path
$targetFull = (Resolve-Path -LiteralPath $TargetCfg).Path
$elfFull = (Resolve-Path -LiteralPath $firmwareElf).Path
$flashBinary = [System.IO.Path]::ChangeExtension($elfFull, ".flash.bin")

Write-Host "Probe : MCU-Link CMSIS-DAP"
Write-Host "Target: ML63Q25x7"
Write-Host "HEX   : $hexFull"
if (-not $Execute) {
    Write-Host "Dry run only. Run again with -Execute to erase, program, verify, reset and start."
    return
}

# The generated HEX also contains .heap/.stack records at RAM addresses. Build a
# persistent image from ELF load sections so verification covers flash only.
& $objcopyFull -O binary `
    --only-section=.text --only-section=.ARM.exidx --only-section=.copy.table `
    --only-section=.zero.table --only-section=.data `
    $elfFull $flashBinary
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $flashBinary)) {
    throw "Failed to create the flash-only binary."
}

$commands = "init; reset halt; flash write_image erase {$flashBinary} 0x10000000 bin; verify_image {$flashBinary} 0x10000000 bin; reset run; shutdown"
& $openOcdFull `
    -f $interfaceFull `
    -f $targetFull `
    -c $commands
if ($LASTEXITCODE -ne 0) {
    throw "OpenOCD flashing failed with exit code $LASTEXITCODE."
}
Write-Host "Firmware flashing and verification completed."
