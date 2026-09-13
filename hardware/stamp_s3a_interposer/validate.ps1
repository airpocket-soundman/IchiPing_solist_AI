$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$kicad = 'C:\Program Files\KiCad\10.0\bin\kicad-cli.exe'
if (-not (Test-Path -LiteralPath $kicad)) {
    throw "KiCad 10 CLI not found: $kicad"
}

$buildDir = Join-Path $projectDir 'build'
$reportDir = Join-Path $buildDir 'reports'
$renderDir = Join-Path $buildDir 'render'
New-Item -ItemType Directory -Force $reportDir, $renderDir | Out-Null

$kicadPython = 'C:\Program Files\KiCad\10.0\bin\python.exe'

& $kicadPython (Join-Path $projectDir 'tools\audit_pinmap.py')
foreach ($variant in @('pcba', 'tht')) {
    $schematic = Join-Path $projectDir "stamp_s3a_interposer_$variant.kicad_sch"
    $board = Join-Path $projectDir "stamp_s3a_interposer_$variant.kicad_pcb"
    $ercReport = Join-Path $reportDir "erc_$variant.rpt"
    $drcReport = Join-Path $reportDir "drc_$variant.rpt"
    $gerberDir = Join-Path $buildDir "gerbers_$variant"
    $drillDir = Join-Path $buildDir "drill_$variant"
    New-Item -ItemType Directory -Force $gerberDir, $drillDir | Out-Null

    & $kicad sch erc --severity-all -o $ercReport $schematic
    if ((Get-Content -Raw -LiteralPath $ercReport) -notmatch 'Errors 0') {
        throw "Schematic ERC contains errors: $ercReport"
    }
    & $kicad sch export pdf -o (Join-Path $buildDir "stamp_s3a_interposer_$variant-schematic.pdf") $schematic

    & $kicad pcb drc --schematic-parity --severity-all -o $drcReport $board
    if ((Get-Content -Raw -LiteralPath $drcReport) -notmatch 'Found 0 Footprint errors') {
        throw "PCB/schematic parity failed: $drcReport"
    }
    & $kicad pcb export gerbers --layers 'F.Cu,B.Cu,F.Mask,B.Mask,F.SilkS,B.SilkS,Edge.Cuts' --check-zones -o $gerberDir $board
    & $kicad pcb export drill --format excellon --excellon-units mm --excellon-separate-th --generate-map --map-format pdf --generate-report --report-path (Join-Path $reportDir "drill_$variant.rpt") -o $drillDir $board
    & $kicad pcb export pos --format csv --units mm --side both -o (Join-Path $buildDir "positions_$variant.csv") $board
    & $kicad pcb render --output (Join-Path $renderDir "board-top_$variant.png") --side top --width 1600 --height 1000 --quality high $board
    & $kicad pcb render --output (Join-Path $renderDir "board-bottom_$variant.png") --side bottom --width 1600 --height 1000 --quality high $board
}

Write-Host "Validation outputs: $buildDir"
