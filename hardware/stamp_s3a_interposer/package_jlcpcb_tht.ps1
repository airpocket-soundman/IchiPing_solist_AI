$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $projectDir '..\..')).Path
$kicad = 'C:\Program Files\KiCad\10.0\bin\kicad-cli.exe'
$kicadPython = 'C:\Program Files\KiCad\10.0\bin\python.exe'
$board = Join-Path $projectDir 'stamp_s3a_interposer_tht.kicad_pcb'
$schematic = Join-Path $projectDir 'stamp_s3a_interposer_tht.kicad_sch'
$sourceBom = Join-Path $projectDir 'BOM_THT.csv'
$releaseRoot = Join-Path $repoRoot 'output\jlcpcb\stamp_s3a_interposer_tht_rev_b'
$gerberDir = Join-Path $releaseRoot 'gerber_drill'
$reportDir = Join-Path $releaseRoot 'reports'

foreach ($required in @($kicad, $kicadPython, $board, $schematic, $sourceBom)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file not found: $required" }
}

# Generated output is replaced only inside the repository's dedicated JLCPCB tree.
$jlcOutputRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot 'output\jlcpcb'))
$resolvedRelease = [IO.Path]::GetFullPath($releaseRoot)
if (-not $resolvedRelease.StartsWith($jlcOutputRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe release path: $resolvedRelease"
}
if (Test-Path -LiteralPath $releaseRoot) {
    Remove-Item -LiteralPath $releaseRoot -Recurse -Force
}
New-Item -ItemType Directory -Force $gerberDir, $reportDir | Out-Null

& $kicadPython (Join-Path $projectDir 'tools\audit_pinmap.py')
if ($LASTEXITCODE -ne 0) { throw 'Pin-map audit failed' }

$ercReport = Join-Path $reportDir 'erc_tht.rpt'
$drcReport = Join-Path $reportDir 'drc_tht.rpt'
& $kicad sch erc --severity-all -o $ercReport $schematic
if ($LASTEXITCODE -ne 0 -or (Get-Content -Raw -LiteralPath $ercReport) -notmatch 'Errors 0') {
    throw "Schematic ERC failed: $ercReport"
}
& $kicad pcb drc --schematic-parity --severity-all -o $drcReport $board
if ($LASTEXITCODE -ne 0) { throw "PCB DRC command failed: $drcReport" }
$drcText = Get-Content -Raw -LiteralPath $drcReport
if ($drcText -notmatch 'Found 0 DRC violations' -or
    $drcText -notmatch 'Found 0 unconnected pads' -or
    $drcText -notmatch 'Found 0 Footprint errors') {
    throw "THT board is not release-clean: $drcReport"
}

# A bare through-hole PCB needs copper, solder mask, silkscreen, outline and
# separate plated/non-plated drill files. Paste layers and CPL are not needed.
$layers = 'F.Cu,B.Cu,F.Mask,B.Mask,F.SilkS,B.SilkS,Edge.Cuts'
& $kicad pcb export gerbers --layers $layers --check-zones --subtract-soldermask -o $gerberDir $board
if ($LASTEXITCODE -ne 0) { throw 'Gerber export failed' }
& $kicad pcb export drill --format excellon --excellon-units mm --excellon-separate-th `
    --generate-report --report-path (Join-Path $reportDir 'drill_tht.rpt') -o $gerberDir $board
if ($LASTEXITCODE -ne 0) { throw 'Drill export failed' }

Copy-Item -LiteralPath $sourceBom -Destination (Join-Path $releaseRoot 'BOM_THT.csv')
$gerberZip = Join-Path $releaseRoot 'GERBER_JLCPCB.zip'
Compress-Archive -Path (Join-Path $gerberDir '*') -DestinationPath $gerberZip -CompressionLevel Optimal

$orderNotes = @'
# JLCPCB order settings (THT bare PCB)

- Layers: 2
- Dimensions: 92 x 69 mm
- Base material: FR-4
- PCB thickness: 1.6 mm
- Copper weight: 1 oz
- Surface finish: HASL with lead (lowest-cost option)
- Outer copper color: green solder mask
- Silkscreen: white
- Remove order number: optional
- Assembly: none; populate manually using BOM_THT.csv

Upload only GERBER_JLCPCB.zip to the PCB quotation page. Do not upload
BOM_THT.csv as a JLCPCB assembly BOM.
'@
$orderNotes | Set-Content -LiteralPath (Join-Path $releaseRoot 'ORDER_NOTES.md') -Encoding utf8

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    source_board = 'hardware/stamp_s3a_interposer/stamp_s3a_interposer_tht.kicad_pcb'
    source_bom = 'hardware/stamp_s3a_interposer/BOM_THT.csv'
    board_stackup = '2-layer FR-4, 1.6 mm, 1 oz copper'
    board_size_mm = '92 x 69'
    assembly = 'none (manual through-hole assembly)'
    drc_violations = 0
    unconnected_pads = 0
    schematic_parity_errors = 0
    gerber_zip_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $gerberZip).Hash
    bom_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $releaseRoot 'BOM_THT.csv')).Hash
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $releaseRoot 'MANIFEST.json') -Encoding utf8

Write-Host "JLCPCB THT package generated: $releaseRoot"
Write-Host "Upload: $gerberZip"
Write-Host "BOM:    $(Join-Path $releaseRoot 'BOM_THT.csv')"
