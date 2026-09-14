$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $projectDir '..\..')).Path
$kicad = 'C:\Program Files\KiCad\10.0\bin\kicad-cli.exe'
$board = Join-Path $projectDir 'stamp_s3a_interposer_pcba.kicad_pcb'
$schematic = Join-Path $projectDir 'stamp_s3a_interposer_pcba.kicad_sch'
$sourceBom = Join-Path $projectDir 'BOM_PCBA.csv'
$releaseRoot = Join-Path $repoRoot 'output\jlcpcb\stamp_s3a_interposer_pcba_rev_b'
$gerberDir = Join-Path $releaseRoot 'gerber_drill'
$reportDir = Join-Path $releaseRoot 'reports'

if (-not (Test-Path -LiteralPath $kicad)) {
    throw "KiCad 10 CLI not found: $kicad"
}

# This directory is generated output. Refuse to remove anything outside the
# repository's dedicated output/jlcpcb tree.
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

$ercReport = Join-Path $reportDir 'erc_pcba.rpt'
$drcReport = Join-Path $reportDir 'drc_pcba.rpt'
& $kicad sch erc --severity-all -o $ercReport $schematic
if ($LASTEXITCODE -ne 0 -or (Get-Content -Raw -LiteralPath $ercReport) -notmatch 'Errors 0') {
    throw "Schematic ERC failed: $ercReport"
}
& $kicad pcb drc --schematic-parity --severity-all -o $drcReport $board
if ($LASTEXITCODE -ne 0) {
    throw "PCB DRC command failed: $drcReport"
}
if ((Get-Content -Raw -LiteralPath $drcReport) -notmatch 'Found 0 Footprint errors') {
    throw "PCB/schematic parity failed: $drcReport"
}
$drcText = Get-Content -Raw -LiteralPath $drcReport
if ($drcText -notmatch 'Found 0 unconnected pads') {
    throw "PCB contains unconnected pads: $drcReport"
}
$drcKinds = @([regex]::Matches($drcText, '(?m)^\[([^\]]+)\]') |
    ForEach-Object { $_.Groups[1].Value })
$nonSilkKinds = @($drcKinds | Where-Object { $_ -notlike 'silk_*' } | Sort-Object -Unique)
if ($nonSilkKinds.Count -gt 0) {
    throw "Non-silkscreen DRC violations found: $($nonSilkKinds -join ', ')"
}

# JLCPCB expects copper, mask, silk, paste and Edge.Cuts in RS-274X plus
# Excellon drill data. Protel extensions are emitted by KiCad automatically.
$layers = 'F.Cu,B.Cu,F.Paste,B.Paste,F.Mask,B.Mask,F.SilkS,B.SilkS,Edge.Cuts'
& $kicad pcb export gerbers --layers $layers --check-zones --subtract-soldermask -o $gerberDir $board
if ($LASTEXITCODE -ne 0) { throw 'Gerber export failed' }
& $kicad pcb export drill --format excellon --excellon-units mm --excellon-separate-th `
    --generate-report --report-path (Join-Path $reportDir 'drill_pcba.rpt') -o $gerberDir $board
if ($LASTEXITCODE -ne 0) { throw 'Drill export failed' }

$rawPos = Join-Path $releaseRoot '_positions_kicad.csv'
& $kicad pcb export pos --format csv --units mm --side both --exclude-dnp -o $rawPos $board
if ($LASTEXITCODE -ne 0) { throw 'Position export failed' }

# Keep only parts that JLCPCB is intended to fit. Off-board, user-fitted and
# explicitly DNP lines remain in BOM_PCBA.csv for procurement documentation.
$assemblyKinds = @('SMT', 'Wave solder')
$bomRows = Import-Csv -LiteralPath $sourceBom | Where-Object { $_.Assembly -in $assemblyKinds }
$uploadBom = Join-Path $releaseRoot 'BOM_JLCPCB.csv'
$bomRows | Select-Object Comment, Designator, Footprint,
    @{Name = 'LCSC Part #'; Expression = { $_.'LCSC Part #' }} |
    Export-Csv -LiteralPath $uploadBom -NoTypeInformation -Encoding utf8

$wantedRefs = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($row in $bomRows) {
    foreach ($ref in ($row.Designator -split '[,\s]+' | Where-Object { $_ })) {
        [void]$wantedRefs.Add($ref)
    }
}

$positions = Import-Csv -LiteralPath $rawPos
$positionByRef = @{}
foreach ($position in $positions) { $positionByRef[$position.Ref] = $position }
$missing = @($wantedRefs | Where-Object { -not $positionByRef.ContainsKey($_) })
if ($missing.Count -gt 0) {
    throw "BOM references missing from CPL: $($missing -join ', ')"
}

$cplRows = foreach ($ref in ($wantedRefs | Sort-Object)) {
    $p = $positionByRef[$ref]
    [pscustomobject]@{
        Designator = $ref
        'Mid X' = ('{0:F4}mm' -f [double]$p.PosX)
        'Mid Y' = ('{0:F4}mm' -f [double]$p.PosY)
        Rotation = ('{0:F2}' -f [double]$p.Rot)
        Layer = if ($p.Side -eq 'top') { 'Top' } else { 'Bottom' }
    }
}
$uploadCpl = Join-Path $releaseRoot 'CPL_JLCPCB.csv'
$cplRows | Export-Csv -LiteralPath $uploadCpl -NoTypeInformation -Encoding utf8
Remove-Item -LiteralPath $rawPos

$gerberZip = Join-Path $releaseRoot 'GERBER_JLCPCB.zip'
Compress-Archive -Path (Join-Path $gerberDir '*') -DestinationPath $gerberZip -CompressionLevel Optimal

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    source_board = 'hardware/stamp_s3a_interposer/stamp_s3a_interposer_pcba.kicad_pcb'
    source_bom = 'hardware/stamp_s3a_interposer/BOM_PCBA.csv'
    board_stackup = '2-layer FR-4, 1.6 mm, 1 oz copper'
    board_size_mm = '110 x 72'
    assembled_bom_rows = $bomRows.Count
    assembled_designators = $wantedRefs.Count
    cpl_rows = @($cplRows).Count
    drc_warning_kinds = @($drcKinds | Group-Object | ForEach-Object {
        [ordered]@{ kind = $_.Name; count = $_.Count }
    })
    gerber_zip_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $gerberZip).Hash
    bom_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $uploadBom).Hash
    cpl_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $uploadCpl).Hash
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $releaseRoot 'MANIFEST.json') -Encoding utf8

Write-Host "JLCPCB package generated: $releaseRoot"
Write-Host "Gerber: $gerberZip"
Write-Host "BOM:    $uploadBom"
Write-Host "CPL:    $uploadCpl"
