# Download every external resource the GPU8 pipeline needs, with provenance logging.
# ASCII-only on purpose: Windows PowerShell 5.1 reads .ps1 as ANSI, so a literal
# non-ASCII path in this file would be mangled. The root is derived from
# $PSScriptRoot instead (tools/ -> gpu8/ -> external/).
param(
    [string]$Root = "",
    [switch]$Overwrite
)
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

if (-not $Root -or $Root -eq "") {
    $Root = Join-Path (Split-Path -Parent $PSScriptRoot) 'external'
}

$items = @(
    @{ group = 'hgnc';    name = 'hgnc_complete_set.txt';           urls = @('https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt') },
    @{ group = 'mondo';   name = 'mondo.obo';                       urls = @('https://purl.obolibrary.org/obo/mondo.obo') },
    @{ group = 'mondo';   name = 'mondo.sssom.tsv';                 urls = @('https://purl.obolibrary.org/obo/mondo/mappings/mondo.sssom.tsv', 'https://raw.githubusercontent.com/monarch-initiative/mondo/master/src/ontology/mappings/mondo.sssom.tsv') },
    @{ group = 'gencc';   name = 'gencc-submissions.tsv';           urls = @('https://thegencc.org/download/action/submissions-export-tsv?format=new', 'https://search.thegencc.org/download/action/submissions-export-tsv') },
    @{ group = 'clinvar'; name = 'gene_condition_source_id';        urls = @('https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id') },
    @{ group = 'clinvar'; name = 'variant_summary.txt.gz';          urls = @('https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz') },
    @{ group = 'string';  name = '9606.protein.links.v12.0.txt.gz'; urls = @('https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz') },
    @{ group = 'geo';     name = 'GSE13355_series_matrix.txt.gz';   urls = @('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE13nnn/GSE13355/matrix/GSE13355_series_matrix.txt.gz') },
    @{ group = 'geo';     name = 'GSE30999_series_matrix.txt.gz';   urls = @('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE30nnn/GSE30999/matrix/GSE30999_series_matrix.txt.gz') },
    @{ group = 'geo';     name = 'GSE87466_series_matrix.txt.gz';   urls = @('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87466/matrix/GSE87466_series_matrix.txt.gz') },
    @{ group = 'geo';     name = 'GSE75214_series_matrix.txt.gz';   urls = @('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE75nnn/GSE75214/matrix/GSE75214_series_matrix.txt.gz') },
    @{ group = 'geo';     name = 'GPL570.annot.gz';                 urls = @('https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz') },
    @{ group = 'geo';     name = 'GPL13158.annot.gz';               urls = @('https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13158/annot/GPL13158.annot.gz') },
    @{ group = 'geo';     name = 'GPL6244.annot.gz';                urls = @('https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6244/annot/GPL6244.annot.gz') },
    @{ group = 'kdgene';  name = 'KDGene-master.zip';               urls = @('https://github.com/2020MEAI/KDGene/archive/refs/heads/master.zip') },
    @{ group = 'qdgp';    name = 'qdgp-main.zip';                   urls = @('https://github.com/markgolds/qdgp/archive/refs/heads/main.zip') }
)

New-Item -ItemType Directory -Force $Root | Out-Null
$log = Join-Path $Root 'download_log.csv'
if (-not (Test-Path $log)) {
    'timestamp_utc,group,name,url,status,bytes,sha256,seconds,curl_exit' | Out-File -Encoding utf8 $log
}
Write-Output "ROOT=$Root"

foreach ($it in $items) {
    $dir = Join-Path $Root $it.group
    New-Item -ItemType Directory -Force $dir | Out-Null
    $dest = Join-Path $dir $it.name
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0) -and (-not $Overwrite)) {
        $sha = (Get-FileHash $dest -Algorithm SHA256).Hash
        $line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}' -f (Get-Date).ToUniversalTime().ToString('o'), $it.group, $it.name, 'existing', 'skipped_existing', (Get-Item $dest).Length, $sha, 0, 0
        Add-Content -Encoding utf8 $log $line
        Write-Output "SKIP $($it.name)"
        continue
    }
    $ok = $false
    foreach ($url in $it.urls) {
        $part = "$dest.part"
        if (Test-Path $part) { Remove-Item -Force $part }
        $t0 = Get-Date
        & curl.exe -L --fail --retry 10 --retry-delay 5 --retry-all-errors --connect-timeout 30 -sS -A 'gpu8-pipeline/1.0' -o $part $url
        $code = $LASTEXITCODE
        $secs = [int]((Get-Date) - $t0).TotalSeconds
        if ($code -eq 0 -and (Test-Path $part) -and ((Get-Item $part).Length -gt 0)) {
            Move-Item -Force $part $dest
            $sha = (Get-FileHash $dest -Algorithm SHA256).Hash
            $len = (Get-Item $dest).Length
            $line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}' -f (Get-Date).ToUniversalTime().ToString('o'), $it.group, $it.name, $url, 'ok', $len, $sha, $secs, $code
            Add-Content -Encoding utf8 $log $line
            Write-Output "OK   $($it.name) ($len bytes, $secs s)"
            $ok = $true
            break
        } else {
            $line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}' -f (Get-Date).ToUniversalTime().ToString('o'), $it.group, $it.name, $url, 'failed', 0, '', $secs, $code
            Add-Content -Encoding utf8 $log $line
            Write-Output "FAIL $($it.name) <- $url (curl exit $code)"
            if (Test-Path $part) { Remove-Item -Force $part }
        }
    }
    if (-not $ok) { Write-Output "GIVEUP $($it.name)" }
}
Write-Output "DONE download_external"
