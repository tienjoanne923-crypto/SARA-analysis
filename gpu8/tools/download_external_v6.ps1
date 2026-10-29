# V6 external resources: gene function / pathway / expression features, the ClinVar
# citation table, and the historical (2023Q4) versions needed for the temporal audit.
# ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI); the root is
# derived from $PSScriptRoot (tools/ -> gpu8/ -> external/v6).
param(
    [string]$Root = "",
    [switch]$Overwrite
)
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

if (-not $Root -or $Root -eq "") {
    $Root = Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'external') 'v6'
}

$items = @(
    @{ group = 'gtex';      name = 'GTEx_v8_gene_median_tpm.gct.gz';  urls = @('https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz') },
    @{ group = 'reactome';  name = 'NCBI2Reactome_current.txt';       urls = @('https://reactome.org/download/current/NCBI2Reactome.txt') },
    @{ group = 'reactome';  name = 'NCBI2Reactome_v86.txt';           urls = @('https://download.reactome.org/86/NCBI2Reactome.txt') },
    @{ group = 'go';        name = 'go-basic_current.obo';            urls = @('https://purl.obolibrary.org/obo/go/go-basic.obo') },
    @{ group = 'go';        name = 'goa_human_current.gaf.gz';        urls = @('http://current.geneontology.org/annotations/goa_human.gaf.gz', 'https://current.geneontology.org/annotations/goa_human.gaf.gz') },
    @{ group = 'go';        name = 'go-basic_2023-10-09.obo';         urls = @('http://release.geneontology.org/2023-10-09/ontology/go-basic.obo') },
    @{ group = 'go';        name = 'goa_human_2023-10-09.gaf.gz';     urls = @('http://release.geneontology.org/2023-10-09/annotations/goa_human.gaf.gz') },
    @{ group = 'hpo';       name = 'genes_to_phenotype_v2023-10-09.txt'; urls = @('https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2023-10-09/genes_to_phenotype.txt') },
    @{ group = 'hgnc';      name = 'hgnc_complete_set_2023-10-01.txt'; urls = @('https://storage.googleapis.com/public-download-files/hgnc/archive/archive/quarterly/tsv/hgnc_complete_set_2023-10-01.txt') },
    @{ group = 'clinvar';   name = 'var_citations.txt';               urls = @('https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/var_citations.txt') },
    @{ group = 'wayback';   name = 'cdx_CTD_genes_diseases_2023.json'; urls = @('http://web.archive.org/cdx/search/cdx?url=ctdbase.org/reports/CTD_genes_diseases.tsv.gz&output=json&from=2023&to=2024') }
)

New-Item -ItemType Directory -Force $Root | Out-Null
$log = Join-Path $Root 'download_log_v6.csv'
if (-not (Test-Path $log)) {
    'timestamp_utc,group,name,url,status,bytes,sha256,seconds,curl_exit' | Out-File -Encoding utf8 $log
}
Write-Output "ROOT=$Root"

foreach ($it in $items) {
    $dir = Join-Path $Root $it.group
    New-Item -ItemType Directory -Force $dir | Out-Null
    $dest = Join-Path $dir $it.name
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 0) -and (-not $Overwrite)) {
        Write-Output "SKIP $($it.name)"
        continue
    }
    $ok = $false
    foreach ($url in $it.urls) {
        $part = "$dest.part"
        $t0 = Get-Date
        # -C - resumes a stalled partial file; the speed limit aborts a dead stream
        & curl.exe -L --fail --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 --speed-limit 2000 --speed-time 60 -C - -sS -A 'gpu8-pipeline/1.0' -o $part $url
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            & curl.exe -L --fail --retry 8 --retry-delay 5 --retry-all-errors --connect-timeout 30 --speed-limit 2000 --speed-time 60 -C - -sS -A 'gpu8-pipeline/1.0' -o $part $url
            $code = $LASTEXITCODE
        }
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
        }
    }
    if (-not $ok) { Write-Output "GIVEUP $($it.name)" }
}
Write-Output "DONE download_external_v6"
