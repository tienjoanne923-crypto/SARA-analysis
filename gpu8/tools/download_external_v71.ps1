# V7.1 external resources: HPA consensus tissue expression, representative protein
# sequences (UniProt reference proteome + MANE), IMPC phenotype calls, MP ontology,
# MGI human-mouse homology, the MGI curated MP-HP SSSOM mapping, and the HPA image
# features / subcellular locations for the conditional image channel.
# ASCII-only on purpose (Windows PowerShell 5.1 reads .ps1 as ANSI); the root is
# derived from $PSScriptRoot (tools/ -> gpu8/ -> external/v71).
param(
    [string]$Root = "",
    [switch]$Overwrite,
    [switch]$Large
)
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'

if (-not $Root -or $Root -eq "") {
    $Root = Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'external') 'v71'
}

$items = @(
    @{ group = 'hpa';     name = 'rna_tissue_consensus.tsv.zip';          urls = @('https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip') },
    @{ group = 'hpa';     name = 'subcellular_location.tsv.zip';          urls = @('https://www.proteinatlas.org/download/tsv/subcellular_location.tsv.zip') },
    @{ group = 'uniprot'; name = 'UP000005640_9606.fasta.gz';             urls = @('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz') },
    @{ group = 'uniprot'; name = 'RELEASE.metalink';                      urls = @('https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/RELEASE.metalink') },
    @{ group = 'mane';    name = 'MANE.GRCh38.v1.5.summary.txt.gz';       urls = @('https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz') },
    @{ group = 'mane';    name = 'MANE.GRCh38.v1.5.refseq_protein.faa.gz'; urls = @('https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.refseq_protein.faa.gz') },
    @{ group = 'impc';    name = 'genotype-phenotype-assertions-ALL.csv.gz'; urls = @('https://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/latest/results/genotype-phenotype-assertions-ALL.csv.gz') },
    @{ group = 'ontology'; name = 'mp.obo';                               urls = @('https://purl.obolibrary.org/obo/mp.obo') },
    @{ group = 'mgi';     name = 'HOM_MouseHumanSequence.rpt';            urls = @('https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt') },
    @{ group = 'mapping'; name = 'mp_hp_mgi_all.sssom.tsv';               urls = @('https://raw.githubusercontent.com/mapping-commons/mh_mapping_initiative/master/mappings/mp_hp_mgi_all.sssom.tsv') }
)
if ($Large) {
    $items = @(
        @{ group = 'hpa'; name = 'subcell_image_umap_features.tsv.zip'; urls = @('https://www.proteinatlas.org/download/subcell_image_umap_features.tsv.zip', 'https://www.proteinatlas.org/download/tsv/subcell_image_umap_features.tsv.zip') }
    )
}

New-Item -ItemType Directory -Force $Root | Out-Null
$log = Join-Path $Root 'download_log_v71.csv'
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
Write-Output "DONE download_external_v71"
