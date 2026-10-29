from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


USER_AGENT = "bio-paper-data-downloader/1.0"


@dataclass(frozen=True)
class DownloadItem:
    group: str
    name: str
    filename: str
    urls: tuple[str, ...]
    note: str = ""


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_catalog(string_version: str = "12.0", string_network: str = "physical") -> list[DownloadItem]:
    string_prefix = f"https://stringdb-downloads.org/download"
    if string_network == "functional":
        string_links_dir = f"protein.links.v{string_version}"
        string_links_file = f"9606.protein.links.v{string_version}.txt.gz"
        string_note = "STRING Homo sapiens functional association network"
    else:
        string_links_dir = f"protein.physical.links.v{string_version}"
        string_links_file = f"9606.protein.physical.links.v{string_version}.txt.gz"
        string_note = "STRING Homo sapiens physical interaction network"

    return [
        DownloadItem(
            group="association",
            name="ctd_genes_diseases",
            filename="CTD_genes_diseases.tsv.gz",
            urls=(
                "https://ctdbase.org/reports/CTD_genes_diseases.tsv.gz",
                "http://ctdbase.org/reports/CTD_genes_diseases.tsv.gz",
            ),
            note="CTD curated and inferred gene-disease associations",
        ),
        DownloadItem(
            group="association",
            name="ctd_diseases",
            filename="CTD_diseases.tsv.gz",
            urls=(
                "https://ctdbase.org/reports/CTD_diseases.tsv.gz",
                "http://ctdbase.org/reports/CTD_diseases.tsv.gz",
            ),
            note="CTD disease vocabulary",
        ),
        DownloadItem(
            group="phenotype",
            name="hp_obo",
            filename="hp.obo",
            urls=("https://purl.obolibrary.org/obo/hp.obo", "http://purl.obolibrary.org/obo/hp.obo"),
            note="Human Phenotype Ontology in OBO format",
        ),
        DownloadItem(
            group="phenotype",
            name="phenotype_hpoa",
            filename="phenotype.hpoa",
            urls=(
                "https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
                "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
            ),
            note="HPO disease-to-phenotype annotations",
        ),
        DownloadItem(
            group="phenotype",
            name="genes_to_phenotype",
            filename="genes_to_phenotype.txt",
            urls=(
                "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
                "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt",
            ),
            note="HPO gene-to-phenotype annotations",
        ),
        DownloadItem(
            group="function",
            name="gene2go",
            filename="gene2go.gz",
            urls=(
                "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz",
                "https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz",
            ),
            note="NCBI Gene-to-GO annotations",
        ),
        DownloadItem(
            group="function",
            name="homo_sapiens_gene_info",
            filename="Homo_sapiens.gene_info.gz",
            urls=(
                "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
                "https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            ),
            note="NCBI human gene metadata",
        ),
        DownloadItem(
            group="ontology",
            name="doid_obo",
            filename="doid.obo",
            urls=(
                "https://purl.obolibrary.org/obo/doid.obo",
                "http://purl.obolibrary.org/obo/doid.obo",
            ),
            note="Human Disease Ontology in OBO format",
        ),
        DownloadItem(
            group="ppi",
            name=f"string_{string_network}_links",
            filename=string_links_file,
            urls=(f"{string_prefix}/{string_links_dir}/{string_links_file}",),
            note=string_note,
        ),
        DownloadItem(
            group="ppi",
            name="string_protein_info",
            filename=f"9606.protein.info.v{string_version}.txt.gz",
            urls=(
                f"{string_prefix}/protein.info.v{string_version}/9606.protein.info.v{string_version}.txt.gz",
            ),
            note="STRING protein display names and descriptions",
        ),
        DownloadItem(
            group="ppi",
            name="string_protein_aliases",
            filename=f"9606.protein.aliases.v{string_version}.txt.gz",
            urls=(
                f"{string_prefix}/protein.aliases.v{string_version}/9606.protein.aliases.v{string_version}.txt.gz",
            ),
            note="STRING protein aliases for ID mapping",
        ),
    ]


def parse_only(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    selected: set[str] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                selected.add(part)
    return selected


def select_items(catalog: list[DownloadItem], selected: set[str]) -> list[DownloadItem]:
    if not selected or "all" in selected:
        return catalog
    out = [item for item in catalog if item.group in selected or item.name in selected]
    unknown = selected - {item.group for item in catalog} - {item.name for item in catalog}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown --only value(s): {names}")
    return out


def sizeof_fmt(num_bytes: int | None) -> str:
    if num_bytes is None or num_bytes < 0:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def make_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def remote_size(url: str, timeout: int) -> int | None:
    request = make_request(url)
    request.get_method = lambda: "HEAD"  # type: ignore[method-assign]
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_url(url: str, dest: Path, timeout: int, chunk_size: int = 1024 * 1024) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = make_request(url)
    downloaded = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total else None
        with tmp.open("wb") as out:
            last_print = time.time()
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2:
                    print(f"  {dest.name}: {sizeof_fmt(downloaded)} / {sizeof_fmt(total_size)}")
                    last_print = now
    if total_size is not None and downloaded != total_size:
        raise IOError(f"incomplete download: got {downloaded} bytes, expected {total_size}")
    if dest.name.endswith(".gz"):
        with gzip.open(tmp, "rb") as gz:
            while gz.read(chunk_size):
                pass
    tmp.replace(dest)
    return downloaded


def download_item(
    item: DownloadItem,
    root: Path,
    overwrite: bool,
    dry_run: bool,
    retries: int,
    timeout: int,
) -> dict:
    dest_dir = root / item.group
    dest = dest_dir / item.filename
    record = {
        "group": item.group,
        "name": item.name,
        "path": str(dest),
        "urls": list(item.urls),
        "note": item.note,
        "status": "pending",
        "bytes": None,
        "error": None,
    }

    if dest.exists() and not overwrite:
        record["status"] = "skipped_existing"
        record["bytes"] = dest.stat().st_size
        print(f"[skip] {item.name}: {dest}")
        return record

    if dry_run:
        size = remote_size(item.urls[0], timeout=timeout)
        record["status"] = "dry_run"
        record["bytes"] = size
        print(f"[dry-run] {item.name}: {dest} ({sizeof_fmt(size)})")
        return record

    dest_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and dest.exists():
        dest.unlink()

    errors: list[str] = []
    for url in item.urls:
        for attempt in range(1, retries + 1):
            try:
                print(f"[download] {item.name}: {url}")
                bytes_written = download_url(url, dest, timeout=timeout)
                record["status"] = "downloaded"
                record["bytes"] = bytes_written
                return record
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
                message = f"{url} attempt {attempt}/{retries}: {exc}"
                errors.append(message)
                print(f"  failed: {message}")
                part = dest.with_suffix(dest.suffix + ".part")
                if part.exists():
                    part.unlink()
                if attempt < retries:
                    time.sleep(min(10, 2 * attempt))

    record["status"] = "failed"
    record["error"] = " | ".join(errors)
    raise RuntimeError(f"Failed to download {item.name}: {record['error']}")


def write_manifest(root: Path, records: list[dict], args: argparse.Namespace) -> Path:
    manifest_path = root / "raw_download_manifest.json"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "command": " ".join(sys.argv),
        "options": {
            "only": args.only,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "string_version": args.string_version,
            "string_network": args.string_network,
        },
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def check_disk_space(root: Path, min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"Only {free_gb:.2f} GB free under {root}; need at least {min_free_gb:.2f} GB.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download raw public data for the disease-gene prediction project.")
    parser.add_argument("--root", type=Path, default=project_root(), help="Project data root. Default: parent of code directory.")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Comma-separated groups or item names to download. Groups: association, phenotype, function, ontology, ppi, all.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redownload files even if they already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned downloads without writing data files.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue downloading other files if one item fails.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per URL.")
    parser.add_argument("--timeout", type=int, default=120, help="Network timeout in seconds.")
    parser.add_argument("--min_free_gb", type=float, default=2.0, help="Abort if less free disk space is available.")
    parser.add_argument("--string_version", default="12.0", help="STRING version, e.g. 12.0.")
    parser.add_argument(
        "--string_network",
        choices=("physical", "functional"),
        default="physical",
        help="STRING link file to download. Physical is smaller; functional is broader.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = args.root.resolve()
    check_disk_space(root, args.min_free_gb)

    catalog = build_catalog(string_version=args.string_version, string_network=args.string_network)
    selected = parse_only(args.only)
    items = select_items(catalog, selected)

    print(f"Root: {root}")
    print("Selected items:")
    for item in items:
        print(f"  - {item.group}/{item.name}: {item.filename}")

    records: list[dict] = []
    failed = 0
    for item in items:
        try:
            records.append(
                download_item(
                    item,
                    root=root,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    retries=max(1, args.retries),
                    timeout=max(5, args.timeout),
                )
            )
        except Exception as exc:
            failed += 1
            records.append(
                {
                    **asdict(item),
                    "path": str(root / item.group / item.filename),
                    "urls": list(item.urls),
                    "status": "failed",
                    "bytes": None,
                    "error": str(exc),
                }
            )
            print(f"[error] {item.name}: {exc}")
            if not args.continue_on_error:
                break

    manifest_path = write_manifest(root, records, args)
    print(f"Manifest: {manifest_path}")
    if failed:
        print(f"Finished with {failed} failed item(s).")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
