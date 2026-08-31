"""LSA64 fetch with integrity check; AUTSL registration instructions.

LSA64 is distributed from Google Drive and Mega. The Google Drive endpoint supports HTTP
range requests, so the download here is resumable -- a 1.5 GB fetch that dies at 90 % on a
laptop wifi should not start over.

Note on the upstream page: https://facundoq.github.io/datasets/lsa64/ mislabels its links.
Both the "Raw" and "Cut" Google Drive links point at the *same* file id (the raw archive),
and the archive actually containing the cut clips is behind the link labelled
"Preprocessed Version (800mb) - Mega". The file ids below were resolved by reading the
``content-disposition`` header of each endpoint, not by trusting the labels.

Usage:
    python -m signadapt.data.download --config configs/data.yaml
    python -m signadapt.data.download --config configs/data.yaml --variant raw
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from signadapt.utils.config import load_config

__all__ = ["DRIVE_VARIANTS", "DriveFile", "download_drive_file", "sha256_file", "extract_archive"]

_DRIVE_ENDPOINT = "https://drive.usercontent.google.com/download"
_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DriveFile:
    """A Google Drive hosted archive.

    Attributes:
        file_id: Drive file id.
        filename: Name the server reports in ``content-disposition``.
        size_bytes: Expected total size, used as a cheap integrity check.
        description: What the archive contains.
    """

    file_id: str
    filename: str
    size_bytes: int
    description: str


#: LSA64 variants. ``cut`` is the one the project uses: clips trimmed to the sign itself.
DRIVE_VARIANTS: dict[str, DriveFile] = {
    "cut": DriveFile(
        file_id="18VuWBAxHaSBbO7wx57kQVre78FN7GYzQ",
        filename="lsa64_cut.zip",
        size_bytes=1_504_519_604,
        description="3200 clips trimmed to the sign, 1920x1080, named NNN_SSS_RRR.mp4",
    ),
    "raw": DriveFile(
        file_id="1C7k_m2m4n5VzI4lljMoezc-uowDEgIUh",
        filename="lsa64_raw.zip",
        size_bytes=1_899_156_443,
        description="untrimmed recordings, includes idle frames before and after each sign",
    ),
    "preprocessed": DriveFile(
        file_id="1yhfPpI2iJzPXyx4C7MYR6IPZC3YuuYaL",
        filename="lsa64_preprocessed.zip",
        size_bytes=803_693_904,
        description="authors' hand-segmented version; not used here, we extract our own keypoints",
    ),
}

AUTSL_INSTRUCTIONS = """\
AUTSL (226 signs, 43 signers, ~38k clips) is not downloadable without registration.

  1. Register at the ChaLearn LAP challenge page for AUTSL and accept the terms.
  2. Approval is manual and can take days -- PLAN.md section 5 says do this in week 1,
     because it is the one latency you cannot compress.
  3. You receive per-file download links plus the password for the encrypted archives.
  4. Place the archives under data/raw/autsl/ and re-run this command with
     --dataset autsl to verify their checksums.

AUTSL ships an official signer-independent split; use it rather than inventing one.
LSA64 alone is sufficient for a complete thesis (PLAN.md section 10) -- AUTSL is an
upgrade, not a dependency.
"""


def sha256_file(path: Path, *, chunk: int = _CHUNK) -> str:
    """Compute the SHA-256 of a file without loading it into memory.

    Args:
        path: File to hash.
        chunk: Read size in bytes.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_drive_file(spec: DriveFile, dest: Path, *, force: bool = False) -> Path:
    """Download a Drive-hosted archive, resuming an interrupted transfer if possible.

    Args:
        spec: The archive to fetch.
        dest: Directory to download into.
        force: Delete any existing partial or complete file first.

    Returns:
        Path to the downloaded archive.

    Raises:
        RuntimeError: If the server refuses the request or the final size is wrong.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / spec.filename
    if force:
        target.unlink(missing_ok=True)
    if target.is_file() and target.stat().st_size == spec.size_bytes:
        print(f"[download] {spec.filename} already present and complete")
        return target

    have = target.stat().st_size if target.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    if have:
        print(f"[download] resuming {spec.filename} at {have / 1e6:.1f} MB")

    params = {"id": spec.file_id, "export": "download", "confirm": "t"}
    with requests.get(
        _DRIVE_ENDPOINT, params=params, headers=headers, stream=True, timeout=60
    ) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(
                f"drive returned HTTP {response.status_code} for {spec.filename}; "
                "the file may have been moved -- download it by hand from "
                "https://facundoq.github.io/datasets/lsa64/"
            )
        if have and response.status_code == 200:  # server ignored the range: start over
            have, target = 0, target
            mode = "wb"
        else:
            mode = "ab" if have else "wb"

        with (
            target.open(mode) as handle,
            tqdm(
                total=spec.size_bytes,
                initial=have,
                unit="B",
                unit_scale=True,
                desc=spec.filename,
            ) as bar,
        ):
            for block in response.iter_content(chunk_size=_CHUNK):
                handle.write(block)
                bar.update(len(block))

    size = target.stat().st_size
    if size != spec.size_bytes:
        raise RuntimeError(
            f"{spec.filename}: expected {spec.size_bytes} bytes, got {size}. "
            "The download is truncated -- re-run to resume."
        )
    return target


def extract_archive(archive: Path, dest: Path, *, force: bool = False) -> Path:
    """Extract a zip archive, skipping the work if it already looks extracted.

    Args:
        archive: Zip file to extract.
        dest: Directory to extract into.
        force: Remove ``dest`` first.

    Returns:
        The directory extracted into.
    """
    if force and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        existing = sum(1 for m in members if (dest / m).is_file())
        if existing == len(members):
            print(f"[extract] {dest} already holds all {len(members)} files")
            return dest
        for member in tqdm(members, desc=f"extract {archive.name}", unit="file"):
            zf.extract(member, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dataset", default="lsa64", choices=["lsa64", "autsl"])
    parser.add_argument("--variant", default="cut", choices=sorted(DRIVE_VARIANTS))
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset == "autsl":
        print(AUTSL_INSTRUCTIONS)
        return 0

    cfg = load_config(args.config)
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    spec = DRIVE_VARIANTS[args.variant]
    print(f"[download] {spec.filename}: {spec.description}")

    archive = download_drive_file(spec, raw_dir.parent, force=args.force)

    digest = sha256_file(archive)
    expected = cfg["dataset"].get("archive_sha256")
    if expected and expected != digest:
        print(f"[verify] FAIL sha256 {digest} != configured {expected}", file=sys.stderr)
        return 1
    if not expected:
        print(f"[verify] sha256 = {digest}")
        print("[verify] record this in configs/data.yaml as dataset.archive_sha256")
    else:
        print(f"[verify] sha256 OK ({digest[:16]}...)")

    if not args.no_extract:
        extract_archive(archive, raw_dir, force=args.force)
        videos = sorted(raw_dir.rglob("*.mp4"))
        print(f"[extract] {len(videos)} mp4 files under {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
