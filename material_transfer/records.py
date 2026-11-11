"""Evaluation-facing input manifest and self-contained run records."""
import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pair:
    pair_id: str
    target: Path
    exemplar: Path
    mask: Path | None
    depth: Path | None
    material_prompt: str
    seed: int | None = None


def read_manifest(path):
    path = Path(path).resolve()
    pairs, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"pair_id", "target", "exemplar", "mask", "depth", "material_prompt"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        for row in reader:
            pair_id = row["pair_id"].strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", pair_id) or pair_id in seen:
                raise ValueError(f"Unsafe or duplicate pair_id: {pair_id!r}")
            seen.add(pair_id)
            paths = {}
            for field in ("target", "exemplar", "mask", "depth"):
                value = row[field].strip()
                if not value and field in {"target", "exemplar"}:
                    raise ValueError(f"{pair_id}: missing {field}")
                paths[field] = (path.parent / value).resolve() if value else None
                if paths[field] is not None and not paths[field].is_file():
                    raise FileNotFoundError(f"{pair_id}: {paths[field]}")
            seed = int(row["seed"]) if row.get("seed", "").strip() else None
            pairs.append(Pair(pair_id, **paths, material_prompt=row["material_prompt"], seed=seed))
    if not pairs:
        raise ValueError("Manifest has no pairs")
    return pairs


def source_state():
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    digest = hashlib.sha256()
    for path in sorted((root / "material_transfer").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return dict(git_commit=revision, git_dirty=dirty, package_source_sha256=digest.hexdigest())


def save_run(directory, artifacts, config, timings, inputs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "run.json").exists():
        raise FileExistsError(f"Completed run already exists: {directory}")
    for name, image in artifacts.items():
        image.save(directory / f"{name}.png")
    versions = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = dict(schema_version=1, config=asdict(config), inputs=inputs,
                  timings=timings, versions=versions, python=sys.version,
                  artifacts={name: f"{name}.png" for name in artifacts}, **source_state())
    # A completed run is identifiable only after images and metadata are written.
    temporary = directory / "run.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(directory / "run.json")
    return directory / "result.png"
