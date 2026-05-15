#!/usr/bin/env python3
"""Prepare small ShapeNetCore samples for the SMART Mesh2Tet pipeline.

SMART's legacy Mesh2Tet code expects:

    data/shapenet_airplane/<model_id>/model.obj
    data/shapenet_chair/<model_id>/model.obj
    data/shapenet_table/<model_id>/model.obj

This script can extract that layout from either a local ShapeNetCore v1/v2
directory or category zip archives. It can also download the gated Hugging Face
ShapeNetCore category zips when HF_TOKEN is available and access has been
granted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CATEGORIES = {
    "airplane": "02691156",
    "chair": "03001627",
    "table": "04379243",
}


@dataclass
class ObjStats:
    vertices: int
    minimum: list[float]
    maximum: list[float]
    bbox_center: list[float]
    bbox_extents: list[float]
    bbox_diagonal: float
    max_radius_from_origin: float
    mean: list[float]
    mean_norm: float
    bbox_center_norm: float
    detected_normalization: str


@dataclass
class PreparedModel:
    category: str
    synset: str
    model_id: str
    source: str
    output: str
    version_guess: str
    stats: ObjStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 10 ShapeNet airplane/chair/table meshes for SMART."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Existing ShapeNetCore root or category folder to sample from.",
    )
    parser.add_argument(
        "--archive-file",
        type=Path,
        default=None,
        help="Single ShapeNetCore archive containing all categories, such as archive.zip.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Directory containing category zips such as 02691156.zip.",
    )
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help="Download missing category zips from Hugging Face ShapeNet/ShapeNetCore. Requires HF_TOKEN and accepted access.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_TOKEN/HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Output root. Default: data",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["airplane", "chair", "table"],
        choices=sorted(CATEGORIES),
        help="Categories to prepare.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of valid meshes per category.",
    )
    parser.add_argument(
        "--prefer-small",
        action="store_true",
        help="Pick the smallest OBJ files first to keep sample data compact.",
    )
    parser.add_argument(
        "--normalize",
        choices=["preserve", "bbox-diagonal", "unit-sphere"],
        default="preserve",
        help="Whether to rewrite OBJ vertices. Preserve is recommended for ShapeNetCore normalized archives.",
    )
    parser.add_argument(
        "--require-normalized",
        action="store_true",
        help="Fail when output meshes are not close to SMART/ShapeNet bbox-diagonal normalization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing category sample folders.",
    )
    return parser.parse_args()


def read_obj_vertices(lines: Iterable[str]) -> list[list[float]]:
    vertices: list[list[float]] = []
    for line in lines:
        if line.startswith("v "):
            fields = line.split()
            if len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return vertices


def get_stats(vertices: list[list[float]]) -> ObjStats:
    if not vertices:
        raise ValueError("OBJ has no vertices")

    n = len(vertices)
    minimum = [min(v[i] for v in vertices) for i in range(3)]
    maximum = [max(v[i] for v in vertices) for i in range(3)]
    center = [(minimum[i] + maximum[i]) * 0.5 for i in range(3)]
    extents = [maximum[i] - minimum[i] for i in range(3)]
    diagonal = math.sqrt(sum(x * x for x in extents))
    max_radius = max(math.sqrt(sum(coord * coord for coord in v)) for v in vertices)
    mean = [sum(v[i] for v in vertices) / n for i in range(3)]
    mean_norm = math.sqrt(sum(x * x for x in mean))
    center_norm = math.sqrt(sum(x * x for x in center))

    if abs(diagonal - 1.0) <= 0.05 and center_norm <= 0.05:
        normalization = "bbox-diagonal-centered"
    elif abs(max(extents) - 1.0) <= 0.05 and center_norm <= 0.05:
        normalization = "unit-cube-centered"
    elif abs(max_radius - 1.0) <= 0.05:
        normalization = "unit-sphere-ish"
    elif abs(diagonal - 1.0) <= 0.05 and mean_norm <= 0.05:
        normalization = "centroid-diagonal"
    else:
        normalization = "unknown"

    return ObjStats(
        vertices=n,
        minimum=minimum,
        maximum=maximum,
        bbox_center=center,
        bbox_extents=extents,
        bbox_diagonal=diagonal,
        max_radius_from_origin=max_radius,
        mean=mean,
        mean_norm=mean_norm,
        bbox_center_norm=center_norm,
        detected_normalization=normalization,
    )


def normalize_obj_lines(lines: list[str], mode: str) -> tuple[list[str], ObjStats]:
    vertices = read_obj_vertices(lines)
    stats = get_stats(vertices)
    if mode == "preserve":
        return lines, stats

    if mode == "bbox-diagonal":
        center = stats.bbox_center
        scale = stats.bbox_diagonal
    elif mode == "unit-sphere":
        center = stats.mean
        scale = max(
            math.sqrt(sum((v[i] - center[i]) ** 2 for i in range(3))) for v in vertices
        )
    else:
        raise ValueError(f"unknown normalization mode: {mode}")

    if scale <= 0:
        raise ValueError("degenerate OBJ scale")

    out: list[str] = []
    for line in lines:
        if line.startswith("v "):
            fields = line.split()
            x, y, z = (float(fields[1]), float(fields[2]), float(fields[3]))
            nx = (x - center[0]) / scale
            ny = (y - center[1]) / scale
            nz = (z - center[2]) / scale
            suffix = " " + " ".join(fields[4:]) if len(fields) > 4 else ""
            out.append(f"v {nx:.9g} {ny:.9g} {nz:.9g}{suffix}\n")
        else:
            out.append(line if line.endswith("\n") else line + "\n")
    return out, get_stats(read_obj_vertices(out))


def local_candidates(source_dir: Path, synset: str) -> list[tuple[str, Path, str]]:
    roots = []
    if (source_dir / synset).is_dir():
        roots.append(source_dir / synset)
    roots.append(source_dir)

    found: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            candidates = [
                (model_dir / "model.obj", "v1"),
                (model_dir / "models" / "model_normalized.obj", "v2"),
            ]
            for obj_path, version_guess in candidates:
                if obj_path.exists() and obj_path not in seen:
                    found.append((model_dir.name, obj_path, version_guess))
                    seen.add(obj_path)
                    break
    return found


def zip_model_id(path: str, version_guess: str) -> str:
    parts = [p for p in Path(path).parts if p not in ("", ".")]
    if version_guess == "v2":
        return parts[-3]
    return parts[-2]


def zip_candidates(zip_path: Path, synset: str | None = None) -> list[tuple[str, str, str, int]]:
    found: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if synset is not None and f"/{synset}/" not in f"/{name}":
                continue
            if name.endswith("/model.obj"):
                found.append((zip_model_id(name, "v1"), name, "v1", info.file_size))
            elif name.endswith("/models/model_normalized.obj"):
                found.append((zip_model_id(name, "v2"), name, "v2", info.file_size))
    found.sort(key=lambda item: (item[3], item[0]) if synset else item[1])
    return found


def download_hf_zip(synset: str, archive_dir: Path, token: str | None) -> Path:
    if not token:
        raise RuntimeError(
            "HF token is required. Accept ShapeNet/ShapeNetCore access on Hugging Face "
            "and set HF_TOKEN before running --download-hf."
        )
    archive_dir.mkdir(parents=True, exist_ok=True)
    out = archive_dir / f"{synset}.zip"
    if out.exists():
        return out

    url = f"https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/{synset}.zip"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response, out.open("wb") as f:
        shutil.copyfileobj(response, f)
    return out


def write_sample(
    category: str,
    synset: str,
    model_id: str,
    version_guess: str,
    source: str,
    lines: list[str],
    args: argparse.Namespace,
) -> PreparedModel:
    output_dir = args.output_root / f"shapenet_{category}" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_lines, stats = normalize_obj_lines(lines, args.normalize)
    output_path = output_dir / "model.obj"
    output_path.write_text("".join(obj_lines))

    if args.require_normalized:
        ok = (
            stats.detected_normalization == "bbox-diagonal-centered"
            or stats.detected_normalization == "unit-cube-centered"
        )
        if not ok:
            raise RuntimeError(
                f"{category}/{model_id} is not close to SMART/ShapeNet normalization: "
                f"{stats.detected_normalization}"
            )

    return PreparedModel(
        category=category,
        synset=synset,
        model_id=model_id,
        source=source,
        output=str(output_path),
        version_guess=version_guess,
        stats=stats,
    )


def prepare_category(category: str, args: argparse.Namespace) -> list[PreparedModel]:
    synset = CATEGORIES[category]
    out_category = args.output_root / f"shapenet_{category}"
    if out_category.exists() and args.overwrite:
        shutil.rmtree(out_category)
    out_category.mkdir(parents=True, exist_ok=True)

    prepared: list[PreparedModel] = []

    if args.source_dir is not None:
        candidates = local_candidates(args.source_dir, synset)
        if args.prefer_small:
            candidates.sort(key=lambda item: (item[1].stat().st_size, item[0]))
        for model_id, obj_path, version_guess in candidates:
            lines = obj_path.read_text(errors="replace").splitlines(keepends=True)
            prepared.append(
                write_sample(
                    category,
                    synset,
                    model_id,
                    version_guess,
                    str(obj_path),
                    lines,
                    args,
                )
            )
            if len(prepared) >= args.limit:
                return prepared

    if args.archive_file is not None and args.archive_file.exists():
        with zipfile.ZipFile(args.archive_file) as zf:
            candidates = zip_candidates(args.archive_file, synset=synset)
            if not args.prefer_small:
                candidates.sort(key=lambda item: item[1])
            for model_id, member, version_guess, _size in candidates:
                with zf.open(member) as obj_file:
                    text = obj_file.read().decode("utf-8", errors="replace")
                prepared.append(
                    write_sample(
                        category,
                        synset,
                        model_id,
                        version_guess,
                        f"{args.archive_file}:{member}",
                        text.splitlines(keepends=True),
                        args,
                    )
                )
                if len(prepared) >= args.limit:
                    return prepared

    archive_dir = args.archive_dir
    if args.download_hf:
        archive_dir = archive_dir or args.output_root / "downloads" / "shapenetcore"
        download_hf_zip(synset, archive_dir, args.hf_token)

    if archive_dir is not None:
        zip_path = archive_dir / f"{synset}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                candidates = zip_candidates(zip_path)
                if args.prefer_small:
                    candidates.sort(key=lambda item: (item[3], item[0]))
                for model_id, member, version_guess, _size in candidates:
                    with zf.open(member) as obj_file:
                        text = obj_file.read().decode("utf-8", errors="replace")
                    prepared.append(
                        write_sample(
                            category,
                            synset,
                            model_id,
                            version_guess,
                            f"{zip_path}:{member}",
                            text.splitlines(keepends=True),
                            args,
                        )
                    )
                    if len(prepared) >= args.limit:
                        return prepared

    return prepared


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[PreparedModel] = []
    for category in args.categories:
        prepared = prepare_category(category, args)
        if len(prepared) < args.limit:
            print(
                f"warning: prepared only {len(prepared)}/{args.limit} {category} meshes",
                file=sys.stderr,
            )
        manifest.extend(prepared)

    manifest_path = args.output_root / "shapenet_samples_manifest.json"
    manifest_path.write_text(json.dumps([asdict(item) for item in manifest], indent=2))

    for item in manifest:
        stats = item.stats
        print(
            f"{item.category:8s} {item.model_id:32s} "
            f"{stats.detected_normalization:24s} "
            f"diag={stats.bbox_diagonal:.6f} "
            f"center_norm={stats.bbox_center_norm:.6f} "
            f"radius={stats.max_radius_from_origin:.6f}"
        )
    print(f"wrote manifest: {manifest_path}")

    missing = [c for c in args.categories if not (args.output_root / f"shapenet_{c}").exists()]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
