import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import yaml

from src import ROOT


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def list_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for item in paths:
        root = repo_path(item)
        if not root.exists():
            raise FileNotFoundError(f"image source does not exist: {root}")
        images.extend(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return sorted(images, key=lambda p: rel(p))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def map_variant(raw_image: Path, image_root: str) -> Path:
    raw_rel = raw_image.relative_to(ROOT / "data")
    return ROOT / image_root / raw_rel


def load_tile_manifest(path: str | None) -> dict[str, list[dict]] | None:
    if not path:
        return None
    manifest_path = repo_path(path)
    with manifest_path.open() as f:
        manifest = json.load(f)
    return manifest["tiles_by_source"]


def label_for_image(image: Path, label_root: str | None = None) -> Path:
    parts = list(image.relative_to(ROOT).parts)
    if "images" not in parts:
        raise ValueError(f"cannot infer YOLO label path for image without images segment: {image}")
    parts[parts.index("images")] = "labels"
    if label_root is not None:
        root_parts = Path(label_root).parts
        parts = list(root_parts) + parts[1:]
    return ROOT.joinpath(*parts).with_suffix(".txt")


def write_lines(path: Path, lines: list[str], write: bool) -> None:
    if not write:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_yaml(path: Path, data: dict, write: bool) -> None:
    if not write:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def assert_no_overlap(dev_images: list[Path], test_images: list[Path]) -> dict:
    dev_basenames = {p.name for p in dev_images}
    test_basenames = {p.name for p in test_images}
    basename_overlap = sorted(dev_basenames & test_basenames)

    dev_hashes = {sha256(p): rel(p) for p in dev_images}
    test_hashes = {sha256(p): rel(p) for p in test_images}
    hash_overlap = [
        {"sha256": h, "dev": dev_hashes[h], "test": test_hashes[h]}
        for h in sorted(dev_hashes.keys() & test_hashes.keys())
    ]

    if basename_overlap or hash_overlap:
        raise RuntimeError(
            "locked test overlaps with development images: "
            f"basename_overlap={basename_overlap[:5]}, hash_overlap={hash_overlap[:3]}"
        )
    return {"basename_overlap": [], "hash_overlap": []}


def build_folds(images: list[Path], folds: int, seed: int) -> list[list[Path]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    buckets = [[] for _ in range(folds)]
    for idx, image in enumerate(shuffled):
        buckets[idx % folds].append(image)
    return [sorted(bucket, key=lambda p: rel(p)) for bucket in buckets]


def tile_variant_images(raw_image: Path, tiles_by_source: dict[str, list[dict]]) -> list[Path]:
    raw_rel = rel(raw_image)
    tiles = tiles_by_source.get(raw_rel)
    if not tiles:
        raise FileNotFoundError(f"tile manifest has no entries for source image: {raw_rel}")
    return [repo_path(tile["image"]) for tile in tiles]


def ensure_variant_files(
    images: list[Path],
    image_root: str,
    label_root: str,
    tiles_by_source: dict[str, list[dict]] | None = None,
) -> list[Path]:
    mapped: list[Path] = []
    missing: list[str] = []
    for image in images:
        variant_images = tile_variant_images(image, tiles_by_source) if tiles_by_source is not None else [map_variant(image, image_root)]
        for variant_image in variant_images:
            variant_label = label_for_image(variant_image, label_root)
            if not variant_image.exists():
                missing.append(rel(variant_image))
            if not variant_label.exists():
                missing.append(rel(variant_label))
            mapped.append(variant_image)
    if missing:
        sample = "\n".join(missing[:20])
        raise FileNotFoundError(f"missing files for dataset variant {image_root}:\n{sample}")
    return sorted(mapped, key=lambda p: rel(p))


def generate(protocol_path: Path, write: bool) -> dict:
    with protocol_path.open() as f:
        protocol = yaml.safe_load(f)

    split_cfg = protocol["splits"]
    seed = int(protocol["seed"])
    fold_count = int(protocol["folds"])
    output_dir = repo_path(split_cfg["output_dir"])

    dev_images = list_images(split_cfg["dev_sources"])
    test_images = list_images([split_cfg["locked_test_source"]])
    overlap = assert_no_overlap(dev_images, test_images)
    folds = build_folds(dev_images, fold_count, seed)

    locked_test_file = repo_path(split_cfg["locked_test_file"])
    write_lines(locked_test_file, [rel(p) for p in test_images], write)

    manifest = {
        "protocol": protocol["name"],
        "seed": seed,
        "folds": fold_count,
        "patient_level": bool(protocol.get("patient_level", False)),
        "patient_level_note": protocol.get("patient_level_note", ""),
        "dev_count": len(dev_images),
        "locked_test_count": len(test_images),
        "overlap_check": overlap,
        "folds_detail": [],
        "datasets": {},
    }

    for fold_idx, val_images in enumerate(folds):
        val_set = set(val_images)
        train_images = [p for p in dev_images if p not in val_set]
        manifest["folds_detail"].append(
            {
                "fold": fold_idx,
                "train_count": len(train_images),
                "val_count": len(val_images),
            }
        )

    class_names = protocol.get("class_names", ["tooth"])
    for dataset_name, dataset_cfg in protocol["datasets"].items():
        image_root = dataset_cfg["image_root"]
        label_root = dataset_cfg.get("label_root", image_root)
        tiles_by_source = load_tile_manifest(dataset_cfg.get("tile_manifest"))
        dataset_dir = output_dir / dataset_name
        manifest["datasets"][dataset_name] = {
            "image_root": image_root,
            "label_root": label_root,
            "tile_manifest": dataset_cfg.get("tile_manifest"),
            "yamls": [],
        }

        variant_test = ensure_variant_files(test_images, image_root, label_root, tiles_by_source)
        variant_locked_file = dataset_dir / "locked_test.txt"
        write_lines(variant_locked_file, [rel(p) for p in variant_test], write)

        for fold_idx, val_images in enumerate(folds):
            val_set = set(val_images)
            train_images = [p for p in dev_images if p not in val_set]
            variant_train = ensure_variant_files(train_images, image_root, label_root, tiles_by_source)
            variant_val = ensure_variant_files(val_images, image_root, label_root, tiles_by_source)

            train_file = dataset_dir / f"fold{fold_idx}_train.txt"
            val_file = dataset_dir / f"fold{fold_idx}_val.txt"
            yaml_file = dataset_dir / f"fold{fold_idx}.yaml"
            write_lines(train_file, [rel(p) for p in variant_train], write)
            write_lines(val_file, [rel(p) for p in variant_val], write)
            write_yaml(
                yaml_file,
                {
                    "path": ".",
                    "train": rel(train_file),
                    "val": rel(val_file),
                    "test": rel(variant_locked_file),
                    "nc": len(class_names),
                    "names": class_names,
                },
                write,
            )
            manifest["datasets"][dataset_name]["yamls"].append(rel(yaml_file))

    if write:
        manifest_path = output_dir / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

        csv_path = output_dir / "fold_counts.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fold", "train_count", "val_count"])
            writer.writeheader()
            writer.writerows(manifest["folds_detail"])

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate locked-test and CV split files.")
    parser.add_argument("--protocol", default="configs/protocol/cv5_locked_test.yaml")
    parser.add_argument("--write", action="store_true", help="Write split files. Without this flag, dry-run only.")
    args = parser.parse_args()

    manifest = generate(repo_path(args.protocol), write=args.write)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not args.write:
        print("\nDry run only. Re-run with --write to create split files.")


if __name__ == "__main__":
    main()
