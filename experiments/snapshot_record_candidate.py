from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKBENCH = ROOT / "records" / "track_10min_16mb" / "2026-03-20_SOTA_Workbench"
VENDORED_FILES = {
    "lowbit_utils.py": "lowbit_utils.py",
    "optimizer_variants.py": "optimizer_variants.py",
    "validation_utils.py": "validation_utils.py",
    "mlp_variants.py": "mlp_variants.py",
    "train_gpt.py": "root_train_gpt_vendor.py",
}


def snapshot_candidate(
    target: Path,
    *,
    source: Path = WORKBENCH,
    overwrite: bool = False,
    include_log: bool = True,
) -> Path:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists; pass overwrite=True to replace it.")
        shutil.rmtree(target)
    shutil.copytree(source, target)
    for source_name, target_name in VENDORED_FILES.items():
        shutil.copy2(ROOT / source_name, target / target_name)
    if not include_log:
        log_path = target / "train.log"
        if log_path.exists():
            log_path.unlink()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot the SOTA workbench into a portable records candidate folder."
    )
    parser.add_argument("name", help="Target folder name under records/track_10min_16mb")
    parser.add_argument("--source", default=str(WORKBENCH), help="Source workbench folder to snapshot")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--without-log",
        action="store_true",
        help="Drop train.log from the copied folder. Useful before a real measured run exists.",
    )
    args = parser.parse_args()

    target = ROOT / "records" / "track_10min_16mb" / args.name
    snapshot_candidate(
        target,
        source=Path(args.source),
        overwrite=args.overwrite,
        include_log=not args.without_log,
    )
    print(target)


if __name__ == "__main__":
    main()
