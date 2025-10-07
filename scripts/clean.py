#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Set


ROOT = Path(__file__).resolve().parents[1]


PATTERNS_DEFAULT = [
    # Python caches
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyo",
    
    # Generated masks/temps
    "temp_*",
    "temp_*.*",
    "temp_mask_for_removal.png",
    "temp_pil_mask_for_removal.png",
    "temp_improved_mask_for_removal.png",
    "temp_mask_debug_*.png",
    "*_mask.png",
    "*_detected_mask.png",
    "*_pil_detected_mask.png",
    "*_lightweight_detected_mask.png",
    "*_improved_detected_mask.png",
    
    # Outputs
    "*-output.jpg",
    "*-output.png",
    "*_enhanced_output.png",
    "*_cleaned*.png",
    "perfect_output.png",
    "*.webp",
    "final_outputs.webp",

    # Sample datasets or demo folders
    "watermark-available",
    "watermark-unavailable",
]


def tracked_files(root: Path) -> Set[Path]:
    git = root / ".git"
    if not git.exists():
        return set()
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        files = set()
        for p in out.split(b"\x00"):
            if p:
                files.add((root / p.decode("utf-8")).resolve())
        return files
    except Exception:
        return set()


def iter_matches(root: Path, patterns: Iterable[str]) -> List[Path]:
    result: List[Path] = []
    for pat in patterns:
        # Use glob with recursive patterns
        result.extend(root.glob(pat))
    # Deduplicate and keep inside repo
    uniq = []
    seen = set()
    for p in result:
        try:
            rp = p.resolve()
        except FileNotFoundError:
            continue
        if ROOT not in rp.parents and rp != ROOT:
            continue
        if rp not in seen and rp.exists():
            uniq.append(rp)
            seen.add(rp)
    return uniq


def remove_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"DRY-RUN remove: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser(description="Clean generated artifacts from the repo")
    ap.add_argument("--dry-run", action="store_true", help="List what would be removed")
    ap.add_argument("--include-tracked", action="store_true", help="Allow deleting tracked files (use with care)")
    ap.add_argument("--remove-venv", action="store_true", help="Also remove .venv if present")
    ap.add_argument("--patterns", nargs="*", default=None, help="Extra glob patterns to remove")
    args = ap.parse_args()

    pats = list(PATTERNS_DEFAULT)
    if args.patterns:
        pats.extend(args.patterns)

    to_remove = iter_matches(ROOT, pats)

    # Filter tracked files unless explicitly allowed
    if not args.include_tracked:
        tracked = tracked_files(ROOT)
        to_remove = [p for p in to_remove if p.resolve() not in tracked]

    # .venv is protected by default
    venv_path = ROOT / ".venv"
    if args.remove_venv and venv_path.exists():
        to_remove.append(venv_path.resolve())

    if not to_remove:
        print("Nothing to clean.")
        return

    # Sort: files before dirs for nicer output
    to_remove.sort(key=lambda p: (p.is_dir(), str(p)))

    for p in to_remove:
        remove_path(p, args.dry_run)

    print(("DRY-RUN: " if args.dry_run else "") + f"Cleaned {len(to_remove)} items.")


if __name__ == "__main__":
    main()
