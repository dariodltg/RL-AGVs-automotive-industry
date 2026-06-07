"""
Clean up TensorBoard run directories under logs/tensorboard/.

Always shows what would be deleted and asks for confirmation unless --yes
is passed. Use --dry-run to preview without deleting.

Usage
-----
    # List every TB run
    python scripts/clean_tensorboard.py --dry-run

    # Delete everything (asks for confirmation)
    python scripts/clean_tensorboard.py --all

    # Delete every PPO run from the grid by pattern
    python scripts/clean_tensorboard.py --pattern "ppo_*"

    # Delete one specific run, no prompt
    python scripts/clean_tensorboard.py --name PPO_3 --yes

    # Delete runs older than 14 days
    python scripts/clean_tensorboard.py --older-than 14
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TB_DIR = Path("logs/tensorboard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean TensorBoard run dirs")
    parser.add_argument("--dir", type=Path, default=_TB_DIR,
                        help=f"TensorBoard root (default: {_TB_DIR})")
    parser.add_argument("--all",     action="store_true", help="Delete every TB run")
    parser.add_argument("--name",    type=str, default=None,
                        help="Delete one TB run by exact directory name")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern to match TB run directory names")
    parser.add_argument("--older-than", type=int, default=None, metavar="DAYS",
                        help="Delete runs whose mtime is older than DAYS")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted and exit")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt")
    return parser.parse_args()


def _matches(path: Path, args: argparse.Namespace) -> bool:
    if args.name and path.name != args.name:
        return False
    if args.pattern and not fnmatch.fnmatch(path.name, args.pattern):
        return False
    if args.older_than is not None:
        cutoff = time.time() - args.older_than * 86400
        if path.stat().st_mtime >= cutoff:
            return False
    return True


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> None:
    args = parse_args()

    if not args.dir.exists():
        print(f"TB dir not found: {args.dir}")
        sys.exit(1)

    if not (args.all or args.name or args.pattern or args.older_than is not None
            or args.dry_run):
        print("Refusing to delete: pass a filter (--all, --name, --pattern, "
              "--older-than) or --dry-run.")
        sys.exit(2)

    all_runs = sorted(p for p in args.dir.iterdir() if p.is_dir())
    if args.all:
        matches = all_runs
    elif args.dry_run and not (args.name or args.pattern or args.older_than is not None):
        matches = all_runs       # pure dry-run: show everything
    else:
        matches = [p for p in all_runs if _matches(p, args)]

    print(f"TB dir: {args.dir}")
    if not matches:
        print("No matching runs.")
        return

    total = sum(_dir_size_bytes(p) for p in matches)
    print(f"Matched {len(matches)} runs ({_human(total)}):")
    for p in matches:
        print(f"  {p.name:<40} {_human(_dir_size_bytes(p)):>8}")

    if args.dry_run:
        return

    if not args.yes:
        confirm = input(f"\nDelete these {len(matches)} dirs? Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    deleted = 0
    for p in matches:
        try:
            shutil.rmtree(p)
            deleted += 1
        except OSError as exc:
            print(f"  [error] {p.name}: {exc}")
    print(f"Deleted {deleted} dirs ({_human(total)}).")


if __name__ == "__main__":
    main()
