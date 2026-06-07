"""
Clean up rows from the experiment registry (experiments.db).

Always shows a summary of what would be deleted and asks for confirmation
unless --yes is passed. Use --dry-run to preview without deleting.

Usage
-----
    # Show what's in the DB without touching anything
    python scripts/clean_registry.py --dry-run

    # Delete every run (asks for confirmation)
    python scripts/clean_registry.py --all

    # Delete only interrupted runs
    python scripts/clean_registry.py --status interrupted

    # Delete all PPO runs
    python scripts/clean_registry.py --agent ppo

    # Delete one specific run, no prompt
    python scripts/clean_registry.py --run-id ppo_seed0_20260531_132236 --yes

    # Delete runs started before a date (UTC, YYYY-MM-DD)
    python scripts/clean_registry.py --before 2026-05-01

Filters combine with AND. Both episodes and experiments rows are removed.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same default location as src/experiments/registry.py
_DB_PATH = Path(r"C:\agv-experiments\experiments.db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean rows from experiments.db")
    parser.add_argument("--db", type=Path, default=_DB_PATH,
                        help=f"Path to experiments.db (default: {_DB_PATH})")
    parser.add_argument("--all",    action="store_true", help="Delete every run")
    parser.add_argument("--agent",  type=str, default=None,
                        choices=["ppo", "astar", "random"],
                        help="Only delete runs by this agent")
    parser.add_argument("--status", type=str, default=None,
                        choices=["running", "completed", "interrupted"],
                        help="Only delete runs with this status")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Delete one specific run by id")
    parser.add_argument("--before", type=str, default=None,
                        help="Delete runs started before this date (YYYY-MM-DD, UTC)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted and exit")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the confirmation prompt")
    return parser.parse_args()


def _build_where(args: argparse.Namespace) -> Tuple[str, List]:
    clauses: List[str] = []
    params:  List      = []
    if args.agent:
        clauses.append("agent = ?");  params.append(args.agent)
    if args.status:
        clauses.append("status = ?"); params.append(args.status)
    if args.run_id:
        clauses.append("run_id = ?"); params.append(args.run_id)
    if args.before:
        # started_at is stored as "YYYYMMDD_HHMMSS"
        clauses.append("started_at < ?")
        params.append(args.before.replace("-", "") + "_000000")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _preview(conn: sqlite3.Connection, where: str, params: List) -> int:
    rows = conn.execute(
        f"SELECT agent, status, COUNT(*) AS n "
        f"FROM experiments{where} GROUP BY agent, status ORDER BY agent, status",
        params,
    ).fetchall()
    total_runs = sum(r[2] for r in rows)
    if total_runs == 0:
        print("No matching runs.")
        return 0

    n_eps = conn.execute(
        f"SELECT COUNT(*) FROM episodes "
        f"WHERE run_id IN (SELECT run_id FROM experiments{where})",
        params,
    ).fetchone()[0]

    print(f"Matched {total_runs} runs ({n_eps} episode rows):")
    for agent, status, n in rows:
        print(f"  {agent:<8} {status:<12} {n:>4}")
    return total_runs


def main() -> None:
    args = parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        sys.exit(1)

    if not (args.all or args.agent or args.status or args.run_id or args.before
            or args.dry_run):
        print("Refusing to delete: pass a filter (--all, --agent, --status, "
              "--run-id, --before) or --dry-run.")
        sys.exit(2)

    where, params = ("", []) if args.all else _build_where(args)
    if not args.all and not (args.agent or args.status or args.run_id or args.before):
        # Pure --dry-run: just summarise everything
        where, params = "", []

    conn = sqlite3.connect(str(args.db))
    try:
        print(f"DB: {args.db}")
        matched = _preview(conn, where, params)
        if matched == 0 or args.dry_run:
            return

        if not args.yes:
            confirm = input(f"\nDelete these {matched} runs? Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                return

        # Delete episodes first (foreign key references experiments)
        conn.execute(
            f"DELETE FROM episodes "
            f"WHERE run_id IN (SELECT run_id FROM experiments{where})",
            params,
        )
        conn.execute(f"DELETE FROM experiments{where}", params)
        conn.commit()
        print(f"Deleted {matched} runs.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
