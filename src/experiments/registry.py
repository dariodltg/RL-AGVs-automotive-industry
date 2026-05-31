"""
SQLite-backed experiment registry for evaluation runs.

Tables
------
experiments
    run_id            TEXT  PRIMARY KEY
    agent             TEXT  e.g. "ppo", "astar", "random"
    seed              INT
    n_agvs            INT
    n_tasks_max       INT
    max_steps         INT
    task_arrival_rate REAL
    hyperparameters   TEXT  JSON blob (PPO params); NULL for baselines
    status            TEXT  "running" | "completed" | "interrupted"
    started_at        TEXT  UTC timestamp
    finished_at       TEXT  UTC timestamp; NULL while running

episodes
    run_id            TEXT  FK → experiments.run_id
    episode           INT   1-based
    steps             INT
    total_reward      REAL
    tasks_completed   INT
    tasks_s0–s3       INT   tasks delivered per pipeline stage
    collisions        INT
    avg_cycle_time    REAL
    mean_utilization  REAL

Usage
-----
    with ExperimentRegistry() as registry:
        run_id = registry.start_run(
            agent="astar", seed=42, n_agvs=4,
            n_tasks_max=8, max_steps=500, task_arrival_rate=0.15)
        for ep in range(1, episodes + 1):
            registry.log_episode(run_id, ep, total_reward, info)
        registry.finish_run(run_id)

An interrupted run (exception inside the context) is marked "interrupted"
automatically so partial data is visible but excluded from aggregate queries.

Grafana
-------
Connect with the frser-sqlite-datasource plugin pointing to experiments.db
(read-only mode). See docs/grafana_setup.md for SQL queries and dashboard JSON.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

_DB_PATH = Path(r"C:\agv-experiments\experiments.db")

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    run_id            TEXT PRIMARY KEY,
    agent             TEXT NOT NULL,
    seed              INTEGER NOT NULL,
    n_agvs            INTEGER NOT NULL,
    n_tasks_max       INTEGER NOT NULL,
    max_steps         INTEGER NOT NULL,
    task_arrival_rate REAL NOT NULL,
    hyperparameters   TEXT,
    status            TEXT NOT NULL DEFAULT 'running',
    started_at        TEXT NOT NULL,
    finished_at       TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    run_id           TEXT    NOT NULL REFERENCES experiments(run_id),
    episode          INTEGER NOT NULL,
    steps            INTEGER NOT NULL DEFAULT 0,
    total_reward     REAL    NOT NULL,
    tasks_completed  INTEGER NOT NULL DEFAULT 0,
    tasks_s0         INTEGER NOT NULL DEFAULT 0,
    tasks_s1         INTEGER NOT NULL DEFAULT 0,
    tasks_s2         INTEGER NOT NULL DEFAULT 0,
    tasks_s3         INTEGER NOT NULL DEFAULT 0,
    collisions       INTEGER NOT NULL DEFAULT 0,
    avg_cycle_time   REAL    NOT NULL DEFAULT 0.0,
    mean_utilization REAL    NOT NULL DEFAULT 0.0,
    PRIMARY KEY (run_id, episode)
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class ExperimentRegistry:
    """
    Stores evaluation run metadata and per-episode metrics in SQLite.

    Parameters
    ----------
    db_path : Path, optional
        Path to the SQLite database file. Defaults to experiments.db at
        the project root. Created automatically on first use.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        path = db_path or _DB_PATH
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)
        self._conn.commit()
        self._active: Set[str] = set()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentRegistry":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            for run_id in self._active:
                self._set_status(run_id, "interrupted")
        self.close()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        agent: str,
        seed: int,
        n_agvs: int,
        n_tasks_max: int,
        max_steps: int,
        task_arrival_rate: float,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new evaluation run and return its run_id.

        Parameters
        ----------
        agent : str
            Agent identifier ("ppo", "astar", "random").
        seed : int
            Environment seed for this run.
        n_agvs : int
            Fleet size.
        n_tasks_max : int
            Maximum concurrent tasks in the environment.
        max_steps : int
            Episode length cap.
        task_arrival_rate : float
            Poisson rate used to spawn tasks.
        hyperparameters : dict, optional
            PPO hyperparameters dict. Pass None for baseline agents.

        Returns
        -------
        str
            Unique run_id to pass to log_episode() and finish_run().
        """
        ts = _utcnow()
        run_id = f"{agent}_seed{seed}_{ts}"
        self._conn.execute(
            """
            INSERT INTO experiments
              (run_id, agent, seed, n_agvs, n_tasks_max, max_steps,
               task_arrival_rate, hyperparameters, status, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                run_id, agent, seed, n_agvs, n_tasks_max, max_steps,
                task_arrival_rate,
                json.dumps(hyperparameters) if hyperparameters is not None else None,
                ts,
            ),
        )
        self._conn.commit()
        self._active.add(run_id)
        return run_id

    def finish_run(self, run_id: str) -> None:
        """Mark a run as completed."""
        self._set_status(run_id, "completed")
        self._active.discard(run_id)

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def log_episode(
        self,
        run_id: str,
        episode: int,
        total_reward: float,
        info: Dict[str, Any],
    ) -> None:
        """
        Record per-episode metrics.

        Parameters
        ----------
        run_id : str
            Identifier returned by start_run().
        episode : int
            Episode number (1-based).
        total_reward : float
            Cumulative reward for the episode.
        info : dict
            The info dict returned by env.step() at episode end.
        """
        by_stage = info.get("tasks_by_stage", [0, 0, 0, 0])
        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
              (run_id, episode, steps, total_reward, tasks_completed,
               tasks_s0, tasks_s1, tasks_s2, tasks_s3,
               collisions, avg_cycle_time, mean_utilization)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                episode,
                info.get("step", 0),
                round(total_reward, 4),
                info.get("tasks_completed", 0),
                by_stage[0] if len(by_stage) > 0 else 0,
                by_stage[1] if len(by_stage) > 1 else 0,
                by_stage[2] if len(by_stage) > 2 else 0,
                by_stage[3] if len(by_stage) > 3 else 0,
                info.get("collisions", 0),
                round(info.get("avg_cycle_time", 0.0), 2),
                round(info.get("mean_utilization", 0.0), 4),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a summary table of completed runs to stdout."""
        rows = self._conn.execute(
            """
            SELECT e.run_id, e.agent, e.seed,
                   ROUND(AVG(ep.total_reward), 2)    AS mean_reward,
                   ROUND(AVG(ep.tasks_completed), 2) AS mean_tasks,
                   ROUND(AVG(ep.collisions), 2)      AS mean_collisions,
                   ROUND(AVG(ep.mean_utilization), 4) AS mean_util,
                   COUNT(ep.episode)                  AS n_episodes
            FROM experiments e
            JOIN episodes ep USING (run_id)
            WHERE e.status = 'completed'
            GROUP BY e.run_id
            ORDER BY e.agent, e.seed
            """
        ).fetchall()

        if not rows:
            print("No completed runs in experiments.db")
            return

        header = (
            f"{'run_id':<38} {'agent':<8} {'seed':>4}  "
            f"{'reward':>8}  {'tasks':>6}  {'coll':>5}  {'util':>6}  {'eps':>4}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r[0]:<38} {r[1]:<8} {r[2]:>4}  "
                f"{r[3]:>8.2f}  {r[4]:>6.1f}  {r[5]:>5.1f}  {r[6]:>6.2%}  {r[7]:>4}"
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_status(self, run_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE experiments SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, _utcnow(), run_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
