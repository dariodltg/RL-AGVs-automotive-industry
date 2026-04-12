"""
Episode metrics logger — writes one CSV row per episode.

Usage:
    logger = MetricsLogger(agent="astar", n_agvs=4, seed=42)

    # at the end of each episode:
    logger.log_episode(episode=1, total_reward=45.3, info=info)

    logger.close()   # flushes and closes the file

Output: runs/{agent}_{n_agvs}agvs_{timestamp}/metrics.csv
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


_RUNS_DIR = Path(__file__).parent.parent.parent / "runs"

_FIELDNAMES = [
    "run_id",
    "agent",
    "n_agvs",
    "seed",
    "episode",
    "steps",
    "total_reward",
    "tasks_completed",
    "tasks_s0",
    "tasks_s1",
    "tasks_s2",
    "tasks_s3",
    "collisions",
    "avg_cycle_time",
    "mean_utilization",
]


class MetricsLogger:
    """
    Writes per-episode metrics to a CSV file under runs/.

    Parameters
    ----------
    agent : str
        Agent identifier (e.g. "astar", "ppo", "random").
    n_agvs : int
        Number of AGVs in the fleet.
    seed : int
        Environment seed used for the run.
    run_id : str, optional
        Custom run identifier. Defaults to "{agent}_{n_agvs}agvs_{timestamp}".
    runs_dir : Path, optional
        Root directory for run output. Defaults to runs/ at project root.
    """

    def __init__(
        self,
        agent:    str,
        n_agvs:   int,
        seed:     int,
        run_id:   Optional[str] = None,
        runs_dir: Optional[Path] = None,
    ) -> None:
        self.agent   = agent
        self.n_agvs  = n_agvs
        self.seed    = seed

        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id  = run_id or f"{agent}_{n_agvs}agvs_{timestamp}"

        out_dir = (runs_dir or _RUNS_DIR) / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / "metrics.csv"

        self._file   = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    # ------------------------------------------------------------------

    def log_episode(
        self,
        episode:      int,
        total_reward: float,
        info:         Dict[str, Any],
    ) -> None:
        """
        Write one row for a completed episode.

        Parameters
        ----------
        episode : int
            Episode number (1-based).
        total_reward : float
            Cumulative reward accumulated during the episode.
        info : dict
            The info dict returned by env.step() at episode end.
        """
        by_stage = info.get("tasks_by_stage", [0, 0, 0, 0])

        self._writer.writerow({
            "run_id":           self.run_id,
            "agent":            self.agent,
            "n_agvs":           self.n_agvs,
            "seed":             self.seed,
            "episode":          episode,
            "steps":            info.get("step", 0),
            "total_reward":     round(total_reward, 4),
            "tasks_completed":  info.get("tasks_completed", 0),
            "tasks_s0":         by_stage[0] if len(by_stage) > 0 else 0,
            "tasks_s1":         by_stage[1] if len(by_stage) > 1 else 0,
            "tasks_s2":         by_stage[2] if len(by_stage) > 2 else 0,
            "tasks_s3":         by_stage[3] if len(by_stage) > 3 else 0,
            "collisions":       info.get("collisions", 0),
            "avg_cycle_time":   round(info.get("avg_cycle_time", 0.0), 2),
            "mean_utilization": round(info.get("mean_utilization", 0.0), 4),
        })
        self._file.flush()

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the CSV file being written."""
        return self._path
