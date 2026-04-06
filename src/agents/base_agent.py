"""
Abstract base class for AGV fleet management agents.

Both the classical baseline (AStarAgent) and the learned policy (PPOAgent)
implement this interface, so experiments can swap agents without touching
the environment or evaluation code.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.env.agv_fleet_env import AGVFleetEnv


class BaseAgent(ABC):
    """
    Agent interface for AGV fleet task assignment.

    select_action() receives the live environment and returns an action
    array compatible with AGVFleetEnv.action_space (MultiDiscrete).

    Classical agents (AStarAgent) read env state directly.
    Learned agents (PPOAgent) use the flattened observation vector.
    Both produce the same action format, enabling fair comparison.
    """

    @abstractmethod
    def select_action(self, env: "AGVFleetEnv") -> np.ndarray:
        """
        Choose a task assignment action for the current environment state.

        Parameters
        ----------
        env : AGVFleetEnv
            The live environment instance. Classical agents may inspect
            env.agvs and env.tasks directly; learned agents use env
            only to obtain the observation via _get_observation().

        Returns
        -------
        np.ndarray
            Action array of shape (n_agvs,), dtype int64.
            Value 0 means wait; value i > 0 means assign task at index i-1.
        """
        ...

    def reset(self) -> None:
        """
        Called at the start of each episode.
        Override if the agent holds episode-level state (e.g. PPO hidden states).
        """
