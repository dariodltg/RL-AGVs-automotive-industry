"""
PPO agent wrapper for AGV fleet management.

Wraps Stable-Baselines3 PPO to implement the BaseAgent interface,
allowing fair comparison with the AStarAgent baseline using the same
select_action(env) -> np.ndarray protocol.

Training is done via train(), which delegates to SB3's model.learn().
Inference uses model.predict(obs, deterministic=True) so the policy
is fully deterministic at evaluation time.

Typical usage:
    # Training
    env   = AGVFleetEnv(...)
    agent = PPOAgent(env)
    agent.train(total_timesteps=500_000, progress_bar=True)
    agent.save("models/ppo_fleet")

    # Inference
    agent = PPOAgent.load("models/ppo_fleet", env)
    obs, _ = env.reset()
    action = agent.select_action(env)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .base_agent import BaseAgent

if TYPE_CHECKING:
    from src.env.agv_fleet_env import AGVFleetEnv


# Default PPO hyperparameters tuned for the AGV fleet task.
# These are reasonable starting values; adjust via ppo_kwargs in __init__.
_DEFAULT_PPO_KWARGS: Dict[str, Any] = {
    "policy":        "MlpPolicy",
    "n_steps":       2048,        # steps per env before each update
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,        # discount factor
    "learning_rate": 3e-4,
    "ent_coef":      0.01,        # entropy bonus (encourages exploration)
    "clip_range":    0.2,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "verbose":       1,
}


class PPOAgent(BaseAgent):
    """
    PPO-based AGV fleet management agent using Stable-Baselines3.

    Parameters
    ----------
    env : AGVFleetEnv
        Training environment. Wrapped with SB3 Monitor for episode logging.
    tensorboard_log : Optional[str]
        Directory for TensorBoard logs. None disables logging.
    **ppo_kwargs
        Additional keyword arguments passed to stable_baselines3.PPO.
        These override the defaults in _DEFAULT_PPO_KWARGS.
    """

    def __init__(
        self,
        env: "AGVFleetEnv",
        tensorboard_log: Optional[str] = "logs/tensorboard",
        **ppo_kwargs,
    ):
        kwargs = {**_DEFAULT_PPO_KWARGS, **ppo_kwargs}
        policy = kwargs.pop("policy")

        monitored_env = Monitor(env)
        self.model = PPO(
            policy,
            monitored_env,
            tensorboard_log=tensorboard_log,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, env: "AGVFleetEnv") -> np.ndarray:
        """
        Returns a deterministic action for the current environment state.
        Reads the observation directly from env to match the AStarAgent protocol.
        """
        obs = env._get_observation()
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def reset(self) -> None:
        pass  # SB3 manages internal LSTM / hidden states if needed

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 500_000,
        eval_env: Optional["AGVFleetEnv"] = None,
        eval_freq: int = 10_000,
        save_path: str = "models",
        run_name: str = "ppo_fleet",
        progress_bar: bool = True,
    ) -> None:
        """
        Train the PPO policy.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps for training.
        eval_env : Optional[AGVFleetEnv]
            Separate environment for periodic evaluation. If None,
            no EvalCallback is used.
        eval_freq : int
            Evaluate (and potentially save best model) every N steps.
        save_path : str
            Directory where model checkpoints are saved.
        run_name : str
            Base filename for saved models.
        progress_bar : bool
            Show SB3 progress bar during training.
        """
        os.makedirs(save_path, exist_ok=True)
        callbacks = []

        if eval_env is not None:
            best_model_path = os.path.join(save_path, f"{run_name}_best")
            eval_callback = EvalCallback(
                Monitor(eval_env),
                best_model_save_path=best_model_path,
                log_path=os.path.join(save_path, "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                verbose=1,
            )
            callbacks.append(eval_callback)

        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq,
            save_path=os.path.join(save_path, "checkpoints"),
            name_prefix=run_name,
            verbose=0,
        )
        callbacks.append(checkpoint_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks) if callbacks else None,
            progress_bar=False,   # requires tqdm[extra]; disabled for compatibility
            reset_num_timesteps=True,
        )

        # Save final model
        final_path = os.path.join(save_path, run_name)
        self.model.save(final_path)
        print(f"Training complete. Model saved to {final_path}.zip")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the model to disk. Extension .zip is added automatically."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Optional["AGVFleetEnv"] = None) -> "PPOAgent":
        """
        Load a saved PPO model.

        Parameters
        ----------
        path : str
            Path to the saved model (.zip, extension optional).
        env : Optional[AGVFleetEnv]
            Environment to attach to the loaded model. Required if
            you intend to continue training; optional for inference only.
        """
        instance = cls.__new__(cls)
        instance.model = PPO.load(path, env=Monitor(env) if env is not None else None)
        return instance
