"""
Visual demo — runs the AGV fleet environment with a selected agent
and renders it in real time using Pygame.

With no arguments a graphical launch menu is shown to configure the run.

Usage:
    python run_visual.py                             # show launch menu
    python run_visual.py --agent astar               # greedy A* baseline
    python run_visual.py --agent random              # random actions
    python run_visual.py --agent ppo --model <path>  # trained PPO model
    python run_visual.py --fps 20                    # faster simulation
    python run_visual.py --episodes 5                # stop after 5 episodes
    python run_visual.py --no-log                    # disable CSV logging

Controls (in window):
    SPACE       pause / resume
    UP / DOWN   increase / decrease speed
    R           reset episode manually
    ESC / Q     quit
"""

import argparse
import sys
import time
from typing import Optional

from src.env import AGVFleetEnv
from src.agents import AStarAgent, PPOAgent
from src.rendering import PygameRenderer, LaunchMenu
from src.logging import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGV Fleet visual demo")
    parser.add_argument("--agent",    choices=["astar", "random", "ppo"], default=None)
    parser.add_argument("--model",    type=str,  default=None,
                        help="Path to trained PPO model (required if --agent ppo)")
    parser.add_argument("--fps",      type=int,  default=10)
    parser.add_argument("--episodes", type=int,  default=0,   help="0 = run indefinitely")
    parser.add_argument("--n_agvs",   type=int,  default=None)
    parser.add_argument("--steps",    type=int,  default=500)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--no-log",   action="store_true", help="Disable CSV metrics logging")
    parser.add_argument(
        "--renderer", choices=["pygame", "coppeliasim", "both"], default=None,
        help="Visualisation backend (default: shown in launch menu)",
    )
    parser.add_argument("--interp_steps", type=int,   default=8,    help="Smooth movement sub-frames for CoppeliaSim")
    parser.add_argument("--delay",        type=float, default=0.08, help="Seconds per step (CoppeliaSim-only mode)")
    parser.add_argument("--cs_host",      type=str,   default="localhost")
    parser.add_argument("--cs_port",      type=int,   default=23000)
    return parser.parse_args()


def _build_bridge(args):
    """Create a CoppeliaSimBridge if the selected renderer requires it."""
    if args.renderer not in ("coppeliasim", "both"):
        return None
    try:
        from src.coppeliasim.bridge import CoppeliaSimBridge
        return CoppeliaSimBridge(n_agvs=args.n_agvs, host=args.cs_host, port=args.cs_port)
    except Exception as exc:
        print(f"[bridge] Connection failed: {exc}")
        print("  Make sure CoppeliaSim is running with the plant scene loaded.")
        sys.exit(1)


def _show_menu() -> dict:
    """Open the graphical launch menu and return the selected config."""
    menu   = LaunchMenu()
    config = menu.run()
    menu.close()
    if config is None:
        sys.exit(0)
    return config


def build_agent(args, env: AGVFleetEnv):
    if args.agent == "astar":
        return AStarAgent()
    if args.agent == "ppo":
        if args.model is None:
            print("ERROR: --model <path> is required when using --agent ppo")
            sys.exit(1)
        return PPOAgent.load(args.model, env)
    return None   # random


def _select_action(env, agent):
    return agent.select_action(env) if agent else env.action_space.sample()


def _log_episode(episode: int, total_reward: float, info: dict) -> None:
    print(
        f"Episode {episode:>3} | "
        f"steps={info['step']} | "
        f"reward={total_reward:+.1f} | "
        f"tasks={info['tasks_completed']} | "
        f"by_stage={info['tasks_by_stage']} | "
        f"collisions={info['collisions']} | "
        f"util={info['mean_utilization']:.1%}"
    )


def _reset(env, agent, renderer, bridge=None) -> None:
    env.reset()
    if agent:
        agent.reset()
    renderer.reset_animation(env)
    if bridge:
        bridge.sync(env.agvs)
        bridge.sync_tasks(env.tasks)


def _advance_episode(env, agent, renderer, bridge=None) -> tuple:
    """Take one sim step. Returns (reward, done, info)."""
    renderer.pre_step(env)
    action = _select_action(env, agent)
    _, reward, terminated, truncated, info = env.step(action)
    renderer.notify_step(env)
    if bridge:
        bridge.sync_step(env.agvs, env.tasks)
    return reward, terminated or truncated, info


def _finish_episode(env, agent, renderer, logger, episode, total_reward, info,
                    bridge=None) -> None:
    """Log, persist and reset after an episode ends."""
    _log_episode(episode, total_reward, info)
    if logger:
        logger.log_episode(episode, total_reward, info)
    _reset(env, agent, renderer, bridge)
    renderer.episode += 1


def _tick(env, agent, renderer, logger, max_eps, done, total_reward, info,
          bridge=None):
    """Process one sim tick. Returns updated (done, total_reward, info, stop)."""
    if not renderer.should_step():
        return done, total_reward, info, False
    if done:
        _finish_episode(env, agent, renderer, logger,
                        renderer.episode - 1, total_reward, info, bridge)
        stop = max_eps > 0 and renderer.episode > max_eps
        return False, 0.0, info, stop
    reward, done, info = _advance_episode(env, agent, renderer, bridge)
    return done, total_reward + reward, info, False


def _run_loop(env, agent, renderer, logger, max_eps: int,
              bridge=None) -> None:
    """Pygame-driven loop. When bridge is set, CoppeliaSim syncs each frame."""
    renderer.episode = 1
    _reset(env, agent, renderer, bridge)

    done         = False
    info         = {}
    total_reward = 0.0

    while True:
        signal = renderer.handle_events()
        if signal == "quit":
            break
        if signal == "reset":
            _reset(env, agent, renderer, bridge)
            renderer.episode += 1
            done, total_reward = False, 0.0

        if not renderer.paused:
            done, total_reward, info, stop = _tick(
                env, agent, renderer, logger, max_eps, done, total_reward, info, bridge)
            if stop:
                break

        # Sync CoppeliaSim position at the same interpolation t as Pygame
        if bridge:
            bridge.sync_frame(env.agvs, renderer._anim_t)

        renderer.render(env)
        renderer.tick()


def _run_coppeliasim_loop(env, agent, bridge, logger, max_eps: int,
                          interp_steps: int, delay: float) -> None:
    """CoppeliaSim-only loop — no Pygame, paced by time.sleep."""
    episode      = 1
    total_reward = 0.0
    step_delay   = delay / max(interp_steps, 1)

    env.reset()
    bridge.sync(env.agvs)
    bridge.sync_tasks(env.tasks)

    while True:
        action = _select_action(env, agent)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        bridge.sync_tasks(env.tasks)
        bridge.sync(env.agvs, interp_steps=interp_steps, step_delay=step_delay)

        if terminated or truncated:
            _log_episode(episode, total_reward, info)
            if logger:
                logger.log_episode(episode, total_reward, info)
            episode += 1
            if max_eps > 0 and episode > max_eps:
                break
            env.reset()
            total_reward = 0.0
            bridge.sync(env.agvs)
            bridge.sync_tasks(env.tasks)


def _resolve_args(args: argparse.Namespace) -> None:
    """Populate args from the launch menu when no CLI flags were given,
    or apply safe defaults when running headless."""
    if args.agent is None and args.n_agvs is None:
        config        = _show_menu()
        args.agent    = config["agent"]
        args.model    = config["model"]
        args.n_agvs   = config["n_agvs"]
        args.episodes = config["episodes"]
        if args.renderer is None:
            args.renderer = config.get("renderer", "pygame")
        return
    if args.agent    is None: args.agent    = "astar"
    if args.n_agvs   is None: args.n_agvs   = 4
    if args.renderer is None: args.renderer = "pygame"


def _build_logger(args: argparse.Namespace) -> Optional[MetricsLogger]:
    if args.no_log:
        return None
    logger = MetricsLogger(agent=args.agent, n_agvs=args.n_agvs, seed=args.seed)
    print(f"Logging    : {logger.path}")
    return logger


def main() -> None:
    args = parse_args()
    _resolve_args(args)

    env = AGVFleetEnv(
        n_agvs=args.n_agvs,
        n_tasks_max=8,
        max_steps=args.steps,
        seed=args.seed,
    )
    agent = build_agent(args, env)
    agent_label = {
        "astar":  "A*+Greedy",
        "ppo":    f"PPO ({args.model})",
        "random": "Random",
    }[args.agent]

    logger = _build_logger(args)

    print(f"Agent      : {agent_label}")
    print(f"Renderer   : {args.renderer}")
    print(f"Episodes   : {'infinite' if args.episodes == 0 else args.episodes}")

    # ── CoppeliaSim-only: no Pygame ──────────────────────────────────────
    if args.renderer == "coppeliasim":
        bridge = _build_bridge(args)
        try:
            _run_coppeliasim_loop(
                env, agent, bridge, logger,
                max_eps=args.episodes,
                interp_steps=args.interp_steps,
                delay=args.delay,
            )
        finally:
            bridge.close()
            if logger:
                logger.close()
            env.close()
        return

    # ── Pygame (alone or with CoppeliaSim) ───────────────────────────────
    bridge = _build_bridge(args)   # None when renderer == "pygame"

    renderer = PygameRenderer(
        title=f"AGV Fleet — {agent_label}",
        fps=args.fps,
        agent_label=agent_label,
    )
    print(f"Sim speed  : {args.fps} steps/s  (UP/DOWN to change)")
    print("Controls   : SPACE=pause  UP/DOWN=speed  R=reset  ESC/Q=quit\n")

    try:
        _run_loop(env, agent, renderer, logger=logger,
                  max_eps=args.episodes, bridge=bridge)
    finally:
        if bridge:
            bridge.close()
        if logger:
            logger.close()
        env.close()
        renderer.close()


if __name__ == "__main__":
    main()
