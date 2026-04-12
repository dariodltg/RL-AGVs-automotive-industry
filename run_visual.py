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
    return parser.parse_args()


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


def _reset(env, agent, renderer) -> None:
    env.reset()
    if agent:
        agent.reset()
    renderer.reset_animation(env)


def _advance_episode(env, agent, renderer) -> tuple:
    """Take one sim step. Returns (reward, done, info)."""
    renderer.pre_step(env)
    action = _select_action(env, agent)
    _, reward, terminated, truncated, info = env.step(action)
    renderer.notify_step(env)
    return reward, terminated or truncated, info


def _finish_episode(env, agent, renderer, logger, episode, total_reward, info) -> None:
    """Log, persist and reset after an episode ends."""
    _log_episode(episode, total_reward, info)
    if logger:
        logger.log_episode(episode, total_reward, info)
    _reset(env, agent, renderer)
    renderer.episode += 1


def _tick(env, agent, renderer, logger, max_eps, done, total_reward, info):
    """Process one sim tick. Returns updated (done, total_reward, info, stop)."""
    if not renderer.should_step():
        return done, total_reward, info, False
    if done:
        _finish_episode(env, agent, renderer, logger,
                        renderer.episode - 1, total_reward, info)
        stop = max_eps > 0 and renderer.episode > max_eps
        return False, 0.0, info, stop
    reward, done, info = _advance_episode(env, agent, renderer)
    return done, total_reward + reward, info, False


def _run_loop(env, agent, renderer, logger, max_eps: int) -> None:
    renderer.episode = 1
    _reset(env, agent, renderer)

    done         = False
    info         = {}
    total_reward = 0.0

    while True:
        signal = renderer.handle_events()
        if signal == "quit":
            break
        if signal == "reset":
            _reset(env, agent, renderer)
            renderer.episode += 1
            done, total_reward = False, 0.0

        if not renderer.paused:
            done, total_reward, info, stop = _tick(
                env, agent, renderer, logger, max_eps, done, total_reward, info)
            if stop:
                break

        renderer.render(env)
        renderer.tick()


def main() -> None:
    args = parse_args()

    # Show the graphical menu only when launched without arguments
    if args.agent is None and args.n_agvs is None:
        config         = _show_menu()
        args.agent     = config["agent"]
        args.model     = config["model"]
        args.n_agvs    = config["n_agvs"]
        args.episodes  = config["episodes"]
    else:
        if args.agent is None:
            args.agent = "astar"
        if args.n_agvs is None:
            args.n_agvs = 4

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

    renderer = PygameRenderer(
        title=f"AGV Fleet — {agent_label}",
        fps=args.fps,
        agent_label=agent_label,
    )

    logger = None
    if not args.no_log:
        logger = MetricsLogger(agent=args.agent, n_agvs=args.n_agvs, seed=args.seed)
        print(f"Logging  : {logger.path}")

    print(f"Agent    : {agent_label}")
    print(f"Sim speed: {args.fps} steps/s  (UP/DOWN to change)")
    print(f"Episodes : {'infinite' if args.episodes == 0 else args.episodes}")
    print("Controls : SPACE=pause  UP/DOWN=speed  R=reset  ESC/Q=quit\n")

    _run_loop(env, agent, renderer, logger=logger, max_eps=args.episodes)

    if logger:
        logger.close()
    env.close()
    renderer.close()


if __name__ == "__main__":
    main()
