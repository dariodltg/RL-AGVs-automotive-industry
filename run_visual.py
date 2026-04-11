"""
Visual demo — runs the AGV fleet environment with a selected agent
and renders it in real time using Pygame.

Usage:
    python run_visual.py                             # AStarAgent, 10 fps
    python run_visual.py --agent astar               # greedy A* baseline
    python run_visual.py --agent random              # random actions
    python run_visual.py --agent ppo --model <path>  # trained PPO model
    python run_visual.py --fps 20                    # faster simulation
    python run_visual.py --episodes 5                # stop after 5 episodes

Controls (in window):
    SPACE       pause / resume
    UP / DOWN   increase / decrease speed
    R           reset episode manually
    ESC / Q     quit
"""

import argparse
import sys

import pygame

from src.env import AGVFleetEnv
from src.agents import AStarAgent, PPOAgent
from src.rendering import PygameRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGV Fleet visual demo")
    parser.add_argument("--agent",    choices=["astar", "random", "ppo"], default="astar")
    parser.add_argument("--model",    type=str,  default=None,
                        help="Path to trained PPO model (required if --agent ppo)")
    parser.add_argument("--fps",      type=int,  default=10)
    parser.add_argument("--episodes", type=int,  default=0,   help="0 = run indefinitely")
    parser.add_argument("--n_agvs",   type=int,  default=4)
    parser.add_argument("--steps",    type=int,  default=500)
    parser.add_argument("--seed",     type=int,  default=42)
    return parser.parse_args()


def build_agent(args, env: AGVFleetEnv):
    if args.agent == "astar":
        return AStarAgent()
    if args.agent == "ppo":
        if args.model is None:
            print("ERROR: --model <path> is required when using --agent ppo")
            sys.exit(1)
        return PPOAgent.load(args.model, env)
    return None   # random


def _reset_episode(env, agent):
    env.reset()
    if agent:
        agent.reset()


def _log_episode(episode: int, info: dict) -> None:
    print(
        f"Episode {episode:>3} | "
        f"steps={info['step']} | "
        f"tasks={info['tasks_completed']} | "
        f"by_stage={info['tasks_by_stage']} | "
        f"collisions={info['collisions']} | "
        f"util={info['mean_utilization']:.1%}"
    )


def _update_fps(fps: int) -> int:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return min(fps + 1, 60)
    if keys[pygame.K_DOWN]:
        return max(fps - 1, 1)
    return fps


def _step(env, agent) -> tuple:
    """Execute one environment step and return (done, info)."""
    action = agent.select_action(env) if agent else env.action_space.sample()
    _, _, terminated, truncated, info = env.step(action)
    return terminated or truncated, info


def _handle_episode_end(env, agent, episode: int, info: dict, max_eps: int) -> tuple:
    """Log episode, reset, return (should_quit, new_episode)."""
    _log_episode(episode, info)
    if max_eps > 0 and episode >= max_eps:
        return True, episode
    _reset_episode(env, agent)
    return False, episode + 1


def _run_loop(env, agent, renderer, fps: int, max_eps: int) -> None:
    """Main simulation loop. Runs until quit signal or max_eps reached."""
    episode = 1
    _reset_episode(env, agent)

    while True:
        signal = renderer.handle_events()
        if signal == "quit":
            break
        if signal == "reset":
            _reset_episode(env, agent)
            episode += 1

        fps = _update_fps(fps)

        if not renderer.paused:
            done, info = _step(env, agent)
            if done:
                quit_loop, episode = _handle_episode_end(env, agent, episode, info, max_eps)
                if quit_loop:
                    break

        renderer.render(env)
        renderer.tick(fps)


def main() -> None:
    args = parse_args()

    env = AGVFleetEnv(
        n_agvs=args.n_agvs,
        n_tasks_max=8,
        max_steps=args.steps,
        seed=args.seed,
    )
    agent = build_agent(args, env)
    agent_label = {"astar": "A*+Greedy", "ppo": f"PPO ({args.model})", "random": "Random"}[args.agent]
    renderer = PygameRenderer(title=f"AGV Fleet — {agent_label}")

    print(f"Agent    : {agent_label}")
    print(f"FPS      : {args.fps}  (UP/DOWN to change in window)")
    print(f"Episodes : {'infinite' if args.episodes == 0 else args.episodes}")
    print("Controls : SPACE=pause  UP/DOWN=speed  R=reset  ESC/Q=quit\n")

    _run_loop(env, agent, renderer, fps=args.fps, max_eps=args.episodes)

    env.close()
    renderer.close()


if __name__ == "__main__":
    main()
