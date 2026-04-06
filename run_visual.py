"""
Visual demo — runs the AGV fleet environment with a selected agent
and renders it in real time using Pygame.

Usage:
    python run_visual.py                  # default: AStarAgent, 10 fps
    python run_visual.py --agent astar    # greedy A* baseline
    python run_visual.py --agent random   # random actions
    python run_visual.py --fps 20         # faster simulation
    python run_visual.py --episodes 5     # run 5 episodes then exit

Controls (in window):
    SPACE       pause / resume
    UP / DOWN   increase / decrease speed
    R           reset episode manually
    ESC / Q     quit
"""

import argparse

import numpy as np

from src.env import AGVFleetEnv
from src.agents import AStarAgent
from src.rendering import PygameRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGV Fleet visual demo")
    parser.add_argument("--agent",    choices=["astar", "random"], default="astar")
    parser.add_argument("--fps",      type=int,   default=10)
    parser.add_argument("--episodes", type=int,   default=0,  help="0 = run indefinitely")
    parser.add_argument("--n_agvs",   type=int,   default=4)
    parser.add_argument("--steps",    type=int,   default=500)
    parser.add_argument("--seed",     type=int,   default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = AGVFleetEnv(
        n_agvs=args.n_agvs,
        n_tasks_max=8,
        max_steps=args.steps,
        seed=args.seed,
    )

    agent = AStarAgent() if args.agent == "astar" else None
    renderer = PygameRenderer(
        title=f"AGV Fleet — {'A*+Greedy' if agent else 'Random'}"
    )

    fps       = args.fps
    episode   = 0
    max_eps   = args.episodes  # 0 = infinite

    print(f"Agent    : {'AStarAgent (greedy)' if agent else 'Random'}")
    print(f"FPS      : {fps}  (UP/DOWN to change in window)")
    print(f"Episodes : {'infinite' if max_eps == 0 else max_eps}")
    print("Window controls: SPACE=pause  UP/DOWN=speed  R=reset  ESC/Q=quit\n")

    obs, info = env.reset()
    if agent:
        agent.reset()
    episode += 1

    while True:
        # --- event handling ---
        signal = renderer.handle_events()
        if signal == "quit":
            break
        if signal == "reset":
            obs, info = env.reset()
            if agent:
                agent.reset()
            episode += 1

        # --- speed control via keyboard held ---
        import pygame
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            fps = min(fps + 1, 60)
        if keys[pygame.K_DOWN]:
            fps = max(fps - 1, 1)

        # --- step ---
        if not renderer.paused:
            if agent:
                action = agent.select_action(env)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(
                    f"Episode {episode:>3} done | "
                    f"steps={info['step']} | "
                    f"tasks={info['tasks_completed']} | "
                    f"collisions={info['collisions']} | "
                    f"util={info['mean_utilization']:.1%}"
                )
                if max_eps > 0 and episode >= max_eps:
                    break
                obs, info = env.reset()
                if agent:
                    agent.reset()
                episode += 1

        # --- render ---
        renderer.render(env)
        renderer.tick(fps)

    env.close()
    renderer.close()


if __name__ == "__main__":
    main()
