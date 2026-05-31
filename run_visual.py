"""
Visual launcher for the AGV fleet environment.

With no arguments a graphical menu lets you choose:
  - Evaluation  Visual run with Pygame/CoppeliaSim; results logged to experiments.db.
  - Training    Headless PPO training; progress shown in a Pygame window.

CLI usage (skips the menu):
    python run_visual.py --agent astar
    python run_visual.py --agent ppo --model models/ppo_fleet.zip
    python run_visual.py --mode training --run_name exp01 --timesteps 500000

Controls (Pygame evaluation window):
    SPACE       pause / resume
    UP / DOWN   increase / decrease speed
    R           reset episode manually
    ESC / Q     quit
"""

import argparse
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import pygame

from src.env import AGVFleetEnv
from src.agents import AStarAgent, PPOAgent
from src.rendering import PygameRenderer, LaunchMenu

_LOAD_LEVELS = {"low": 0.08, "medium": 0.15, "high": 0.25}
_N_TASKS_MAX = 8
_MAX_STEPS   = 500


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AGV Fleet launcher")
    parser.add_argument("--mode",       choices=["evaluation", "training"], default=None)
    # Evaluation args
    parser.add_argument("--agent",      choices=["astar", "random", "ppo"], default=None)
    parser.add_argument("--model",      type=str,  default=None,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--fps",        type=int,  default=10)
    parser.add_argument("--episodes",   type=int,  default=0, help="0 = run indefinitely")
    parser.add_argument("--n_agvs",     type=int,  default=None)
    parser.add_argument("--steps",      type=int,  default=_MAX_STEPS)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument(
        "--renderer", choices=["pygame", "coppeliasim", "both"], default=None)
    parser.add_argument("--interp_steps", type=int,   default=8)
    parser.add_argument("--delay",        type=float, default=0.08)
    parser.add_argument("--cs_host",      type=str,   default="localhost")
    parser.add_argument("--cs_port",      type=int,   default=23000)
    # Training args
    parser.add_argument("--run_name",   type=str,  default="ppo_fleet")
    parser.add_argument("--timesteps",  type=int,  default=300_000)
    parser.add_argument("--load_level", choices=["low", "medium", "high"], default="medium")
    return parser.parse_args()


def _resolve_args(args: argparse.Namespace) -> None:
    """Apply safe defaults for CLI invocations."""
    if args.mode     is None: args.mode     = "evaluation"
    if args.agent    is None: args.agent    = "astar"
    if args.n_agvs   is None: args.n_agvs   = 4
    if args.renderer is None: args.renderer = "pygame"


def _show_menu() -> Optional[dict]:
    menu   = LaunchMenu()
    config = menu.run()
    menu.close()
    return config  # None means the user quit the menu


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_agent(args, env: AGVFleetEnv):
    if args.agent == "astar":
        return AStarAgent()
    if args.agent == "ppo":
        if args.model is None:
            print("ERROR: --model <path> is required when using --agent ppo")
            sys.exit(1)
        return PPOAgent.load(args.model, env)
    return None  # random


def _select_action(env, agent):
    return agent.select_action(env) if agent else env.action_space.sample()


def _build_bridge(args):
    if args.renderer not in ("coppeliasim", "both"):
        return None
    try:
        from src.coppeliasim.bridge import CoppeliaSimBridge
        return CoppeliaSimBridge(n_agvs=args.n_agvs, host=args.cs_host, port=args.cs_port)
    except Exception as exc:
        print(f"[bridge] Connection failed: {exc}")
        print("  Make sure CoppeliaSim is running with the plant scene loaded.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

def _train_ppo(args, rate: float, state: dict) -> None:
    """PPO training worker — runs in a background thread."""
    try:
        from stable_baselines3.common.callbacks import BaseCallback

        class _ProgressTracker(BaseCallback):
            def _on_step(self) -> bool:
                state["timestep"] = self.num_timesteps
                return True

        train_env = AGVFleetEnv(
            n_agvs=args.n_agvs, n_tasks_max=_N_TASKS_MAX,
            max_steps=_MAX_STEPS, task_arrival_rate=rate, seed=args.seed)
        eval_env = AGVFleetEnv(
            n_agvs=args.n_agvs, n_tasks_max=_N_TASKS_MAX,
            max_steps=_MAX_STEPS, task_arrival_rate=rate, seed=args.seed + 100)
        try:
            import tensorboard  # noqa: F401
            tb_log = "logs/tensorboard"
        except ImportError:
            tb_log = None
            print("[warning] tensorboard not importable — training without TB logging")
        agent = PPOAgent(train_env, tensorboard_log=tb_log)
        agent.train(
            total_timesteps=args.timesteps,
            eval_env=eval_env,
            eval_freq=10_000,
            save_path="models",
            run_name=args.run_name,
            progress_bar=False,
            extra_callbacks=[_ProgressTracker()],
        )
        sidecar = {
            "hyperparameters": {
                "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
                "n_epochs": 10, "gamma": 0.99, "ent_coef": 0.01,
                "clip_range": 0.2, "vf_coef": 0.5, "max_grad_norm": 0.5,
            },
            "env_config": {
                "n_agvs": args.n_agvs, "n_tasks_max": _N_TASKS_MAX,
                "max_steps": _MAX_STEPS, "load_level": args.load_level,
                "task_arrival_rate": rate,
            },
            "timesteps": args.timesteps,
            "seed":      args.seed,
        }
        Path("models").mkdir(exist_ok=True)
        sidecar_path = Path("models") / f"{args.run_name}.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        state["model_path"] = f"models/{args.run_name}.zip"
        train_env.close()
        eval_env.close()
        state["done"] = True
    except Exception as exc:
        traceback.print_exc()
        state["error"] = str(exc)


def _training_window_quit_requested() -> bool:
    """Return True if the user closed or pressed ESC in the progress window."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _render_training_status(screen, fonts, state: dict, args, elapsed: float) -> None:
    """Draw one frame of the training progress window."""
    f_lg, f_md, f_sm = fonts
    BG     = (20,  20,  28)
    ACCENT = (90,  160, 255)
    DIM    = (120, 120, 140)
    WHITE  = (230, 230, 230)
    GREEN  = (60,  200, 100)
    RED    = (220, 80,  80)

    screen.fill(BG)
    surf = f_lg.render("PPO Training", True, ACCENT)
    screen.blit(surf, surf.get_rect(centerx=260, top=18))

    y = 58
    for text, color in [
        (f"Run      :  {args.run_name}", WHITE),
        (f"Timesteps:  {args.timesteps:,}", DIM),
        (f"AGVs: {args.n_agvs}   Load: {args.load_level}   Seed: {args.seed}", DIM),
        (f"Elapsed  :  {elapsed:.0f}s", DIM),
    ]:
        screen.blit(f_md.render(text, True, color), (28, y))
        y += 26
    y += 10

    # Progress bar
    total_ts  = args.timesteps
    current   = state["timestep"] if state["done"] else state["timestep"]
    pct       = min(current / total_ts, 1.0) if total_ts > 0 else 0.0
    bar_x, bar_w, bar_h = 28, 464, 18
    pygame.draw.rect(screen, (35, 35, 48),   (bar_x, y, bar_w, bar_h), border_radius=4)
    if pct > 0:
        pygame.draw.rect(screen, ACCENT, (bar_x, y, int(bar_w * pct), bar_h), border_radius=4)
    pct_surf = f_sm.render(f"{pct:.0%}", True, WHITE)
    screen.blit(pct_surf, pct_surf.get_rect(centerx=bar_x + bar_w // 2, centery=y + bar_h // 2))
    y += bar_h + 6

    # Step counter + ETA
    step_txt = f"{current:,} / {total_ts:,} steps"
    if pct > 0.01 and elapsed > 1 and not state["done"] and not state["error"]:
        eta_s   = elapsed / pct * (1.0 - pct)
        minutes = int(eta_s // 60)
        seconds = int(eta_s % 60)
        eta_txt = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
        step_txt += f"   ETA: {eta_txt}"
    screen.blit(f_sm.render(step_txt, True, DIM), (28, y))
    y += 22

    if state["error"]:
        screen.blit(f_md.render(f"Error: {state['error'][:52]}", True, RED), (28, y))
        screen.blit(f_sm.render("Press any key to return to menu", True, DIM), (28, y + 28))
    elif state["done"]:
        screen.blit(f_md.render("Training complete!", True, GREEN), (28, y))
        screen.blit(f_md.render(f"Model: {state['model_path']}", True, ACCENT), (28, y + 26))
        screen.blit(f_sm.render("Press any key to return to menu", True, DIM), (28, y + 52))
    else:
        dots = "." * (int(elapsed * 2) % 4)
        screen.blit(f_md.render(f"Training in progress{dots}", True, ACCENT), (28, y))
        screen.blit(f_sm.render(
            "SB3 output in terminal  |  TensorBoard: logs/tensorboard", True, DIM),
            (28, y + 26))


def _run_training_mode(args) -> None:
    """
    Start PPO training in a background thread and show a Pygame progress window.
    Saves models/<run_name>.zip and models/<run_name>.json sidecar on completion.
    """
    rate  = _LOAD_LEVELS[args.load_level]
    state: dict = {"done": False, "error": None, "model_path": "", "timestep": 0}
    threading.Thread(target=_train_ppo, args=(args, rate, state), daemon=True).start()

    screen = pygame.display.set_mode((520, 340))
    pygame.display.set_caption(f"PPO Training — {args.run_name}")
    clock  = pygame.time.Clock()
    fonts  = (
        pygame.font.SysFont("consolas", 18, bold=True),
        pygame.font.SysFont("consolas", 14),
        pygame.font.SysFont("consolas", 12),
    )
    start = time.time()

    while not _training_window_quit_requested():
        elapsed = time.time() - start
        _render_training_status(screen, fonts, state, args, elapsed)
        pygame.display.flip()
        clock.tick(10)
        if state["done"] or state["error"]:
            # Re-render so the final state (complete/error) is visible before blocking
            elapsed = time.time() - start
            _render_training_status(screen, fonts, state, args, elapsed)
            pygame.display.flip()
            _wait_for_close(clock)
            break


_MODIFIER_KEYS = frozenset({
    pygame.K_LSHIFT, pygame.K_RSHIFT,
    pygame.K_LCTRL,  pygame.K_RCTRL,
    pygame.K_LALT,   pygame.K_RALT,
    pygame.K_LMETA,  pygame.K_RMETA,
    pygame.K_CAPSLOCK, pygame.K_NUMLOCK,
})


def _wait_for_close(clock) -> None:
    """Block until the user deliberately closes the window (ESC, Enter, Space, or X)."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key not in _MODIFIER_KEYS:
                return
        clock.tick(10)


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

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
    renderer.pre_step(env)
    action = _select_action(env, agent)
    _, reward, terminated, truncated, info = env.step(action)
    renderer.notify_step(env)
    if bridge:
        bridge.sync_step(env.agvs, env.tasks)
    return reward, terminated or truncated, info


def _finish_episode(env, agent, renderer, episode, total_reward, info,
                    bridge=None, registry=None, run_id=None) -> None:
    _log_episode(episode, total_reward, info)
    if registry is not None:
        registry.log_episode(run_id, episode, total_reward, info)
    _reset(env, agent, renderer, bridge)
    renderer.episode += 1


def _tick(env, agent, renderer, max_eps, done, total_reward, info,
          bridge=None, registry=None, run_id=None):
    if not renderer.should_step():
        return done, total_reward, info, False
    if done:
        _finish_episode(env, agent, renderer,
                        renderer.episode - 1, total_reward, info,
                        bridge, registry, run_id)
        stop = max_eps > 0 and renderer.episode > max_eps
        return False, 0.0, info, stop
    reward, done, info = _advance_episode(env, agent, renderer, bridge)
    return done, total_reward + reward, info, False


def _run_loop(env, agent, renderer, max_eps: int,
              bridge=None, registry=None, run_id=None) -> None:
    """Pygame-driven loop. Episodes are logged to registry when provided."""
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
                env, agent, renderer, max_eps, done, total_reward, info,
                bridge, registry, run_id)
            if stop:
                break

        if bridge:
            bridge.sync_frame(env.agvs, renderer._anim_t)

        renderer.render(env)
        renderer.tick()


def _run_coppeliasim_loop(env, agent, bridge, max_eps: int,
                          interp_steps: int, delay: float,
                          registry=None, run_id=None) -> None:
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
            if registry is not None:
                registry.log_episode(run_id, episode, total_reward, info)
            episode += 1
            if max_eps > 0 and episode > max_eps:
                break
            env.reset()
            total_reward = 0.0
            bridge.sync(env.agvs)
            bridge.sync_tasks(env.tasks)


def _run_evaluation_mode(args) -> None:
    """
    Visual evaluation run. Results are logged to experiments.db so they
    can be compared in Grafana against other agents and seeds.
    """
    from src.experiments import ExperimentRegistry

    env   = AGVFleetEnv(
        n_agvs=args.n_agvs, n_tasks_max=_N_TASKS_MAX,
        max_steps=args.steps, seed=args.seed)
    agent = build_agent(args, env)
    agent_label = {
        "astar":  "A*+Greedy",
        "ppo":    f"PPO ({args.model})",
        "random": "Random",
    }[args.agent]

    print(f"Agent      : {agent_label}")
    print(f"Renderer   : {args.renderer}")
    print(f"Episodes   : {'infinite' if args.episodes == 0 else args.episodes}")
    print(f"Seed       : {args.seed}")

    registry = ExperimentRegistry()
    run_id   = registry.start_run(
        agent=args.agent,
        seed=args.seed,
        n_agvs=args.n_agvs,
        n_tasks_max=_N_TASKS_MAX,
        max_steps=args.steps,
        task_arrival_rate=_LOAD_LEVELS.get("medium", 0.15),
        hyperparameters=_load_ppo_hyperparameters(args.model) if args.agent == "ppo" else None,
    )
    print(f"Registry   : run_id={run_id}")

    # ── CoppeliaSim-only ─────────────────────────────────────────────────
    if args.renderer == "coppeliasim":
        bridge = _build_bridge(args)
        try:
            _run_coppeliasim_loop(
                env, agent, bridge,
                max_eps=args.episodes,
                interp_steps=args.interp_steps,
                delay=args.delay,
                registry=registry,
                run_id=run_id,
            )
            registry.finish_run(run_id)
        except Exception:
            pass  # registry marks run interrupted via context manager on close
        finally:
            bridge.close()
            registry.close()
            env.close()
        return

    # ── Pygame (alone or with CoppeliaSim) ───────────────────────────────
    bridge = _build_bridge(args)

    renderer = PygameRenderer(
        title=f"AGV Fleet — {agent_label}",
        fps=args.fps,
        agent_label=agent_label,
    )
    print(f"Sim speed  : {args.fps} steps/s  (UP/DOWN to change)")
    print("Controls   : SPACE=pause  UP/DOWN=speed  R=reset  ESC/Q=quit\n")

    try:
        _run_loop(env, agent, renderer,
                  max_eps=args.episodes,
                  bridge=bridge,
                  registry=registry,
                  run_id=run_id)
        registry.finish_run(run_id)
    except Exception:
        pass
    finally:
        if bridge:
            bridge.close()
        registry.close()
        env.close()
        renderer.close()


def _load_ppo_hyperparameters(model_path: Optional[str]) -> Optional[dict]:
    if not model_path:
        return None
    sidecar = Path(str(model_path).removesuffix(".zip") + ".json")
    if sidecar.exists():
        with open(sidecar, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cli_mode = (args.agent is not None or args.n_agvs is not None or args.mode is not None)

    if cli_mode:
        # CLI invocation: run once and exit
        _resolve_args(args)
        if args.mode == "training":
            _run_training_mode(args)
        else:
            _run_evaluation_mode(args)
        pygame.quit()
        return

    # Interactive menu loop: after each run, return to the menu
    pygame.init()
    while True:
        config = _show_menu()
        if config is None:
            break
        mode = config.get("mode", "evaluation")
        args.mode = mode
        if mode == "training":
            args.run_name   = config["run_name"]
            args.n_agvs     = config["n_agvs"]
            args.timesteps  = config["timesteps"]
            args.load_level = config["load_level"]
            args.seed       = config["seed"]
            _run_training_mode(args)
        else:
            args.agent    = config["agent"]
            args.model    = config["model"]
            args.n_agvs   = config["n_agvs"]
            args.episodes = config["episodes"]
            args.seed     = config["seed"]
            args.renderer = config.get("renderer", "pygame")
            _run_evaluation_mode(args)

    pygame.quit()


if __name__ == "__main__":
    main()
