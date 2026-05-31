"""
Pre-launch configuration menu for run_visual.py.

Shown when the script is invoked without arguments. Returns a config
dict that run_visual.py uses to build the env, agent, and renderer.

Modes
-----
evaluation  Visual run + automatic logging to ExperimentRegistry (experiments.db).
            Fields: agent, n_agvs, episodes, seed, layout, renderer.

training    Headless PPO training with a progress window.
            Fields: run_name, n_agvs, timesteps, load_level, seed.
            Output: models/<run_name>.zip  +  models/<run_name>.json sidecar.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pygame

from src.env.plant_map import PlantMap


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

_BG           = ( 20,  20,  28)
_TEXT         = (230, 230, 230)
_TEXT_DIM     = (120, 120, 140)
_ACCENT       = ( 90, 160, 255)
_BTN_BG       = ( 45,  45,  60)
_BTN_HOVER    = ( 65,  65,  85)
_BTN_ACTIVE   = ( 40, 110, 200)
_BTN_DISABLED = ( 35,  35,  45)
_BTN_BORDER   = ( 70,  70,  90)
_INPUT_BG     = ( 28,  28,  40)
_INPUT_FOCUS  = ( 38,  48,  80)
_LAUNCH_BG    = ( 30, 130,  55)
_LAUNCH_HOVER = ( 40, 160,  70)

_WIN_W = 600
_WIN_H = 780

_MODELS_DIR      = Path(__file__).parent.parent.parent / "models"
_LEFT_PAD        = 80
_MODELS_PER_PAGE = 3
_MAX_N_AGVS  = len(PlantMap.AGV_SPAWN_POSITIONS)


# ---------------------------------------------------------------------------
# LaunchMenu
# ---------------------------------------------------------------------------

class LaunchMenu:
    """
    Blocking Pygame configuration screen.

    Call run() — it returns a config dict when the user presses LAUNCH,
    or None when the user quits (ESC / window close).

    Config dict keys — evaluation mode:
        mode      : "evaluation"
        n_agvs    : int
        episodes  : int
        seed      : int
        layout    : str   ("L1")
        renderer  : str   ("pygame" | "coppeliasim" | "both")
        agent     : str   ("astar" | "random" | "ppo")
        model     : str | None   (path to .zip for PPO, else None)

    Config dict keys — training mode:
        mode       : "training"
        run_name   : str
        n_agvs     : int
        timesteps  : int
        load_level : str  ("low" | "medium" | "high")
        seed       : int
    """

    def __init__(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((_WIN_W, _WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("AGV Fleet — Setup")
        from ._icon import make_app_icon
        pygame.display.set_icon(make_app_icon())
        self._clock   = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("consolas", 22, bold=True)
        self._font_md = pygame.font.SysFont("consolas", 15)
        self._font_sm = pygame.font.SysFont("consolas", 13)

        # ── Mode ──────────────────────────────────────────────────────────
        self._mode: str = "evaluation"

        # ── Shared / Evaluation fields ────────────────────────────────────
        self._n_input:     str  = "4"
        self._n_focused:   bool = False
        self._eps_input:   str  = "0"
        self._eps_focused: bool = False
        self._seed_input:  str  = "42"
        self._seed_focused: bool = False
        self._layout:    str = "L1"
        self._renderer:  str = "pygame"
        self._agent:     str = "astar"
        self._model:     Optional[Path] = None

        # ── Training fields ───────────────────────────────────────────────
        self._run_name_input:   str  = "ppo_fleet"
        self._run_name_focused: bool = False
        self._ts_input:   str  = "300000"
        self._ts_focused: bool = False
        self._load_level: str  = "medium"

        # ── CoppeliaSim state ─────────────────────────────────────────────
        self._build_status:  str                        = "idle"
        self._build_message: str                        = ""
        self._build_thread:  Optional[threading.Thread] = None
        self._cs_status:  str = "idle"
        self._cs_message: str = ""

        self._models: List[Path] = self._scan_models()
        self._model_page: int = 0
        self._regions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Optional[Dict[str, Any]]:
        """Block until LAUNCH or quit. Returns config dict or None."""
        while True:
            result = self._handle_events()
            if result == "quit":
                return None
            if result == "launch":
                return self._build_config()
            if self._cs_status == "ok":
                return self._build_config()
            self._draw()
            self._clock.tick(30)

    def close(self) -> None:
        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        # pygame stays alive so main() can loop back to the menu

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_models() -> List[Path]:
        if not _MODELS_DIR.exists():
            return []
        return sorted(_MODELS_DIR.glob("*.zip"))

    def _build_config(self) -> Dict[str, Any]:
        seed = max(0, self._int_or(self._seed_input, 42))
        if self._mode == "training":
            return {
                "mode":       "training",
                "run_name":   self._run_name_input.strip() or "ppo_fleet",
                "n_agvs":     max(1, self._int_or(self._n_input, 4)),
                "timesteps":  max(10_000, self._int_or(self._ts_input, 300_000)),
                "load_level": self._load_level,
                "seed":       seed,
            }
        # evaluation
        raw_n   = self._n_input
        raw_eps = self._eps_input
        n   = int(raw_n)   if raw_n.isdigit()   and raw_n   else 4
        eps = int(raw_eps) if raw_eps.isdigit() and raw_eps else 0
        return {
            "mode":     "evaluation",
            "n_agvs":   max(1, n),
            "episodes": max(0, eps),
            "seed":     seed,
            "layout":   self._layout,
            "renderer": self._renderer,
            "agent":    self._agent,
            "model":    str(self._model) if self._model else None,
        }

    def _config_valid(self) -> bool:
        if self._mode == "training":
            n  = self._n_input
            ts = self._ts_input
            return (
                bool(self._run_name_input.strip())
                and bool(n) and n.isdigit() and int(n) >= 1
                and bool(ts) and ts.isdigit() and int(ts) >= 10_000
            )
        if self._agent == "ppo" and self._model is None:
            return False
        n = self._n_input
        return bool(n) and n.isdigit() and int(n) >= 1

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> Optional[str]:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                result = self._handle_key(event)
                if result:
                    return result
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                result = self._handle_click(mouse)
                if result:
                    return result
        return None

    @staticmethod
    def _edit_int_field(current: str, event: pygame.event.Event, max_val: int) -> str:
        if event.key == pygame.K_BACKSPACE:
            return current[:-1]
        if event.unicode.isdigit():
            candidate = (current + event.unicode).lstrip("0") or ""
            if not candidate or int(candidate) <= max_val:
                return candidate
        return current

    @staticmethod
    def _edit_text_field(current: str, event: pygame.event.Event, max_len: int) -> str:
        if event.key == pygame.K_BACKSPACE:
            return current[:-1]
        c = event.unicode
        if c and (c.isalnum() or c in "_-") and len(current) < max_len:
            return current + c
        return current

    def _handle_key(self, event: pygame.event.Event) -> Optional[str]:
        if event.key == pygame.K_ESCAPE:
            return "quit"
        if event.key == pygame.K_RETURN:
            return "launch" if self._config_valid() else None
        if self._run_name_focused:
            self._run_name_input = self._edit_text_field(self._run_name_input, event, 30)
        if self._ts_focused:
            self._ts_input = self._edit_int_field(self._ts_input, event, 9_999_999)
        if self._n_focused:
            self._n_input = self._edit_int_field(self._n_input, event, _MAX_N_AGVS)
        if self._eps_focused:
            self._eps_input = self._edit_int_field(self._eps_input, event, 9999)
        if self._seed_focused:
            self._seed_input = self._edit_int_field(self._seed_input, event, 9999)
        return None

    def _handle_click(self, mouse: tuple) -> Optional[str]:
        self._n_focused        = False
        self._eps_focused      = False
        self._seed_focused     = False
        self._run_name_focused = False
        self._ts_focused       = False

        for region in self._regions:
            if not region["rect"].collidepoint(mouse):
                continue
            result = self._apply_action(region["action"])
            if result:
                return result
        return None

    def _apply_counter_action(self, action: str) -> None:
        if action == "n_dec":
            self._n_input = str(max(1, self._int_or(self._n_input, 4) - 1))
        elif action == "n_inc":
            self._n_input = str(min(_MAX_N_AGVS, self._int_or(self._n_input, 4) + 1))
        elif action == "eps_dec":
            self._eps_input = str(max(0, self._int_or(self._eps_input, 0) - 1))
        elif action == "eps_inc":
            self._eps_input = str(min(9999, self._int_or(self._eps_input, 0) + 1))
        elif action == "seed_dec":
            self._seed_input = str(max(0, self._int_or(self._seed_input, 42) - 1))
        elif action == "seed_inc":
            self._seed_input = str(min(9999, self._int_or(self._seed_input, 42) + 1))
        elif action == "ts_dec":
            self._ts_input = str(max(10_000, self._int_or(self._ts_input, 300_000) - 50_000))
        elif action == "ts_inc":
            self._ts_input = str(min(9_999_999, self._int_or(self._ts_input, 300_000) + 50_000))

    def _apply_action(self, action: str) -> Optional[str]:
        if action == "quit":
            return "quit"
        if action == "launch":
            return self._handle_launch()
        if action == "focus_n":
            self._n_focused        = True
        if action == "focus_eps":
            self._eps_focused      = True
        if action == "focus_seed":
            self._seed_focused     = True
        if action == "focus_run_name":
            self._run_name_focused = True
        if action == "focus_ts":
            self._ts_focused       = True
        self._apply_counter_action(action)
        if action == "mode:evaluation":
            self._mode = "evaluation"
        if action == "mode:training":
            self._mode = "training"
        if action.startswith("load_level:"):
            self._load_level = action.split(":", 1)[1]
        if action.startswith("layout:") and action.split(":", 1)[1] == "L1":
            self._layout = "L1"
        if action == "agent:astar":
            self._agent, self._model = "astar", None
        if action == "agent:random":
            self._agent, self._model = "random", None
        if action.startswith("agent:ppo:"):
            self._agent = "ppo"
            self._model = Path(action.split("agent:ppo:", 1)[1])
        if action == "model_page:prev":
            self._model_page = max(0, self._model_page - 1)
        if action == "model_page:next":
            n_pages = max(1, (len(self._models) + _MODELS_PER_PAGE - 1) // _MODELS_PER_PAGE)
            self._model_page = min(n_pages - 1, self._model_page + 1)
        if action.startswith("renderer:"):
            self._renderer   = action.split("renderer:", 1)[1]
            self._cs_status  = "idle"
            self._cs_message = ""
        if action == "build_scene" and self._build_status != "building":
            self._build_status  = "building"
            self._build_message = ""
            self._build_thread  = threading.Thread(
                target=self._run_build_scene, daemon=True
            )
            self._build_thread.start()
        return None

    def _run_build_scene(self) -> None:
        try:
            from src.coppeliasim.scene_builder import build_scene
            n_agvs = max(1, self._int_or(self._n_input, 4))
            build_scene(n_agvs=n_agvs)
            self._build_status = "done"
        except Exception as exc:
            self._build_message = str(exc)[:55]
            self._build_status  = "error"

    def _check_cs(self) -> None:
        try:
            from src.coppeliasim.bridge import _connect_zmq
            _connect_zmq('localhost', 23000)
            self._cs_status = "ok"
        except Exception as exc:
            self._cs_message = str(exc).split('\n')[0][:55]
            self._cs_status  = "error"

    def _handle_launch(self) -> Optional[str]:
        if not self._config_valid():
            return None
        if self._mode == "training":
            return "launch"
        if self._renderer not in ("coppeliasim", "both"):
            return "launch"
        if self._cs_status == "ok":
            return "launch"
        if self._cs_status == "checking":
            return None
        self._cs_status  = "checking"
        self._cs_message = ""
        threading.Thread(target=self._check_cs, daemon=True).start()
        return None

    @staticmethod
    def _int_or(value: str, default: int) -> int:
        return int(value) if value.isdigit() and value else default

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self._regions.clear()
        self._screen.fill(_BG)
        mouse = pygame.mouse.get_pos()

        w = self._screen.get_width()

        y = 28
        surf = self._font_lg.render("AGV Fleet Simulator", True, _ACCENT)
        self._screen.blit(surf, surf.get_rect(centerx=w // 2, top=y))
        y += surf.get_height() + 4
        surf = self._font_sm.render("Launch configuration", True, _TEXT_DIM)
        self._screen.blit(surf, surf.get_rect(centerx=w // 2, top=y))
        y += surf.get_height() + 20

        y = self._draw_mode_selector(y, mouse) + 22

        if self._mode == "training":
            y = self._draw_training_fields(y, mouse)
        else:
            y = self._draw_int_row(y, mouse)       + 22
            y = self._draw_seed_row(y, mouse)      + 22
            y = self._draw_layouts(y, mouse)       + 22
            y = self._draw_renderers(y, mouse)     + 14
            y = self._draw_setup(y, mouse)         + 22
            y = self._draw_agents(y, mouse)        + 12
            y = self._draw_cs_connect_status(y)    + 10

        self._draw_launch_row(y, mouse)

        hovering = any(r["rect"].collidepoint(mouse) for r in self._regions)
        pygame.mouse.set_cursor(
            pygame.SYSTEM_CURSOR_HAND if hovering else pygame.SYSTEM_CURSOR_ARROW
        )

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Mode selector
    # ------------------------------------------------------------------

    def _draw_mode_selector(self, y: int, mouse: tuple) -> int:
        self._section_label("Mode", y)
        y += 22
        options = [("evaluation", "Evaluation"), ("training", "Train PPO")]
        bw, bh, gap = 150, 34, 10
        x = _LEFT_PAD
        for key, label in options:
            rect = pygame.Rect(x, y, bw, bh)
            x += bw + gap
            self._regions.append({"rect": rect, "action": f"mode:{key}"})
            self._btn(rect, label, mouse, active=(self._mode == key))
        return y + bh

    # ------------------------------------------------------------------
    # Training fields
    # ------------------------------------------------------------------

    def _draw_training_fields(self, y: int, mouse: tuple) -> int:
        # Run name
        self._section_label("Run name", y)
        y += 22
        bh = 36
        run_rect = pygame.Rect(_LEFT_PAD, y, _WIN_W - _LEFT_PAD * 2, bh)
        self._regions.append({"rect": run_rect, "action": "focus_run_name"})
        bg = _INPUT_FOCUS if self._run_name_focused else _INPUT_BG
        pygame.draw.rect(self._screen, bg,         run_rect, border_radius=4)
        pygame.draw.rect(self._screen, _BTN_BORDER, run_rect, 1, border_radius=4)
        surf = self._font_md.render(self._run_name_input or "ppo_fleet", True, _TEXT)
        self._screen.blit(surf, surf.get_rect(midleft=(run_rect.left + 10, run_rect.centery)))
        if self._run_name_focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = run_rect.left + 10 + surf.get_width() + 2
            pygame.draw.line(self._screen, _TEXT,
                             (cx, run_rect.top + 7), (cx, run_rect.bottom - 7), 2)
        y += bh + 18

        # AGVs and Seed on the same row
        self._draw_int_field(
            y, mouse, label=f"AGVs  (max {_MAX_N_AGVS})",
            value=self._n_input, focused=self._n_focused,
            x=_LEFT_PAD, field_w=68,
            action_dec="n_dec", action_inc="n_inc", action_focus="focus_n",
        )
        self._draw_int_field(
            y, mouse, label="Seed",
            value=self._seed_input, focused=self._seed_focused,
            x=_LEFT_PAD + 240, field_w=80,
            action_dec="seed_dec", action_inc="seed_inc", action_focus="focus_seed",
        )
        y += 22 + 36 + 18

        # Timesteps
        self._section_label("Timesteps  (± 50 000)", y)
        y += 22
        bw_btn, bh, gap = 50, 36, 6
        field_w = 120
        dec_rect   = pygame.Rect(_LEFT_PAD, y, bw_btn, bh)
        field_rect = pygame.Rect(_LEFT_PAD + bw_btn + gap, y, field_w, bh)
        inc_rect   = pygame.Rect(_LEFT_PAD + bw_btn + gap + field_w + gap, y, bw_btn, bh)
        self._regions += [
            {"rect": dec_rect,   "action": "ts_dec"},
            {"rect": field_rect, "action": "focus_ts"},
            {"rect": inc_rect,   "action": "ts_inc"},
        ]
        self._btn(dec_rect, "-50k", mouse)
        self._btn(inc_rect, "+50k", mouse)
        bg = _INPUT_FOCUS if self._ts_focused else _INPUT_BG
        pygame.draw.rect(self._screen, bg,         field_rect, border_radius=4)
        pygame.draw.rect(self._screen, _BTN_BORDER, field_rect, 1, border_radius=4)
        val = self._ts_input
        display = f"{int(val):,}" if val.isdigit() else val
        surf = self._font_md.render(display, True, _TEXT)
        self._screen.blit(surf, surf.get_rect(center=field_rect.center))
        if self._ts_focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = field_rect.centerx + surf.get_width() // 2 + 3
            pygame.draw.line(self._screen, _TEXT,
                             (cx, field_rect.top + 7), (cx, field_rect.bottom - 7), 2)
        y += bh + 18

        # Load level
        self._section_label("Load level", y)
        y += 22
        bw, bh, gap = 120, 34, 8
        x = _LEFT_PAD
        for key, label in [("low", "Low  (0.08)"), ("medium", "Med  (0.15)"), ("high", "High  (0.25)")]:
            rect = pygame.Rect(x, y, bw, bh)
            x += bw + gap
            self._regions.append({"rect": rect, "action": f"load_level:{key}"})
            self._btn(rect, label, mouse, active=(self._load_level == key))
        y += bh + 14

        # Default hyperparameter notes
        note = "lr=3e-4   n_steps=2048   batch=64   gamma=0.99   ent=0.01"
        surf = self._font_sm.render(note, True, _TEXT_DIM)
        self._screen.blit(surf, (_LEFT_PAD, y))
        y += surf.get_height() + 16

        return y

    # ------------------------------------------------------------------
    # Evaluation fields
    # ------------------------------------------------------------------

    def _draw_int_row(self, y: int, mouse: tuple) -> int:
        bh = 36
        self._draw_int_field(
            y, mouse, label=f"Number of AGVs  (max {_MAX_N_AGVS})",
            value=self._n_input, focused=self._n_focused,
            x=_LEFT_PAD, field_w=68,
            action_dec="n_dec", action_inc="n_inc", action_focus="focus_n",
        )
        self._draw_int_field(
            y, mouse, label="Max episodes  (0=inf)",
            value=self._eps_input, focused=self._eps_focused,
            x=_LEFT_PAD + 200, field_w=80,
            action_dec="eps_dec", action_inc="eps_inc", action_focus="focus_eps",
        )
        return y + 22 + bh

    def _draw_seed_row(self, y: int, mouse: tuple) -> int:
        self._draw_int_field(
            y, mouse, label="Seed",
            value=self._seed_input, focused=self._seed_focused,
            x=_LEFT_PAD, field_w=80,
            action_dec="seed_dec", action_inc="seed_inc", action_focus="focus_seed",
        )
        # Registry hint
        hint = "Results logged to experiments.db"
        surf = self._font_sm.render(hint, True, (60, 180, 90))
        self._screen.blit(surf, (_LEFT_PAD + 240, y + 30))
        return y + 22 + 36

    def _draw_int_field(
        self, y: int, mouse: tuple,
        label: str, value: str, focused: bool,
        x: int, field_w: int,
        action_dec: str, action_inc: str, action_focus: str,
    ) -> None:
        self._section_label(label, y, x=x)
        y += 22

        bw, bh, gap = 34, 36, 6
        dec_rect   = pygame.Rect(x,                              y, bw,      bh)
        field_rect = pygame.Rect(x + bw + gap,                   y, field_w, bh)
        inc_rect   = pygame.Rect(x + bw + gap + field_w + gap,   y, bw,      bh)

        self._regions += [
            {"rect": dec_rect,   "action": action_dec},
            {"rect": field_rect, "action": action_focus},
            {"rect": inc_rect,   "action": action_inc},
        ]

        self._btn(dec_rect, "-", mouse)
        self._btn(inc_rect, "+", mouse)

        field_bg = _INPUT_FOCUS if focused else _INPUT_BG
        pygame.draw.rect(self._screen, field_bg,    field_rect, border_radius=4)
        pygame.draw.rect(self._screen, _BTN_BORDER, field_rect, 1, border_radius=4)
        surf = self._font_md.render(value, True, _TEXT)
        self._screen.blit(surf, surf.get_rect(center=field_rect.center))

        if focused and (pygame.time.get_ticks() // 500) % 2 == 0:
            cx = field_rect.centerx + surf.get_width() // 2 + 3
            pygame.draw.line(self._screen, _TEXT,
                             (cx, field_rect.top + 7), (cx, field_rect.bottom - 7), 2)

    def _draw_layouts(self, y: int, mouse: tuple) -> int:
        self._section_label("Layout", y)
        y += 22

        layouts = [("L1", True), ("L2", False), ("L3", False)]
        bw, bh, gap = 84, 34, 10
        x = _LEFT_PAD

        for label, enabled in layouts:
            rect = pygame.Rect(x, y, bw, bh)
            x += bw + gap
            if enabled:
                self._regions.append({"rect": rect, "action": f"layout:{label}"})
                self._btn(rect, label, mouse, active=(self._layout == label))
            else:
                self._btn_disabled(rect, f"{label} (soon)")

        return y + bh

    def _draw_renderers(self, y: int, mouse: tuple) -> int:
        self._section_label("Visualisation", y)
        y += 22

        options = [
            ("pygame",       "Pygame 2D"),
            ("coppeliasim",  "CoppeliaSim 3D"),
            ("both",         "Both (sync)"),
        ]
        bw, bh, gap = 124, 34, 8
        x = _LEFT_PAD
        for key, label in options:
            rect = pygame.Rect(x, y, bw, bh)
            x += bw + gap
            self._regions.append({"rect": rect, "action": f"renderer:{key}"})
            self._btn(rect, label, mouse, active=(self._renderer == key))

        return y + bh

    def _draw_setup(self, y: int, mouse: tuple) -> int:
        needs_cs = self._renderer in ("coppeliasim", "both")
        self._section_label("CoppeliaSim setup", y)
        y += 22

        bw = _WIN_W - _LEFT_PAD * 2
        bh = 34
        rect = pygame.Rect(_LEFT_PAD, y, bw, bh)

        is_error = self._build_status == "error"
        _STATUS_LABELS = {
            "idle":     "Build / Rebuild scene in CoppeliaSim",
            "building": "Building...  (please wait)",
            "done":     "✓  Scene ready — save it in CoppeliaSim",
            "error":    f"✗  {(self._build_message or 'Build error')[:32]}  — click to retry",
        }
        label = _STATUS_LABELS.get(self._build_status, "Build scene")

        _STATUS_BG = {
            "done":  (35, 130, 60),
            "error": (130, 50, 20),
        }
        _STATUS_BORDER = {
            "done":  _BTN_BORDER,
            "error": (220, 90, 40),
        }

        if not needs_cs or self._build_status == "building":
            self._btn_disabled(rect, label)
        else:
            self._regions.append({"rect": rect, "action": "build_scene"})
            hov = rect.collidepoint(mouse)
            default_bg = _BTN_HOVER if hov else _BTN_BG
            bg     = _STATUS_BG.get(self._build_status, default_bg)
            if is_error and hov:
                bg = (160, 65, 25)
            border = _STATUS_BORDER.get(self._build_status, _BTN_BORDER)
            pygame.draw.rect(self._screen, bg,     rect, border_radius=5)
            pygame.draw.rect(self._screen, border, rect, 1, border_radius=5)
            surf = self._font_sm.render(label, True, _TEXT)
            self._screen.blit(surf, surf.get_rect(center=rect.center))

        return y + bh

    def _draw_agents(self, y: int, mouse: tuple) -> int:
        self._section_label("Agent", y)
        y += 22

        bw, bh, gap = 140, 34, 10
        x = _LEFT_PAD

        for agent_id, label in [("astar", "A* + Greedy"), ("random", "Random")]:
            rect = pygame.Rect(x, y, bw, bh)
            x += bw + gap
            self._regions.append({"rect": rect, "action": f"agent:{agent_id}"})
            active = self._agent == agent_id and self._model is None
            self._btn(rect, label, mouse, active=active)

        y += bh + 8

        if self._models:
            n_pages  = max(1, (len(self._models) + _MODELS_PER_PAGE - 1) // _MODELS_PER_PAGE)
            page     = max(0, min(self._model_page, n_pages - 1))
            start_i  = page * _MODELS_PER_PAGE
            page_models = self._models[start_i: start_i + _MODELS_PER_PAGE]

            btn_w = _WIN_W - _LEFT_PAD * 2
            for model_path in page_models:
                label = f"PPO:  {model_path.name}"
                rect  = pygame.Rect(_LEFT_PAD, y, btn_w, bh)
                self._regions.append({"rect": rect, "action": f"agent:ppo:{model_path}"})
                active = self._agent == "ppo" and self._model == model_path
                self._btn(rect, label, mouse, active=active)
                y += bh + 6

            if n_pages > 1:
                arr_w, arr_h = 34, bh
                gap = 8
                label_txt = f"{page + 1} / {n_pages}"
                lbl_surf  = self._font_sm.render(label_txt, True, _TEXT_DIM)
                total_w   = arr_w + gap + lbl_surf.get_width() + gap + arr_w
                cx        = _LEFT_PAD + btn_w // 2
                lx        = cx - total_w // 2

                prev_rect = pygame.Rect(lx, y, arr_w, arr_h)
                next_rect = pygame.Rect(lx + arr_w + gap + lbl_surf.get_width() + gap, y, arr_w, arr_h)

                self._regions.append({"rect": prev_rect, "action": "model_page:prev"})
                self._regions.append({"rect": next_rect, "action": "model_page:next"})

                prev_enabled = page > 0
                next_enabled = page < n_pages - 1
                if prev_enabled:
                    self._btn(prev_rect, "◀", mouse)
                else:
                    self._btn_disabled(prev_rect, "◀")
                if next_enabled:
                    self._btn(next_rect, "▶", mouse)
                else:
                    self._btn_disabled(next_rect, "▶")
                self._screen.blit(
                    lbl_surf,
                    lbl_surf.get_rect(midleft=(lx + arr_w + gap, y + arr_h // 2)))
                y += arr_h + 6
        else:
            surf = self._font_sm.render(
                "No PPO models found in models/  (train one first)", True, _TEXT_DIM)
            self._screen.blit(surf, (_LEFT_PAD, y + 8))
            y += bh

        return y

    def _draw_cs_connect_status(self, y: int) -> int:
        needs_cs = self._renderer in ("coppeliasim", "both")
        if not needs_cs or self._cs_status == "idle":
            return y

        cx = self._screen.get_width() // 2
        if self._cs_status == "checking":
            s = self._font_sm.render("Connecting to CoppeliaSim...", True, _TEXT_DIM)
            self._screen.blit(s, s.get_rect(centerx=cx, top=y))
            return y + s.get_height()

        if self._cs_status == "error":
            msg = self._cs_message or "Cannot connect to CoppeliaSim"
            s1 = self._font_sm.render(f"✗  {msg}", True, (220, 80, 80))
            s2 = self._font_sm.render(
                "→ Start CoppeliaSim and load the plant scene", True, _TEXT_DIM)
            self._screen.blit(s1, s1.get_rect(centerx=cx, top=y))
            self._screen.blit(s2, s2.get_rect(centerx=cx, top=y + s1.get_height() + 3))
            return y + s1.get_height() + 3 + s2.get_height()

        return y

    def _draw_launch_row(self, y: int, mouse: tuple) -> None:
        bw, bh = 160, 44
        cx     = self._screen.get_width() // 2

        launch_rect = pygame.Rect(cx - bw - 8, y, bw, bh)
        quit_rect   = pygame.Rect(cx + 8,       y, bw, bh)

        valid    = self._config_valid()
        needs_cs = self._renderer in ("coppeliasim", "both") and self._mode == "evaluation"
        checking = needs_cs and self._cs_status == "checking"
        cs_error = needs_cs and self._cs_status == "error"

        if not valid or checking:
            label = "Connecting..." if checking else "LAUNCH"
            pygame.draw.rect(self._screen, _BTN_DISABLED, launch_rect, border_radius=6)
            pygame.draw.rect(self._screen, (50, 50, 60),  launch_rect, 1, border_radius=6)
            surf = self._font_md.render(label, True, _TEXT_DIM)
        elif cs_error:
            self._regions.append({"rect": launch_rect, "action": "launch"})
            hov = launch_rect.collidepoint(mouse)
            pygame.draw.rect(self._screen,
                             (160, 65, 25) if hov else (130, 50, 20),
                             launch_rect, border_radius=6)
            pygame.draw.rect(self._screen, (220, 90, 40), launch_rect, 1, border_radius=6)
            surf = self._font_md.render("LAUNCH  (retry)", True, _TEXT)
        else:
            self._regions.append({"rect": launch_rect, "action": "launch"})
            hov = launch_rect.collidepoint(mouse)
            pygame.draw.rect(self._screen, _LAUNCH_HOVER if hov else _LAUNCH_BG,
                             launch_rect, border_radius=6)
            pygame.draw.rect(self._screen, (60, 200, 100), launch_rect, 1, border_radius=6)
            label = "TRAIN" if self._mode == "training" else "LAUNCH"
            surf = self._font_md.render(label, True, _TEXT)
        self._screen.blit(surf, surf.get_rect(center=launch_rect.center))

        self._regions.append({"rect": quit_rect, "action": "quit"})
        self._btn(quit_rect, "QUIT", mouse)

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _section_label(self, text: str, y: int, x: int = _LEFT_PAD) -> None:
        surf = self._font_sm.render(text.upper(), True, _TEXT_DIM)
        self._screen.blit(surf, (x, y))

    def _btn(self, rect: pygame.Rect, label: str, mouse: tuple,
             active: bool = False) -> None:
        hov = rect.collidepoint(mouse)
        if active:
            bg = _BTN_ACTIVE
        elif hov:
            bg = _BTN_HOVER
        else:
            bg = _BTN_BG
        pygame.draw.rect(self._screen, bg,         rect, border_radius=5)
        pygame.draw.rect(self._screen, _BTN_BORDER, rect, 1, border_radius=5)
        surf = self._font_sm.render(label, True, _TEXT)
        self._screen.blit(surf, surf.get_rect(center=rect.center))

    def _btn_disabled(self, rect: pygame.Rect, label: str) -> None:
        pygame.draw.rect(self._screen, _BTN_DISABLED, rect, border_radius=5)
        pygame.draw.rect(self._screen, (50, 50, 60),  rect, 1, border_radius=5)
        surf = self._font_sm.render(label, True, _TEXT_DIM)
        self._screen.blit(surf, surf.get_rect(center=rect.center))
