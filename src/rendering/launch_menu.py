"""
Pre-launch configuration menu for run_visual.py.

Shown when the script is invoked without arguments. Returns a config
dict that run_visual.py uses to build the env, agent, and renderer.
"""

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

_WIN_W = 500
_WIN_H = 470

_MODELS_DIR  = Path(__file__).parent.parent.parent / "models"
_LEFT_PAD    = 80
_MAX_N_AGVS  = len(PlantMap.AGV_SPAWN_POSITIONS)


# ---------------------------------------------------------------------------
# LaunchMenu
# ---------------------------------------------------------------------------

class LaunchMenu:
    """
    Blocking Pygame configuration screen.

    Call run() — it returns a config dict when the user presses LAUNCH,
    or None when the user quits (ESC / window close).

    Config dict keys:
        n_agvs  : int
        layout  : str   ("L1")
        agent   : str   ("astar" | "random" | "ppo")
        model   : str | None   (path to .zip for PPO, else None)
    """

    def __init__(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode((_WIN_W, _WIN_H))
        pygame.display.set_caption("AGV Fleet — Setup")
        self._clock   = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("consolas", 22, bold=True)
        self._font_md = pygame.font.SysFont("consolas", 15)
        self._font_sm = pygame.font.SysFont("consolas", 13)

        # State
        self._n_input:    str           = "4"
        self._n_focused:  bool          = False
        self._eps_input:  str           = "0"
        self._eps_focused: bool         = False
        self._layout:     str           = "L1"
        self._agent:     str           = "astar"
        self._model:     Optional[Path] = None

        self._models: List[Path] = self._scan_models()

        # Clickable regions rebuilt each frame
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
            self._draw()
            self._clock.tick(30)

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_models() -> List[Path]:
        if not _MODELS_DIR.exists():
            return []
        return sorted(_MODELS_DIR.glob("*.zip"))

    def _build_config(self) -> Dict[str, Any]:
        raw_n   = self._n_input
        raw_eps = self._eps_input
        n   = int(raw_n)   if raw_n.isdigit()   and raw_n   else 4
        eps = int(raw_eps) if raw_eps.isdigit() and raw_eps else 0
        return {
            "n_agvs":   max(1, n),
            "episodes": max(0, eps),
            "layout":   self._layout,
            "agent":    self._agent,
            "model":    str(self._model) if self._model else None,
        }

    def _config_valid(self) -> bool:
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
        """Apply a keypress to a numeric text field. Returns the updated string."""
        if event.key == pygame.K_BACKSPACE:
            return current[:-1]
        if event.unicode.isdigit():
            candidate = (current + event.unicode).lstrip("0") or ""
            if not candidate or int(candidate) <= max_val:
                return candidate
        return current

    def _handle_key(self, event: pygame.event.Event) -> Optional[str]:
        if event.key == pygame.K_ESCAPE:
            return "quit"
        if event.key == pygame.K_RETURN:
            return "launch" if self._config_valid() else None
        if self._n_focused:
            self._n_input   = self._edit_int_field(self._n_input,   event, _MAX_N_AGVS)
        if self._eps_focused:
            self._eps_input = self._edit_int_field(self._eps_input, event, 9999)
        return None

    def _handle_click(self, mouse: tuple) -> Optional[str]:
        # Blur all text fields by default; re-focus if clicking one
        self._n_focused   = False
        self._eps_focused = False

        for region in self._regions:
            if not region["rect"].collidepoint(mouse):
                continue
            result = self._apply_action(region["action"])
            if result:
                return result
        return None

    def _apply_action(self, action: str) -> Optional[str]:
        if action == "quit":
            return "quit"
        if action == "launch":
            return "launch" if self._config_valid() else None
        if action == "focus_n":
            self._n_focused   = True
        if action == "focus_eps":
            self._eps_focused = True
        if action == "n_dec":
            self._n_input = str(max(1, self._int_or(self._n_input, 4) - 1))
        if action == "n_inc":
            self._n_input = str(min(_MAX_N_AGVS, self._int_or(self._n_input, 4) + 1))
        if action == "eps_dec":
            self._eps_input = str(max(0, self._int_or(self._eps_input, 0) - 1))
        if action == "eps_inc":
            self._eps_input = str(min(9999, self._int_or(self._eps_input, 0) + 1))
        if action.startswith("layout:") and action.split(":", 1)[1] == "L1":
            self._layout = "L1"
        if action == "agent:astar":
            self._agent, self._model = "astar", None
        if action == "agent:random":
            self._agent, self._model = "random", None
        if action.startswith("agent:ppo:"):
            self._agent = "ppo"
            self._model = Path(action.split("agent:ppo:", 1)[1])
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

        # Title
        y = 28
        surf = self._font_lg.render("AGV Fleet Simulator", True, _ACCENT)
        self._screen.blit(surf, surf.get_rect(centerx=_WIN_W // 2, top=y))
        y += surf.get_height() + 4
        surf = self._font_sm.render("Launch configuration", True, _TEXT_DIM)
        self._screen.blit(surf, surf.get_rect(centerx=_WIN_W // 2, top=y))
        y += surf.get_height() + 28

        y = self._draw_int_row(y, mouse) + 22
        y = self._draw_layouts(y, mouse) + 22
        y = self._draw_agents(y, mouse)  + 32
        self._draw_launch_row(y, mouse)

        pygame.display.flip()

    def _draw_int_row(self, y: int, mouse: tuple) -> int:
        """Draw AGVs and Episodes fields side by side."""
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

    def _draw_int_field(
        self, y: int, mouse: tuple,
        label: str, value: str, focused: bool,
        x: int, field_w: int,
        action_dec: str, action_inc: str, action_focus: str,
    ) -> None:
        self._section_label(label, y, x=x)
        y += 22

        bw, bh, gap = 34, 36, 6
        dec_rect   = pygame.Rect(x,                     y, bw,      bh)
        field_rect = pygame.Rect(x + bw + gap,          y, field_w, bh)
        inc_rect   = pygame.Rect(x + bw + gap + field_w + gap, y, bw, bh)

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
            for model_path in self._models:
                label = f"PPO:  {model_path.name}"
                rect  = pygame.Rect(_LEFT_PAD, y, _WIN_W - _LEFT_PAD * 2, bh)
                self._regions.append({"rect": rect, "action": f"agent:ppo:{model_path}"})
                active = self._agent == "ppo" and self._model == model_path
                self._btn(rect, label, mouse, active=active)
                y += bh + 6
        else:
            surf = self._font_sm.render(
                "No PPO models found in models/  (train one first)", True, _TEXT_DIM)
            self._screen.blit(surf, (_LEFT_PAD, y + 8))
            y += bh

        return y

    def _draw_launch_row(self, y: int, mouse: tuple) -> None:
        bw, bh = 160, 44
        cx     = _WIN_W // 2

        launch_rect = pygame.Rect(cx - bw - 8, y, bw, bh)
        quit_rect   = pygame.Rect(cx + 8,       y, bw, bh)

        # LAUNCH
        valid = self._config_valid()
        if valid:
            self._regions.append({"rect": launch_rect, "action": "launch"})
            hov = launch_rect.collidepoint(mouse)
            pygame.draw.rect(self._screen, _LAUNCH_HOVER if hov else _LAUNCH_BG,
                             launch_rect, border_radius=6)
            pygame.draw.rect(self._screen, (60, 200, 100), launch_rect, 1, border_radius=6)
            surf = self._font_md.render("LAUNCH", True, _TEXT)
        else:
            pygame.draw.rect(self._screen, _BTN_DISABLED, launch_rect, border_radius=6)
            pygame.draw.rect(self._screen, (50, 50, 60), launch_rect, 1, border_radius=6)
            surf = self._font_md.render("LAUNCH", True, _TEXT_DIM)
        self._screen.blit(surf, surf.get_rect(center=launch_rect.center))

        # QUIT
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
