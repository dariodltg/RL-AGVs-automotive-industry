"""
Pygame 2D renderer for the AGV fleet environment.

Layout:
  [ left panel 160px ] [ grid 680px ] [ right panel 280px ]
  Total window: 1120 x 690

Left panel  — interactive controls (FPS, pause, reset, display toggles)
Grid        — 20x20 plant with cell sprites, AGVs, task markers
Right panel — live metrics, AGV status, cell legend

Cell sprites are loaded from src/rendering/assets/cells/ as PNG files.
AGV base sprite loaded from src/rendering/assets/agv_base.png and tinted
per-frame using BLEND_RGBA_MULT.

Controls (keyboard shortcuts still work):
  SPACE       — pause / resume
  UP / DOWN   — increase / decrease FPS
  R           — reset episode
  ESC / Q     — quit
"""

import os
from typing import Dict, List, Optional, Tuple

import pygame

from src.env.agv_fleet_env import AGV, AGVStatus, AGVFleetEnv, Task, TaskStatus
from src.env.plant_map import CellType, PlantMap


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_CELL_COLORS: Dict[CellType, Tuple[int, int, int]] = {
    CellType.FREE:     (210, 210, 210),
    CellType.OBSTACLE: ( 45,  45,  45),
    CellType.ENTRY:    ( 50, 180, 100),
    CellType.STAMPING: ( 70, 130, 200),
    CellType.BUFFER:   (180, 130,  50),
    CellType.WELDING:  (200,  70,  70),
    CellType.EXIT:     (120,  60, 200),
    CellType.CHARGING: (210, 200,  30),
}

_AGV_STATUS_COLORS: Dict[AGVStatus, Tuple[int, int, int]] = {
    AGVStatus.IDLE:               (160, 160, 160),
    AGVStatus.MOVING_TO_PICKUP:   (255, 140,   0),
    AGVStatus.LOADING:            ( 40, 200,  40),
    AGVStatus.MOVING_TO_DELIVERY: (220,  60,  60),
    AGVStatus.UNLOADING:          (  0, 190, 190),
    AGVStatus.CHARGING:           (220, 210,   0),
}

_COLOR_BG            = ( 30,  30,  30)
_COLOR_PANEL_BG      = ( 22,  22,  28)
_COLOR_GRID_LINE     = (200, 200, 200)
_COLOR_TEXT          = (240, 240, 240)
_COLOR_TEXT_DIM      = (140, 140, 140)
_COLOR_TASK_PICKUP   = (255,  80,  80)
_COLOR_TASK_DELIVERY = ( 80, 220, 120)
_COLOR_ACCENT        = ( 90, 160, 255)
_COLOR_BTN_BG        = ( 45,  45,  55)
_COLOR_BTN_HOVER     = ( 65,  65,  80)
_COLOR_BTN_ACTIVE    = ( 40, 120, 200)
_COLOR_BTN_BORDER    = ( 80,  80, 100)

_CELL_SPRITE_NAMES: Dict[CellType, str] = {
    CellType.FREE:     "free",
    CellType.OBSTACLE: "obstacle",
    CellType.ENTRY:    "entry",
    CellType.STAMPING: "stamping",
    CellType.BUFFER:   "buffer",
    CellType.WELDING:  "welding",
    CellType.EXIT:     "exit",
    CellType.CHARGING: "charging",
}

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cells")


class PygameRenderer:
    """
    Real-time 2D renderer for AGVFleetEnv using Pygame.

    Layout:
      [ left panel 160px ] [ grid 680px ] [ right panel 280px ]
      Total window: 1120 x 690

    Usage:
        renderer = PygameRenderer(fps=10)
        while renderer.handle_events() != "quit":
            env.step(agent.select_action(env))
            renderer.render(env)
            renderer.tick()   # uses renderer.fps internally
        renderer.close()
    """

    LEFT_PANEL_W = 160
    CELL_SIZE    = 34
    TOP_MARGIN   = 10
    RIGHT_PANEL_W = 280
    GRID_PIXELS  = PlantMap.GRID_SIZE * CELL_SIZE   # 680

    WINDOW_W = LEFT_PANEL_W + GRID_PIXELS + RIGHT_PANEL_W   # 1120
    WINDOW_H = GRID_PIXELS + TOP_MARGIN                      # 690

    # Grid starts at x = LEFT_PANEL_W
    GRID_X = LEFT_PANEL_W

    def __init__(self, title: str = "AGV Fleet — RL vs A* Baseline", fps: int = 10,
                 agent_label: str = ""):
        pygame.init()
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        self._clock  = pygame.time.Clock()
        self._font_md  = pygame.font.SysFont("consolas", 14)
        self._font_sm  = pygame.font.SysFont("consolas", 12)
        self._font_lg  = pygame.font.SysFont("consolas", 18, bold=True)
        self._font_xs  = pygame.font.SysFont("consolas", 11)

        # --- Simulation state ---
        self._paused     = False
        self.fps         = fps
        self.episode     = 1
        self.agent_label = agent_label

        # --- Display toggles ---
        self.show_task_markers = True
        self.show_grid_lines   = True
        self.show_agv_ids      = True
        self.show_battery_bars = True

        # --- Cell sprites ---
        self._cell_sprites: Dict[CellType, pygame.Surface] = {}
        for cell_type, name in _CELL_SPRITE_NAMES.items():
            path = os.path.join(_ASSETS_DIR, f"{name}.png")
            img  = pygame.image.load(path).convert()
            self._cell_sprites[cell_type] = pygame.transform.scale(
                img, (self.CELL_SIZE, self.CELL_SIZE)
            )

        # --- AGV sprite ---
        agv_path = os.path.join(_ASSETS_DIR, "..", "agv_base.png")
        agv_img  = pygame.image.load(agv_path).convert_alpha()
        self._agv_base: pygame.Surface = pygame.transform.scale(
            agv_img, (self.CELL_SIZE, self.CELL_SIZE)
        )
        self._agv_heading: Dict[int, Tuple[int, int]] = {}
        self._heading_angle: Dict[Tuple[int, int], float] = {
            ( 1,  0):   0,
            (-1,  0): 180,
            ( 0,  1): -90,
            ( 0, -1):  90,
        }

        # --- Left panel buttons (built on first render) ---
        self._buttons: List[Dict] = []
        self._build_buttons()

    # ------------------------------------------------------------------
    # Button definitions
    # ------------------------------------------------------------------

    def _build_buttons(self) -> None:
        """Define all left panel buttons with their positions and actions."""
        px = 10
        bw = self.LEFT_PANEL_W - 20   # 140px wide
        bh = 28

        def y(row: int) -> int:
            return 16 + row * (bh + 8)

        self._buttons = [
            # FPS controls
            {
                "label": "FPS  -",
                "rect":  pygame.Rect(px, y(2), bw // 2 - 2, bh),
                "action": "fps_down",
                "toggle": False,
            },
            {
                "label": "FPS  +",
                "rect":  pygame.Rect(px + bw // 2 + 2, y(2), bw // 2 - 2, bh),
                "action": "fps_up",
                "toggle": False,
            },
            # Pause
            {
                "label": "PAUSE",
                "rect":  pygame.Rect(px, y(4), bw, bh),
                "action": "pause",
                "toggle": True,
                "state_attr": "_paused",
            },
            # Reset
            {
                "label": "RESET",
                "rect":  pygame.Rect(px, y(5), bw, bh),
                "action": "reset",
                "toggle": False,
            },
            # Display toggles
            {
                "label": "Task markers",
                "rect":  pygame.Rect(px, y(7), bw, bh),
                "action": "toggle_markers",
                "toggle": True,
                "state_attr": "show_task_markers",
            },
            {
                "label": "Grid lines",
                "rect":  pygame.Rect(px, y(8), bw, bh),
                "action": "toggle_grid",
                "toggle": True,
                "state_attr": "show_grid_lines",
            },
            {
                "label": "AGV IDs",
                "rect":  pygame.Rect(px, y(9), bw, bh),
                "action": "toggle_ids",
                "toggle": True,
                "state_attr": "show_agv_ids",
            },
            {
                "label": "Battery bars",
                "rect":  pygame.Rect(px, y(10), bw, bh),
                "action": "toggle_battery",
                "toggle": True,
                "state_attr": "show_battery_bars",
            },
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_events(self) -> str:
        """
        Process the Pygame event queue and left-panel button clicks.

        Returns
        -------
        str  "quit" | "reset" | "ok"
        """
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return "quit"
                if event.key == pygame.K_r:
                    return "reset"
                if event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                if event.key == pygame.K_UP:
                    self.fps = min(self.fps + 1, 60)
                if event.key == pygame.K_DOWN:
                    self.fps = max(self.fps - 1, 1)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                result = self._handle_button_click(mouse_pos)
                if result == "reset":
                    return "reset"

        return "ok"

    def _handle_button_click(self, pos: Tuple[int, int]) -> Optional[str]:
        _toggle_attrs = {
            "toggle_markers": "show_task_markers",
            "toggle_grid":    "show_grid_lines",
            "toggle_ids":     "show_agv_ids",
            "toggle_battery": "show_battery_bars",
        }
        for btn in self._buttons:
            if not btn["rect"].collidepoint(pos):
                continue
            action = btn["action"]
            if action == "fps_up":
                self.fps = min(self.fps + 1, 60)
            elif action == "fps_down":
                self.fps = max(self.fps - 1, 1)
            elif action == "pause":
                self._paused = not self._paused
            elif action == "reset":
                return "reset"
            elif action in _toggle_attrs:
                attr = _toggle_attrs[action]
                setattr(self, attr, not getattr(self, attr))
        return None

    @property
    def paused(self) -> bool:
        return self._paused

    def render(self, env: AGVFleetEnv) -> None:
        """Draw the current environment state to the screen."""
        self._screen.fill(_COLOR_BG)
        self._draw_left_panel()
        self._draw_grid(env.plant)
        if self.show_task_markers:
            self._draw_task_markers(env.tasks)
        self._draw_agvs(env.agvs)
        self._draw_right_panel(env)
        if self._paused:
            self._draw_paused_overlay()
        pygame.display.flip()

    def tick(self, fps: Optional[int] = None) -> None:
        """Tick the clock. Uses renderer.fps if no fps argument given."""
        self._clock.tick(fps if fps is not None else self.fps)

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    def _draw_left_panel(self) -> None:
        panel_rect = pygame.Rect(0, 0, self.LEFT_PANEL_W, self.WINDOW_H)
        pygame.draw.rect(self._screen, _COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self._screen, (55, 55, 70),
                         (self.LEFT_PANEL_W - 1, 0),
                         (self.LEFT_PANEL_W - 1, self.WINDOW_H))

        mouse_pos = pygame.mouse.get_pos()

        # Title
        surf = self._font_sm.render("CONTROLS", True, _COLOR_TEXT_DIM)
        self._screen.blit(surf, (10, 16))

        # FPS label (above the -/+ buttons)
        fps_label = self._font_md.render(f"FPS: {self.fps:>2}", True, _COLOR_TEXT)
        self._screen.blit(fps_label, (10, 16 + 28 + 8))   # row 1

        # Separator labels
        sep_rows = {4: "SIMULATION", 7: "DISPLAY"}
        bh = 28
        for row, label in sep_rows.items():
            sy = 16 + row * (bh + 8) - 14
            surf = self._font_xs.render(label, True, _COLOR_TEXT_DIM)
            self._screen.blit(surf, (10, sy))

        # Draw buttons
        for btn in self._buttons:
            rect    = btn["rect"]
            hovered = rect.collidepoint(mouse_pos)

            # Background color
            if btn["toggle"] and getattr(self, btn.get("state_attr", ""), False):
                bg = _COLOR_BTN_ACTIVE
            elif hovered:
                bg = _COLOR_BTN_HOVER
            else:
                bg = _COLOR_BTN_BG

            pygame.draw.rect(self._screen, bg, rect, border_radius=4)
            pygame.draw.rect(self._screen, _COLOR_BTN_BORDER, rect, 1, border_radius=4)

            label_surf = self._font_sm.render(btn["label"], True, _COLOR_TEXT)
            self._screen.blit(label_surf, label_surf.get_rect(center=rect.center))

        # Keyboard hint at the bottom
        hints = ["SPACE pause", "R  reset", "ESC quit"]
        hy = self.WINDOW_H - len(hints) * 16 - 8
        for hint in hints:
            s = self._font_xs.render(hint, True, _COLOR_TEXT_DIM)
            self._screen.blit(s, (10, hy))
            hy += 16

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = self.GRID_X + col * self.CELL_SIZE
        y = row * self.CELL_SIZE + self.TOP_MARGIN
        return pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

    def _draw_grid(self, plant: PlantMap) -> None:
        for row in range(plant.GRID_SIZE):
            for col in range(plant.GRID_SIZE):
                cell_type = CellType(plant.grid[row, col])
                rect      = self._cell_rect(row, col)
                self._draw_cell_sprite(cell_type, rect)
                if self.show_grid_lines and cell_type != CellType.OBSTACLE:
                    pygame.draw.rect(self._screen, _COLOR_GRID_LINE, rect, 1)

    def _draw_cell_sprite(self, cell_type: CellType, rect: pygame.Rect) -> None:
        surf = self._cell_sprites.get(cell_type)
        if surf:
            self._screen.blit(surf, rect.topleft)
        else:
            pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], rect)

    def _draw_cell_sprite_scaled(self, cell_type: CellType, rect: pygame.Rect) -> None:
        surf = self._cell_sprites.get(cell_type)
        if surf:
            scaled = pygame.transform.scale(surf, (rect.width, rect.height))
            self._screen.blit(scaled, rect.topleft)
        else:
            pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], rect)

    # ------------------------------------------------------------------
    # Task markers & AGVs
    # ------------------------------------------------------------------

    def _draw_task_markers(self, tasks: List[Task]) -> None:
        for task in tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                rect = self._cell_rect(*task.pickup)
                pygame.draw.circle(
                    self._screen, _COLOR_TASK_PICKUP,
                    rect.center, self.CELL_SIZE // 4
                )
            if task.status in (TaskStatus.IN_PROGRESS,):
                rect = self._cell_rect(*task.delivery)
                pygame.draw.circle(
                    self._screen, _COLOR_TASK_DELIVERY,
                    rect.center, self.CELL_SIZE // 4, 2
                )

    def _draw_agvs(self, agvs: List[AGV]) -> None:
        for agv in agvs:
            rect = self._cell_rect(*agv.position)
            self._draw_agv_sprite(rect, agv)
            if self.show_battery_bars:
                self._draw_battery_bar(rect, agv.battery)

    def _draw_agv_sprite(self, cell_rect: pygame.Rect, agv: AGV) -> None:
        """
        Draws an AGV using the base sprite tinted by status color and rotated
        to face the direction of travel.
        """
        color = _AGV_STATUS_COLORS[agv.status]
        cx    = cell_rect.centerx
        cy    = cell_rect.centery

        if agv.path:
            next_pos = agv.path[0]
            heading  = (next_pos[0] - agv.position[0],
                        next_pos[1] - agv.position[1])
            if heading in self._heading_angle:
                self._agv_heading[agv.id] = heading

        heading = self._agv_heading.get(agv.id, (1, 0))
        angle   = self._heading_angle.get(heading, 0)

        tinted = self._agv_base.copy()
        tinted.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        rotated  = pygame.transform.rotate(tinted, angle)
        rot_rect = rotated.get_rect(center=cell_rect.center)
        self._screen.blit(rotated, rot_rect.topleft)

        if agv.task_id is not None:
            cs    = self.CELL_SIZE
            cw    = cs - 14
            ch    = max(4, cs // 6)
            crect = pygame.Rect(cell_rect.x + 7, cy - ch // 2, cw, ch)
            pygame.draw.rect(self._screen, (255, 220, 60), crect, border_radius=2)

        if self.show_agv_ids:
            label = self._font_sm.render(str(agv.id), True, (255, 255, 255))
            self._screen.blit(label, label.get_rect(center=(cx, cy + 4)))

    def _draw_battery_bar(self, cell_rect: pygame.Rect, battery: float) -> None:
        bar_w = self.CELL_SIZE - 6
        bar_h = 4
        bar_x = cell_rect.x + 3
        bar_y = cell_rect.bottom - bar_h - 2
        pygame.draw.rect(self._screen, (60, 60, 60),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        if battery > 0.5:
            fill_color = (80, 200, 80)
        elif battery > 0.2:
            fill_color = (220, 180, 0)
        else:
            fill_color = (220, 60, 60)
        pygame.draw.rect(self._screen, fill_color,
                         pygame.Rect(bar_x, bar_y, int(bar_w * battery), bar_h))

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _draw_right_panel(self, env: AGVFleetEnv) -> None:
        panel_x    = self.GRID_X + self.GRID_PIXELS
        panel_rect = pygame.Rect(panel_x, 0, self.RIGHT_PANEL_W, self.WINDOW_H)
        pygame.draw.rect(self._screen, _COLOR_PANEL_BG, panel_rect)
        pygame.draw.line(self._screen, (55, 55, 70),
                         (panel_x, 0), (panel_x, self.WINDOW_H))

        info = env._get_info()
        y    = 16

        def text(msg: str, color=_COLOR_TEXT, font=None) -> None:
            nonlocal y
            surf = (font or self._font_md).render(msg, True, color)
            self._screen.blit(surf, (panel_x + 12, y))
            y += surf.get_height() + 4

        def separator() -> None:
            nonlocal y
            pygame.draw.line(
                self._screen, (60, 60, 60),
                (panel_x + 8, y), (panel_x + self.RIGHT_PANEL_W - 8, y)
            )
            y += 10

        text("AGV Fleet Sim", color=_COLOR_ACCENT, font=self._font_lg)
        text(f"  Episode   {self.episode}", color=_COLOR_TEXT)
        if self.agent_label:
            text(f"  Agent     {self.agent_label}", color=_COLOR_TEXT_DIM)
        separator()

        text("METRICS", color=_COLOR_TEXT_DIM)
        text(f"  Step          {info['step']:>6}")
        text(f"  Tasks done    {info['tasks_completed']:>6}")
        text(f"  Tasks pending {info['tasks_pending']:>6}")
        text(f"  Collisions    {info['collisions']:>6}")
        text(f"  Avg cycle     {info['avg_cycle_time']:>6.1f} steps")
        text(f"  Mean util.    {info['mean_utilization']:>6.1%}")
        separator()

        text("AGVs", color=_COLOR_TEXT_DIM)
        for agv in env.agvs:
            status_name = agv.status.name.replace("_", " ")
            color = _AGV_STATUS_COLORS[agv.status]
            surf  = self._font_sm.render(f"  AGV {agv.id}  {status_name:<20}", True, color)
            self._screen.blit(surf, (panel_x + 12, y))
            y += surf.get_height() + 3
            bat_surf = self._font_sm.render(
                f"         bat {agv.battery:.0%}", True, _COLOR_TEXT_DIM
            )
            self._screen.blit(bat_surf, (panel_x + 12, y))
            y += bat_surf.get_height() + 4
        separator()

        text("CELL LEGEND", color=_COLOR_TEXT_DIM)
        legend_cells = [
            (CellType.ENTRY,    "Entry (raw material)"),
            (CellType.STAMPING, "Stamping press"),
            (CellType.BUFFER,   "Buffer / WIP"),
            (CellType.WELDING,  "Welding station"),
            (CellType.EXIT,     "Exit (finished)"),
            (CellType.CHARGING, "AGV Charging"),
            (CellType.OBSTACLE, "Obstacle"),
        ]
        icon_size = 20
        for cell_type, label in legend_cells:
            icon_x    = panel_x + 12
            icon_rect = pygame.Rect(icon_x, y, icon_size, icon_size)
            self._draw_cell_sprite_scaled(cell_type, icon_rect)
            pygame.draw.rect(self._screen, (80, 80, 80), icon_rect, 1, border_radius=2)
            surf = self._font_sm.render(f"  {label}", True, _COLOR_TEXT)
            self._screen.blit(surf, (icon_x + icon_size + 4, y + 3))
            y += icon_size + 4
        separator()

        text("AGV STATUS", color=_COLOR_TEXT_DIM)
        for status, color in _AGV_STATUS_COLORS.items():
            label = status.name.replace("_", " ").title()
            pygame.draw.circle(self._screen, color, (panel_x + 18, y + 6), 5)
            surf = self._font_sm.render(f"      {label}", True, _COLOR_TEXT)
            self._screen.blit(surf, (panel_x + 12, y))
            y += 16
        separator()

        pygame.draw.circle(self._screen, _COLOR_TASK_PICKUP, (panel_x + 18, y + 6), 5)
        surf = self._font_sm.render("      Task pickup", True, _COLOR_TEXT)
        self._screen.blit(surf, (panel_x + 12, y))
        y += 16

        pygame.draw.circle(self._screen, _COLOR_TASK_DELIVERY, (panel_x + 18, y + 6), 5, 2)
        surf = self._font_sm.render("      Task delivery", True, _COLOR_TEXT)
        self._screen.blit(surf, (panel_x + 12, y))

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _draw_paused_overlay(self) -> None:
        overlay = pygame.Surface(
            (self.GRID_PIXELS, self.WINDOW_H), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 100))
        self._screen.blit(overlay, (self.GRID_X, 0))
        surf = self._font_lg.render("PAUSED — press SPACE", True, (255, 255, 255))
        self._screen.blit(surf, surf.get_rect(
            center=(self.GRID_X + self.GRID_PIXELS // 2, self.WINDOW_H // 2)
        ))
