"""
Pygame 2D renderer for the AGV fleet environment.

Draws the plant grid, AGV positions (color-coded by status),
task markers, and a live metrics panel on the right side.

Cell sprites are loaded from src/rendering/assets/cells/ as PNG files.
To update sprites, replace the PNGs and restart — no code changes needed.

Controls (handled by the caller via handle_events()):
  SPACE       — pause / resume
  UP / DOWN   — increase / decrease simulation speed
  R           — reset episode (signal returned via handle_events)
  ESC / Q     — quit
"""

import os
from typing import Dict, List, Tuple

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
_COLOR_GRID_LINE     = (200, 200, 200)
_COLOR_PANEL_BG      = ( 20,  20,  20)
_COLOR_TEXT          = (240, 240, 240)
_COLOR_TEXT_DIM      = (140, 140, 140)
_COLOR_TASK_PICKUP   = (255,  80,  80)
_COLOR_TASK_DELIVERY = ( 80, 220, 120)
_COLOR_ACCENT        = ( 90, 160, 255)

# Mapping from CellType to sprite filename (without extension)
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
      [ grid 680x680 ] [ metrics panel 280px ]
      Total window: 960 x 700 (with 10px top margin)

    Cell sprites are loaded from assets/cells/ at init time and pre-scaled
    to CELL_SIZE. To swap a sprite, replace the PNG file and restart.

    Usage:
        renderer = PygameRenderer()
        while renderer.handle_events() != "quit":
            action = agent.select_action(env)
            env.step(action)
            renderer.render(env)
            renderer.tick(fps=10)
        renderer.close()
    """

    CELL_SIZE   = 34
    TOP_MARGIN  = 10
    PANEL_WIDTH = 280
    GRID_PIXELS = PlantMap.GRID_SIZE * CELL_SIZE   # 680

    WINDOW_W = GRID_PIXELS + PANEL_WIDTH            # 960
    WINDOW_H = GRID_PIXELS + TOP_MARGIN             # 690

    def __init__(self, title: str = "AGV Fleet — RL vs A* Baseline"):
        pygame.init()
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        self._clock  = pygame.time.Clock()
        self._font_md  = pygame.font.SysFont("consolas", 14)
        self._font_sm  = pygame.font.SysFont("consolas", 12)
        self._font_lg  = pygame.font.SysFont("consolas", 18, bold=True)
        self._paused   = False

        # Pre-load and scale all cell sprites to CELL_SIZE once at startup
        self._cell_sprites: Dict[CellType, pygame.Surface] = {}
        for cell_type, name in _CELL_SPRITE_NAMES.items():
            path = os.path.join(_ASSETS_DIR, f"{name}.png")
            img  = pygame.image.load(path).convert()
            self._cell_sprites[cell_type] = pygame.transform.scale(
                img, (self.CELL_SIZE, self.CELL_SIZE)
            )

        # AGV base sprite (white/grey body — tinted at draw time by status color)
        # Loaded with convert_alpha() to preserve transparency in the PNG
        agv_path = os.path.join(_ASSETS_DIR, "..", "agv_base.png")
        agv_img  = pygame.image.load(agv_path).convert_alpha()
        self._agv_base: pygame.Surface = pygame.transform.scale(
            agv_img, (self.CELL_SIZE, self.CELL_SIZE)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_events(self) -> str:
        """
        Process the Pygame event queue.

        Returns
        -------
        str
            "quit"   — window closed or ESC/Q pressed
            "reset"  — R pressed
            "ok"     — nothing special
        """
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
        return "ok"

    @property
    def paused(self) -> bool:
        return self._paused

    def render(self, env: AGVFleetEnv) -> None:
        """Draw the current environment state to the screen."""
        self._screen.fill(_COLOR_BG)
        self._draw_grid(env.plant)
        self._draw_task_markers(env.tasks)
        self._draw_agvs(env.agvs)
        self._draw_panel(env)
        if self._paused:
            self._draw_paused_overlay()
        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self._clock.tick(fps)

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = col * self.CELL_SIZE
        y = row * self.CELL_SIZE + self.TOP_MARGIN
        return pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

    def _draw_grid(self, plant: PlantMap) -> None:
        for row in range(plant.GRID_SIZE):
            for col in range(plant.GRID_SIZE):
                cell_type = CellType(plant.grid[row, col])
                rect      = self._cell_rect(row, col)
                self._draw_cell_sprite(cell_type, rect)
                if cell_type != CellType.OBSTACLE:
                    pygame.draw.rect(self._screen, _COLOR_GRID_LINE, rect, 1)

    def _draw_cell_sprite(self, cell_type: CellType, rect: pygame.Rect) -> None:
        """Blits the pre-scaled PNG sprite for the given cell type."""
        surf = self._cell_sprites.get(cell_type)
        if surf:
            self._screen.blit(surf, rect.topleft)
        else:
            pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], rect)

    def _draw_cell_sprite_scaled(self, cell_type: CellType, rect: pygame.Rect) -> None:
        """Blits the sprite scaled to an arbitrary size (used in the legend panel)."""
        surf = self._cell_sprites.get(cell_type)
        if surf:
            scaled = pygame.transform.scale(surf, (rect.width, rect.height))
            self._screen.blit(scaled, rect.topleft)
        else:
            pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], rect)

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
            self._draw_battery_bar(rect, agv.battery)

    def _draw_agv_sprite(self, cell_rect: pygame.Rect, agv: AGV) -> None:
        """
        Draws an AGV using the base sprite tinted by the AGV's current status color.

        The base sprite (agv_base.png) uses:
          - White/grey for the body   → multiplied by status color
          - Black/dark for wheels and outlines → stays dark regardless of tint
        A cargo indicator (yellow rect) is drawn on top when the AGV has a task.
        """
        color = _AGV_STATUS_COLORS[agv.status]
        cx    = cell_rect.centerx
        cy    = cell_rect.centery

        # Copy base sprite and apply status color tint preserving alpha
        tinted = self._agv_base.copy()
        tinted.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self._screen.blit(tinted, cell_rect.topleft)

        # Cargo indicator: yellow rect in the center when carrying a task
        if agv.task_id is not None:
            cs     = self.CELL_SIZE
            cw     = cs - 14
            ch     = max(4, cs // 6)
            crect  = pygame.Rect(cell_rect.x + 7, cy - ch // 2, cw, ch)
            pygame.draw.rect(self._screen, (255, 220, 60), crect, border_radius=2)

        # ID label
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

    def _draw_panel(self, env: AGVFleetEnv) -> None:
        panel_x = self.GRID_PIXELS
        panel_rect = pygame.Rect(panel_x, 0, self.PANEL_WIDTH, self.WINDOW_H)
        pygame.draw.rect(self._screen, _COLOR_PANEL_BG, panel_rect)

        info = env._get_info()
        y    = 16

        def text(msg: str, color=_COLOR_TEXT, font=None) -> int:
            nonlocal y
            surf = (font or self._font_md).render(msg, True, color)
            self._screen.blit(surf, (panel_x + 12, y))
            y += surf.get_height() + 4
            return y

        def separator() -> None:
            nonlocal y
            pygame.draw.line(
                self._screen, (60, 60, 60),
                (panel_x + 8, y), (panel_x + self.PANEL_WIDTH - 8, y)
            )
            y += 10

        text("AGV Fleet Sim", color=_COLOR_ACCENT, font=self._font_lg)
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
        y += 20
        separator()

        text("CONTROLS", color=_COLOR_TEXT_DIM)
        for line in ["SPACE  pause / resume", "UP     faster",
                     "DOWN   slower", "R      reset episode", "ESC/Q  quit"]:
            text(f"  {line}", color=_COLOR_TEXT_DIM, font=self._font_sm)

    def _draw_paused_overlay(self) -> None:
        overlay = pygame.Surface((self.GRID_PIXELS, self.WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self._screen.blit(overlay, (0, 0))
        surf = self._font_lg.render("PAUSED — press SPACE", True, (255, 255, 255))
        self._screen.blit(surf, surf.get_rect(
            center=(self.GRID_PIXELS // 2, self.WINDOW_H // 2)
        ))
