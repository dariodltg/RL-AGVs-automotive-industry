"""
Pygame 2D renderer for the AGV fleet environment.

Draws the plant grid, AGV positions (color-coded by status),
task markers, and a live metrics panel on the right side.

Controls (handled by the caller via handle_events()):
  SPACE       — pause / resume
  UP / DOWN   — increase / decrease simulation speed
  R           — reset episode (signal returned via handle_events)
  ESC / Q     — quit
"""

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
    CellType.ENTRY:    ( 50, 180, 100),   # green  — raw material input
    CellType.STAMPING: ( 70, 130, 200),   # steel blue — stamping presses
    CellType.BUFFER:   (180, 130,  50),   # amber  — WIP storage
    CellType.WELDING:  (200,  70,  70),   # red    — welding stations
    CellType.EXIT:     (120,  60, 200),   # purple — finished output
    CellType.CHARGING: (210, 200,  30),   # yellow — AGV charging
}

_AGV_STATUS_COLORS: Dict[AGVStatus, Tuple[int, int, int]] = {
    AGVStatus.IDLE:               (160, 160, 160),   # gray
    AGVStatus.MOVING_TO_PICKUP:   (255, 140,   0),   # orange
    AGVStatus.LOADING:            ( 40, 200,  40),   # green
    AGVStatus.MOVING_TO_DELIVERY: (220,  60,  60),   # red
    AGVStatus.UNLOADING:          (  0, 190, 190),   # cyan
    AGVStatus.CHARGING:           (220, 210,   0),   # yellow
}

_COLOR_BG         = ( 30,  30,  30)
_COLOR_GRID_LINE  = (200, 200, 200)
_COLOR_PANEL_BG   = ( 20,  20,  20)
_COLOR_TEXT       = (240, 240, 240)
_COLOR_TEXT_DIM   = (140, 140, 140)
_COLOR_TASK_PICKUP   = (255, 80,  80)   # red marker
_COLOR_TASK_DELIVERY = ( 80, 220, 120)  # green marker
_COLOR_ACCENT     = ( 90, 160, 255)


class PygameRenderer:
    """
    Real-time 2D renderer for AGVFleetEnv using Pygame.

    Layout:
      [ grid 680x680 ] [ metrics panel 280px ]
      Total window: 960 x 700 (with 10px top margin)

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
                color     = _CELL_COLORS[cell_type]
                rect      = self._cell_rect(row, col)
                pygame.draw.rect(self._screen, color, rect)
                # subtle grid lines only on free/non-obstacle cells
                if cell_type != CellType.OBSTACLE:
                    pygame.draw.rect(self._screen, _COLOR_GRID_LINE, rect, 1)

        # Cell type sprites
        for row in range(plant.GRID_SIZE):
            for col in range(plant.GRID_SIZE):
                cell_type = CellType(plant.grid[row, col])
                rect = self._cell_rect(row, col)
                self._draw_cell_sprite(cell_type, rect)

    def _draw_cell_sprite(self, cell_type: CellType, rect: pygame.Rect) -> None:
        """Dispatches to the correct sprite drawing method for a given cell type."""
        _sprite_dispatch = {
            CellType.ENTRY:    self._sprite_entry,
            CellType.STAMPING: self._sprite_stamping,
            CellType.BUFFER:   self._sprite_buffer,
            CellType.WELDING:  self._sprite_welding,
            CellType.EXIT:     self._sprite_exit,
            CellType.CHARGING: self._sprite_charging,
        }
        fn = _sprite_dispatch.get(cell_type)
        if fn:
            fn(rect)

    def _sprite_charging(self, cell_rect: pygame.Rect) -> None:
        """
        Charging station sprite:
          - Base platform at the bottom
          - Vertical charging post
          - Glow halo + lightning bolt
        """
        cs = self.CELL_SIZE
        cx = cell_rect.centerx
        cy = cell_rect.centery

        # Base platform
        platform = pygame.Rect(cell_rect.x + 4, cell_rect.bottom - 9, cs - 8, 6)
        pygame.draw.rect(self._screen, (60, 50, 20), platform, border_radius=2)
        pygame.draw.rect(self._screen, (180, 140, 40), platform, 1, border_radius=2)

        # Vertical post
        post_rect = pygame.Rect(cx - 2, cell_rect.y + 5, 4, cs - 14)
        pygame.draw.rect(self._screen, (120, 100, 30), post_rect, border_radius=2)

        # Glow halo
        glow_surf = pygame.Surface((18, 18), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 210, 0, 60), (9, 9), 9)
        self._screen.blit(glow_surf, (cx - 9, cy - 11))

        # Lightning bolt
        bx, by = cx, cy - 9
        bolt_points = [
            (bx + 3, by),      (bx,     by + 6),
            (bx + 3, by + 6),  (bx - 3, by + 13),
            (bx,     by + 7),  (bx - 3, by + 7),
        ]
        pygame.draw.polygon(self._screen, (255, 220, 0), bolt_points)
        pygame.draw.polygon(self._screen, (200, 160, 0), bolt_points, 1)

    def _sprite_entry(self, cell_rect: pygame.Rect) -> None:
        """
        ENTRY cell sprite — loading dock with inbound arrow:
          - Green arch frame
          - Down arrow (material entering)
          - Dashed threshold line at bottom
        """
        cs = self.CELL_SIZE
        cx = cell_rect.centerx
        x, y = cell_rect.x, cell_rect.y

        arch_rect = pygame.Rect(x + 5, y + 4, cs - 10, cs - 8)
        pygame.draw.rect(self._screen, (20, 80, 40), arch_rect, border_radius=6)
        pygame.draw.rect(self._screen, (100, 230, 130), arch_rect, 2, border_radius=6)
        inner = pygame.Rect(x + 9, y + 8, cs - 18, cs - 14)
        pygame.draw.rect(self._screen, (10, 40, 20), inner, border_radius=3)

        # Down arrow (material coming IN)
        arrow_color = (120, 255, 150)
        dn_tip = (cx, y + cs - 8)
        pygame.draw.polygon(self._screen, arrow_color, [
            dn_tip, (cx - 4, y + cs - 15), (cx + 4, y + cs - 15)
        ])
        pygame.draw.line(self._screen, arrow_color, (cx, y + 10), (cx, y + cs - 15), 2)

        dash_y = cell_rect.bottom - 3
        for i in range(3):
            dx = x + 7 + i * 8
            pygame.draw.line(self._screen, (100, 230, 130), (dx, dash_y), (dx + 4, dash_y), 2)

    def _sprite_stamping(self, cell_rect: pygame.Rect) -> None:
        """
        STAMPING cell sprite — stamping press:
          - Dark frame (press body)
          - Horizontal press plate in the center
          - Downward force arrows suggesting press action
        """
        cs = self.CELL_SIZE
        cx = cell_rect.centerx
        x, y = cell_rect.x, cell_rect.y

        # Press body
        frame = pygame.Rect(x + 4, y + 4, cs - 8, cs - 8)
        pygame.draw.rect(self._screen, (25, 55, 100), frame, border_radius=2)
        pygame.draw.rect(self._screen, (120, 170, 240), frame, 1, border_radius=2)

        # Top cross-beam
        pygame.draw.rect(self._screen, (80, 130, 200),
                         pygame.Rect(x + 5, y + 6, cs - 10, 4))

        # Press plate (lower, thicker bar)
        plate_y = y + cs // 2
        pygame.draw.rect(self._screen, (100, 150, 220),
                         pygame.Rect(x + 6, plate_y, cs - 12, 5))

        # Force arrows (small downward chevrons)
        for ax in (cx - 5, cx + 2):
            pygame.draw.polygon(self._screen, (180, 210, 255), [
                (ax, plate_y - 6), (ax + 3, plate_y - 2), (ax + 6, plate_y - 6)
            ])

    def _sprite_buffer(self, cell_rect: pygame.Rect) -> None:
        """
        BUFFER / WIP cell sprite — storage rack:
          - Amber background frame
          - 2 shelf lines
          - Small boxes on shelves
        """
        cs = self.CELL_SIZE
        x, y = cell_rect.x, cell_rect.y

        frame = pygame.Rect(x + 4, y + 4, cs - 8, cs - 8)
        pygame.draw.rect(self._screen, (90, 60, 20), frame, border_radius=2)
        pygame.draw.rect(self._screen, (220, 170, 60), frame, 1, border_radius=2)

        # Shelf lines
        for i in range(2):
            shelf_y = y + 11 + i * 9
            pygame.draw.line(self._screen, (200, 150, 50),
                             (x + 6, shelf_y), (x + cs - 6, shelf_y), 1)
            # Boxes on shelf
            for bx in (x + 7, x + 13, x + 19):
                pygame.draw.rect(self._screen, (240, 190, 80),
                                 pygame.Rect(bx, shelf_y - 5, 4, 5))

    def _sprite_welding(self, cell_rect: pygame.Rect) -> None:
        """
        WELDING cell sprite — welding station:
          - Dark red frame
          - Welding torch arm
          - Spark burst at tip
        """
        cs = self.CELL_SIZE
        cx = cell_rect.centerx
        cy = cell_rect.centery
        x, y = cell_rect.x, cell_rect.y

        frame = pygame.Rect(x + 4, y + 4, cs - 8, cs - 8)
        pygame.draw.rect(self._screen, (100, 25, 25), frame, border_radius=2)
        pygame.draw.rect(self._screen, (240, 100, 100), frame, 1, border_radius=2)

        # Torch arm (diagonal line)
        torch_start = (x + 7, y + 7)
        torch_end   = (cx + 3, cy + 3)
        pygame.draw.line(self._screen, (200, 80, 80), torch_start, torch_end, 3)
        pygame.draw.circle(self._screen, (220, 100, 80), torch_end, 3)

        # Spark burst at tip
        spark_color = (255, 220, 60)
        for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4), (3, 3), (-3, -3)):
            sx, sy = torch_end[0] + dx, torch_end[1] + dy
            pygame.draw.line(self._screen, spark_color, torch_end, (sx, sy), 1)

    def _sprite_exit(self, cell_rect: pygame.Rect) -> None:
        """
        EXIT cell sprite — outbound dock with upward arrow:
          - Purple arch frame
          - Up arrow (finished parts going OUT)
          - Dashed threshold line
        """
        cs = self.CELL_SIZE
        cx = cell_rect.centerx
        x, y = cell_rect.x, cell_rect.y

        arch_rect = pygame.Rect(x + 5, y + 4, cs - 10, cs - 8)
        pygame.draw.rect(self._screen, (60, 25, 110), arch_rect, border_radius=6)
        pygame.draw.rect(self._screen, (180, 100, 255), arch_rect, 2, border_radius=6)
        inner = pygame.Rect(x + 9, y + 8, cs - 18, cs - 14)
        pygame.draw.rect(self._screen, (30, 10, 60), inner, border_radius=3)

        # Up arrow (finished parts going OUT)
        arrow_color = (200, 150, 255)
        up_tip = (cx, y + 8)
        pygame.draw.polygon(self._screen, arrow_color, [
            up_tip, (cx - 4, y + 15), (cx + 4, y + 15)
        ])
        pygame.draw.line(self._screen, arrow_color, (cx, y + 15), (cx, y + cs - 8), 2)

        dash_y = cell_rect.bottom - 3
        for i in range(3):
            dx = x + 7 + i * 8
            pygame.draw.line(self._screen, (180, 100, 255), (dx, dash_y), (dx + 4, dash_y), 2)

    def _draw_cell_sprite_scaled(self, cell_type: CellType, rect: pygame.Rect) -> None:
        """
        Draws a scaled-down cell sprite for the legend panel.
        Renders to a temporary surface then blits onto the screen.
        """
        size = rect.width
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx   = size // 2

        if cell_type == CellType.ENTRY:
            pygame.draw.rect(surf, (20, 80, 40),
                             pygame.Rect(2, 2, size - 4, size - 4), border_radius=3)
            pygame.draw.rect(surf, (100, 230, 130),
                             pygame.Rect(2, 2, size - 4, size - 4), 1, border_radius=3)
            # Down arrow
            pygame.draw.polygon(surf, (120, 255, 150),
                                [(cx, size - 4), (cx - 3, size - 9), (cx + 3, size - 9)])
            pygame.draw.line(surf, (120, 255, 150), (cx, 4), (cx, size - 9), 1)

        elif cell_type == CellType.STAMPING:
            pygame.draw.rect(surf, (25, 55, 100),
                             pygame.Rect(2, 2, size - 4, size - 4), border_radius=2)
            pygame.draw.rect(surf, (120, 170, 240),
                             pygame.Rect(2, 2, size - 4, size - 4), 1, border_radius=2)
            pygame.draw.rect(surf, (80, 130, 200), pygame.Rect(3, 4, size - 6, 3))
            pygame.draw.rect(surf, (100, 150, 220),
                             pygame.Rect(4, size // 2, size - 8, 3))

        elif cell_type == CellType.BUFFER:
            pygame.draw.rect(surf, (90, 60, 20),
                             pygame.Rect(2, 2, size - 4, size - 4), border_radius=2)
            pygame.draw.rect(surf, (220, 170, 60),
                             pygame.Rect(2, 2, size - 4, size - 4), 1, border_radius=2)
            for i in range(2):
                sy = 8 + i * 7
                pygame.draw.line(surf, (200, 150, 50), (3, sy), (size - 3, sy), 1)
                for bx in (4, 9, 14):
                    pygame.draw.rect(surf, (240, 190, 80),
                                     pygame.Rect(bx, sy - 4, 3, 4))

        elif cell_type == CellType.WELDING:
            pygame.draw.rect(surf, (100, 25, 25),
                             pygame.Rect(2, 2, size - 4, size - 4), border_radius=2)
            pygame.draw.rect(surf, (240, 100, 100),
                             pygame.Rect(2, 2, size - 4, size - 4), 1, border_radius=2)
            tip = (cx + 2, cx + 2)
            pygame.draw.line(surf, (200, 80, 80), (4, 4), tip, 2)
            for dx, dy in ((3, 0), (-3, 0), (0, 3), (2, 2)):
                pygame.draw.line(surf, (255, 220, 60),
                                 tip, (tip[0] + dx, tip[1] + dy), 1)

        elif cell_type == CellType.EXIT:
            pygame.draw.rect(surf, (60, 25, 110),
                             pygame.Rect(2, 2, size - 4, size - 4), border_radius=3)
            pygame.draw.rect(surf, (180, 100, 255),
                             pygame.Rect(2, 2, size - 4, size - 4), 1, border_radius=3)
            # Up arrow
            pygame.draw.polygon(surf, (200, 150, 255),
                                [(cx, 4), (cx - 3, 9), (cx + 3, 9)])
            pygame.draw.line(surf, (200, 150, 255), (cx, 9), (cx, size - 4), 1)

        elif cell_type == CellType.CHARGING:
            pygame.draw.rect(surf, (120, 100, 30),
                             pygame.Rect(cx - 1, 2, 3, size - 8), border_radius=1)
            bx, by = cx, cx - 4
            bolt = [
                (bx + 2, by),     (bx,     by + 4),
                (bx + 2, by + 4), (bx - 2, by + 8),
                (bx,     by + 4), (bx - 2, by + 4),
            ]
            pygame.draw.polygon(surf, (255, 220, 0), bolt)
            pygame.draw.rect(surf, (180, 140, 40),
                             pygame.Rect(2, size - 5, size - 4, 3), border_radius=1)

        self._screen.blit(surf, rect.topleft)

    def _draw_task_markers(self, tasks: List[Task]) -> None:
        for task in tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                # Pickup: red circle
                rect = self._cell_rect(*task.pickup)
                pygame.draw.circle(
                    self._screen, _COLOR_TASK_PICKUP,
                    rect.center, self.CELL_SIZE // 4
                )
            if task.status in (TaskStatus.IN_PROGRESS,):
                # Delivery target: green circle outline
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
        Draws an AGV as a small forklift-style vehicle sprite using Pygame primitives.

        Layout (fits within CELL_SIZE x CELL_SIZE):
          - Chassis: rounded rectangle, color-coded by status
          - 4 wheels: dark rounded squares at corners
          - Fork / cargo indicator: front protrusion (top of cell = north)
          - Status stripe: thin colored bar on the roof
          - ID label: centered on chassis
        """
        cs   = self.CELL_SIZE
        cx   = cell_rect.centerx
        cy   = cell_rect.centery
        color = _AGV_STATUS_COLORS[agv.status]

        # --- dimensions ---
        body_w = cs - 8
        body_h = cs - 10
        body_x = cell_rect.x + 4
        body_y = cell_rect.y + 5

        wheel_size = 5
        wheel_color = (30, 30, 30)

        # --- chassis shadow ---
        shadow_rect = pygame.Rect(body_x + 2, body_y + 2, body_w, body_h)
        pygame.draw.rect(self._screen, (20, 20, 20), shadow_rect, border_radius=4)

        # --- chassis body ---
        body_rect = pygame.Rect(body_x, body_y, body_w, body_h)
        pygame.draw.rect(self._screen, color, body_rect, border_radius=4)

        # --- chassis outline ---
        pygame.draw.rect(self._screen, (255, 255, 255), body_rect, 1, border_radius=4)

        # --- roof stripe (status color darkened) ---
        stripe_color = tuple(max(0, c - 60) for c in color)
        stripe_rect  = pygame.Rect(body_x + 4, body_y + 2, body_w - 8, 4)
        pygame.draw.rect(self._screen, stripe_color, stripe_rect, border_radius=2)

        # --- 4 wheels ---
        wheel_offsets = [
            (body_x,                   body_y),                    # top-left
            (body_x + body_w - wheel_size, body_y),                # top-right
            (body_x,                   body_y + body_h - wheel_size),  # bottom-left
            (body_x + body_w - wheel_size, body_y + body_h - wheel_size),  # bottom-right
        ]
        for wx, wy in wheel_offsets:
            pygame.draw.rect(self._screen, wheel_color,
                             pygame.Rect(wx, wy, wheel_size, wheel_size),
                             border_radius=2)

        # --- fork prongs at the top (direction indicator) ---
        fork_color = (200, 200, 200)
        prong_y    = body_y - 3
        for prong_x in (cx - 5, cx + 2):
            pygame.draw.rect(self._screen, fork_color,
                             pygame.Rect(prong_x, prong_y, 3, 5))

        # --- cargo indicator: filled rect when carrying a task ---
        if agv.task_id is not None:
            cargo_rect = pygame.Rect(body_x + 6, body_y + body_h // 2 - 3, body_w - 12, 8)
            pygame.draw.rect(self._screen, (255, 220, 80), cargo_rect, border_radius=2)

        # --- AGV id label ---
        label = self._font_sm.render(str(agv.id), True, (10, 10, 10))
        label_rect = label.get_rect(center=(cx, cy + 4))
        self._screen.blit(label, label_rect)

    def _draw_battery_bar(self, cell_rect: pygame.Rect, battery: float) -> None:
        bar_w  = self.CELL_SIZE - 6
        bar_h  = 4
        bar_x  = cell_rect.x + 3
        bar_y  = cell_rect.bottom - bar_h - 2
        # Background
        pygame.draw.rect(self._screen, (60, 60, 60),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        # Fill color based on charge level
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

        # Title
        text("AGV Fleet Sim", color=_COLOR_ACCENT, font=self._font_lg)
        separator()

        # Episode metrics
        text("METRICS", color=_COLOR_TEXT_DIM)
        text(f"  Step          {info['step']:>6}")
        text(f"  Tasks done    {info['tasks_completed']:>6}")
        text(f"  Tasks pending {info['tasks_pending']:>6}")
        text(f"  Collisions    {info['collisions']:>6}")
        text(f"  Avg cycle     {info['avg_cycle_time']:>6.1f} steps")
        text(f"  Mean util.    {info['mean_utilization']:>6.1%}")
        separator()

        # Per-AGV status
        text("AGVs", color=_COLOR_TEXT_DIM)
        for agv in env.agvs:
            status_name = agv.status.name.replace("_", " ")
            color = _AGV_STATUS_COLORS[agv.status]
            label = f"  AGV {agv.id}  {status_name:<20}"
            surf  = self._font_sm.render(label, True, color)
            self._screen.blit(surf, (panel_x + 12, y))
            y += surf.get_height() + 3
            # battery %
            bat_surf = self._font_sm.render(
                f"         bat {agv.battery:.0%}", True, _COLOR_TEXT_DIM
            )
            self._screen.blit(bat_surf, (panel_x + 12, y))
            y += bat_surf.get_height() + 4
        separator()

        # Cell legend — mini sprites
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
            icon_x = panel_x + 12
            icon_rect = pygame.Rect(icon_x, y, icon_size, icon_size)
            # Background fill
            pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], icon_rect, border_radius=2)
            pygame.draw.rect(self._screen, (80, 80, 80), icon_rect, 1, border_radius=2)
            # Sprite drawn at icon scale
            if cell_type != CellType.OBSTACLE:
                self._draw_cell_sprite_scaled(cell_type, icon_rect)
            surf = self._font_sm.render(f"  {label}", True, _COLOR_TEXT)
            self._screen.blit(surf, (icon_x + icon_size + 4, y + 3))
            y += icon_size + 4
        separator()

        # Status legend
        text("AGV STATUS", color=_COLOR_TEXT_DIM)
        for status, color in _AGV_STATUS_COLORS.items():
            label = status.name.replace("_", " ").title()
            pygame.draw.circle(self._screen, color,
                               (panel_x + 18, y + 6), 5)
            surf = self._font_sm.render(f"      {label}", True, _COLOR_TEXT)
            self._screen.blit(surf, (panel_x + 12, y))
            y += 16
        separator()

        # Pickup / delivery markers legend
        pygame.draw.circle(self._screen, _COLOR_TASK_PICKUP,
                           (panel_x + 18, y + 6), 5)
        self._font_sm.render("      Pickup", True, _COLOR_TEXT)
        surf = self._font_sm.render("      Task pickup", True, _COLOR_TEXT)
        self._screen.blit(surf, (panel_x + 12, y)); y += 16

        pygame.draw.circle(self._screen, _COLOR_TASK_DELIVERY,
                           (panel_x + 18, y + 6), 5, 2)
        surf = self._font_sm.render("      Task delivery", True, _COLOR_TEXT)
        self._screen.blit(surf, (panel_x + 12, y)); y += 20
        separator()

        # Controls
        text("CONTROLS", color=_COLOR_TEXT_DIM)
        controls = [
            "SPACE  pause / resume",
            "UP     faster",
            "DOWN   slower",
            "R      reset episode",
            "ESC/Q  quit",
        ]
        for line in controls:
            text(f"  {line}", color=_COLOR_TEXT_DIM, font=self._font_sm)

    def _draw_paused_overlay(self) -> None:
        overlay = pygame.Surface((self.GRID_PIXELS, self.WINDOW_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self._screen.blit(overlay, (0, 0))
        surf = self._font_lg.render("PAUSED — press SPACE", True, (255, 255, 255))
        self._screen.blit(surf, surf.get_rect(
            center=(self.GRID_PIXELS // 2, self.WINDOW_H // 2)
        ))
