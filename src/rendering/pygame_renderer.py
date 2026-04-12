"""
Pygame 2D renderer for the AGV fleet environment.

Layout:
  [ left panel ] [ grid ] [ right panel ]

Render rate is fixed at RENDER_FPS (60). Simulation speed is controlled
separately via renderer.sim_speed (steps/sec). This decoupling enables:
  - Smooth AGV movement (pixel interpolation between cells)
  - Particle bursts on task completion

Cell sprites: src/rendering/assets/cells/*.png
AGV sprite:   src/rendering/assets/agv_base.png  (tinted per status)
"""

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pygame

from src.env.agv_fleet_env import AGV, AGVStatus, AGVFleetEnv, Task, TaskStatus
from src.env.plant_map import CellType, PlantMap


RENDER_FPS = 60   # render tick rate — always constant

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

# Particle colors per production stage
_STAGE_PARTICLE_COLORS: Dict[int, List[Tuple[int, int, int]]] = {
    0: [(120, 200, 255), (80, 160, 240), (200, 230, 255)],   # blue  — entry→stamping
    1: [(255, 190, 60),  (240, 150, 30), (255, 230, 140)],   # amber — stamping→buffer
    2: [(255, 80,  60),  (255, 140, 60), (255, 220, 80)],    # red   — buffer→welding
    3: [(200, 120, 255), (255, 255, 200),(160, 80,  255)],   # purple+white — welding→exit
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
_COLOR_DIVIDER       = ( 55,  55,  70)
_COLOR_DIVIDER_HOVER = ( 90, 160, 255)

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

_ASSETS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cells")
_DIVIDER_ZONE = 6
_LEFT_W_MIN   = 100
_RIGHT_W_MIN  = 180
_CELL_SIZE_MIN = 8
_TOP_MARGIN   = 10


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

@dataclass
class _Particle:
    x:     float
    y:     float
    vx:    float
    vy:    float
    life:  float
    decay: float
    color: Tuple[int, int, int]
    size:  float


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class PygameRenderer:
    """
    Real-time 2D renderer for AGVFleetEnv using Pygame.

    Decouples render rate (RENDER_FPS=60) from simulation speed (sim_speed).
    Use should_step() in the game loop to know when to advance the env.

    Usage:
        renderer = PygameRenderer(sim_speed=10)
        env.reset()
        renderer.reset_animation(env)

        while True:
            signal = renderer.handle_events()
            if signal == "quit": break
            if signal == "reset":
                env.reset()
                renderer.reset_animation(env)

            if not renderer.paused and renderer.should_step():
                renderer.pre_step(env)
                obs, r, term, trunc, info = env.step(action)
                renderer.notify_step(env)

            renderer.render(env)
            renderer.tick()
        renderer.close()
    """

    def __init__(
        self,
        title:       str = "AGV Fleet — RL vs A* Baseline",
        fps:         int = 10,
        agent_label: str = "",
        win_w:       int = 1500,
        win_h:       int = 980,
        left_w:      int = 220,
        right_w:     int = 320,
    ):
        pygame.init()
        pygame.display.set_caption(title)

        self._win_w   = win_w
        self._win_h   = win_h
        self._left_w  = left_w
        self._right_w = right_w

        self._screen = pygame.display.set_mode(
            (self._win_w, self._win_h), pygame.RESIZABLE
        )
        self._clock   = pygame.time.Clock()
        self._font_md = pygame.font.SysFont("consolas", 14)
        self._font_sm = pygame.font.SysFont("consolas", 12)
        self._font_lg = pygame.font.SysFont("consolas", 18, bold=True)
        self._font_xs = pygame.font.SysFont("consolas", 11)

        # Simulation / display state
        self._paused     = False
        self.fps         = fps       # sim steps per second
        self.episode     = 1
        self.agent_label = agent_label

        self.show_task_markers = True
        self.show_grid_lines   = True
        self.show_agv_ids      = True
        self.show_battery_bars = True

        # Divider drag state
        self._drag_left  = False
        self._drag_right = False

        # Sprite cache
        self._sprite_cache_size: int = -1
        self._cell_sprites: Dict[CellType, pygame.Surface] = {}
        self._agv_sprite:   Optional[pygame.Surface] = None

        # Raw (full-res) images
        self._raw_cell: Dict[CellType, pygame.Surface] = {}
        for cell_type, name in _CELL_SPRITE_NAMES.items():
            path = os.path.join(_ASSETS_DIR, f"{name}.png")
            self._raw_cell[cell_type] = pygame.image.load(path).convert()

        agv_path = os.path.join(_ASSETS_DIR, "..", "agv_base.png")
        self._raw_agv: pygame.Surface = pygame.image.load(agv_path).convert_alpha()

        # AGV heading / rotation
        self._agv_heading: Dict[int, Tuple[int, int]] = {}
        self._heading_angle: Dict[Tuple[int, int], float] = {
            ( 1,  0):   0,
            (-1,  0): 180,
            ( 0,  1): -90,
            ( 0, -1):  90,
        }

        # Smooth movement
        self._dt:              float = 0.0          # render delta-time (seconds)
        self._time_since_step: float = 0.0          # accumulator for sim stepping
        self._prev_pos:  Dict[int, Tuple[int, int]] = {}   # grid coords before last step
        self._anim_t:    Dict[int, float] = {}             # 0=prev cell, 1=curr cell

        # Particles
        self._particles:       List[_Particle] = []
        self._known_completed: Set[int]         = set()

        # Path preview — ids of AGVs with path overlay active
        self._selected_agvs: Set[int]  = set()
        self._last_agvs:     List[AGV] = []      # snapshot from last render()

        # Button hold-repeat (fps +/-)
        self._held_action: Optional[str] = None
        self._held_time:   float         = 0.0
        self._held_next:   float         = 0.0

        self._buttons: List[Dict] = []

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def _cell_size(self) -> int:
        avail_w = self._win_w - self._left_w - self._right_w
        avail_h = self._win_h - _TOP_MARGIN
        return max(_CELL_SIZE_MIN,
                   min(avail_w // PlantMap.GRID_SIZE,
                       avail_h // PlantMap.GRID_SIZE))

    @property
    def _grid_pixels(self) -> int:
        return PlantMap.GRID_SIZE * self._cell_size

    @property
    def _grid_x(self) -> int:
        return self._left_w

    @property
    def _right_x(self) -> int:
        return self._left_w + self._grid_pixels

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        cs = self._cell_size
        return pygame.Rect(self._grid_x + col * cs, row * cs + _TOP_MARGIN, cs, cs)

    # ------------------------------------------------------------------
    # Sprite cache
    # ------------------------------------------------------------------

    def _refresh_sprites(self) -> None:
        cs = self._cell_size
        if cs == self._sprite_cache_size:
            return
        self._sprite_cache_size = cs
        for ct, raw in self._raw_cell.items():
            self._cell_sprites[ct] = pygame.transform.scale(raw, (cs, cs))
        self._agv_sprite = pygame.transform.scale(self._raw_agv, (cs, cs))

    # ------------------------------------------------------------------
    # Simulation stepping protocol
    # ------------------------------------------------------------------

    def should_step(self) -> bool:
        """Returns True when enough render time has elapsed for a sim step."""
        return self._time_since_step >= 1.0 / max(1, self.fps)

    def pre_step(self, env: AGVFleetEnv) -> None:
        """Record AGV positions and delivery targets before env.step()."""
        for agv in env.agvs:
            self._prev_pos[agv.id] = agv.position

        # Snapshot: AGVs about to arrive this step (path len <= 1).
        # Delivery (MOVING_TO_DELIVERY) → particles at delivery cell, stage color
        # Pickup  (MOVING_TO_PICKUP)    → particles at pickup cell, white/grey burst
        self._pending_completions: List[Tuple[Tuple[int, int], int]] = []
        for agv in env.agvs:
            if len(agv.path) > 1:
                continue
            task = next((t for t in env.tasks if t.id == agv.task_id), None)
            if task is None:
                continue
            if agv.status == AGVStatus.MOVING_TO_DELIVERY:
                delivery = agv.path[0] if agv.path else agv.position
                self._pending_completions.append((delivery, task.stage))
            elif agv.status == AGVStatus.MOVING_TO_PICKUP:
                pickup = agv.path[0] if agv.path else agv.position
                self._pending_completions.append((pickup, -1))   # -1 = pickup event

    def notify_step(self, env: AGVFleetEnv) -> None:
        """
        Call after env.step(). Starts smooth movement animations and
        emits particles for completed tasks.
        """
        self._time_since_step -= 1.0 / max(1, self.fps)

        for agv in env.agvs:
            prev = self._prev_pos.get(agv.id, agv.position)
            self._anim_t[agv.id] = 0.0 if prev != agv.position else 1.0
            if prev != agv.position:
                dr = agv.position[0] - prev[0]
                dc = agv.position[1] - prev[1]
                if (dr, dc) in self._heading_angle:
                    self._agv_heading[agv.id] = (dr, dc)

        # Emit particles for completions detected in pre_step
        for delivery_pos, stage in getattr(self, "_pending_completions", []):
            self._emit_particles(delivery_pos, stage)
        self._pending_completions = []

    def reset_animation(self, env: AGVFleetEnv) -> None:
        """Clear animation state. Call on episode reset."""
        self._anim_t.clear()
        self._prev_pos.clear()
        self._time_since_step = 0.0
        self._particles.clear()
        self._known_completed.clear()
        self._selected_agvs.clear()
        for agv in env.agvs:
            self._prev_pos[agv.id] = agv.position
            self._anim_t[agv.id]   = 1.0

    # ------------------------------------------------------------------
    # Smooth movement helpers
    # ------------------------------------------------------------------

    def _agv_pixel_center(self, agv: AGV) -> Tuple[float, float]:
        """Interpolated pixel center of the AGV between prev and curr cell."""
        t    = min(1.0, self._anim_t.get(agv.id, 1.0))
        curr = self._cell_rect(*agv.position)
        if t >= 1.0:
            return float(curr.centerx), float(curr.centery)
        prev_pos  = self._prev_pos.get(agv.id, agv.position)
        prev_rect = self._cell_rect(*prev_pos)
        px = prev_rect.centerx + (curr.centerx - prev_rect.centerx) * t
        py = prev_rect.centery + (curr.centery - prev_rect.centery) * t
        return px, py

    # ------------------------------------------------------------------
    # Particle system
    # ------------------------------------------------------------------

    def _particle_params(self, stage: int, cs: int) -> Tuple:
        """Return (colors, n_burst, n_sparks, burst_speed, burst_decay, burst_size,
        spark_speed, spark_decay, spark_size) tuples for pickup (stage==-1) or delivery."""
        if stage == -1:
            return (
                [(255, 100, 80), (255, 180, 160), (255, 255, 255)],
                5, 6,
                (cs * 1.0, cs * 2.5), (2.5, 4.0), (2.0, 4.0),
                (cs * 0.2, cs * 0.8), (1.5, 2.5), (1.0, 2.5),
            )
        return (
            _STAGE_PARTICLE_COLORS.get(stage, [(255, 220, 80)]),
            8, 12,
            (cs * 1.5, cs * 3.5), (2.0, 3.5), (3.0, 6.0),
            (cs * 0.3, cs * 1.2), (1.2, 2.2), (1.5, 3.5),
        )

    def _emit_particles(self, grid_pos: Tuple[int, int], stage: int) -> None:
        rect   = self._cell_rect(*grid_pos)
        cx, cy = rect.centerx, rect.centery
        cs     = self._cell_size

        (colors, n_burst, n_sparks,
         burst_speed, burst_decay, burst_size,
         spark_speed, spark_decay, spark_size) = self._particle_params(stage, cs)

        for _ in range(n_burst):
            angle = random.uniform(0, 2 * math.pi)
            self._particles.append(_Particle(
                x=cx, y=cy,
                vx=math.cos(angle) * random.uniform(*burst_speed),
                vy=math.sin(angle) * random.uniform(*burst_speed),
                life=1.0,
                decay=random.uniform(*burst_decay),
                color=random.choice(colors),
                size=random.uniform(*burst_size),
            ))

        for _ in range(n_sparks):
            angle = random.uniform(0, 2 * math.pi)
            self._particles.append(_Particle(
                x=cx + random.uniform(-cs * 0.3, cs * 0.3),
                y=cy + random.uniform(-cs * 0.3, cs * 0.3),
                vx=math.cos(angle) * random.uniform(*spark_speed),
                vy=math.sin(angle) * random.uniform(*spark_speed),
                life=1.0,
                decay=random.uniform(*spark_decay),
                color=random.choice(colors),
                size=random.uniform(*spark_size),
            ))

    def _update_particles(self) -> None:
        for p in self._particles:
            p.x  += p.vx * self._dt
            p.y  += p.vy * self._dt
            p.vy += 40 * self._dt    # subtle gravity
            p.vx *= (1 - 2 * self._dt)  # air friction
            p.vy *= (1 - 2 * self._dt)
            p.life -= p.decay * self._dt
        self._particles = [p for p in self._particles if p.life > 0]

    def _draw_particles(self) -> None:
        for p in self._particles:
            alpha = int(p.life * 255)
            size  = max(1, int(p.size * p.life))
            surf  = pygame.Surface((size * 2 + 1, size * 2 + 1), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p.color, alpha), (size, size), size)
            self._screen.blit(surf, (int(p.x) - size, int(p.y) - size))

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _build_buttons(self) -> None:
        px = 10
        bw = self._left_w - 20
        bh = 28

        def y(row: int) -> int:
            return 16 + row * (bh + 8)

        self._buttons = [
            {"label": "FPS  -",       "rect": pygame.Rect(px, y(2), bw // 2 - 2, bh),
             "action": "fps_down",    "toggle": False},
            {"label": "FPS  +",       "rect": pygame.Rect(px + bw // 2 + 2, y(2), bw // 2 - 2, bh),
             "action": "fps_up",      "toggle": False},
            {"label": "PAUSE",        "rect": pygame.Rect(px, y(4), bw, bh),
             "action": "pause",       "toggle": True,  "state_attr": "_paused"},
            {"label": "RESET",        "rect": pygame.Rect(px, y(5), bw, bh),
             "action": "reset",       "toggle": False},
            {"label": "Task markers", "rect": pygame.Rect(px, y(7), bw, bh),
             "action": "toggle_markers",  "toggle": True, "state_attr": "show_task_markers"},
            {"label": "Grid lines",   "rect": pygame.Rect(px, y(8), bw, bh),
             "action": "toggle_grid",     "toggle": True, "state_attr": "show_grid_lines"},
            {"label": "AGV IDs",      "rect": pygame.Rect(px, y(9), bw, bh),
             "action": "toggle_ids",      "toggle": True, "state_attr": "show_agv_ids"},
            {"label": "Battery bars", "rect": pygame.Rect(px, y(10), bw, bh),
             "action": "toggle_battery",  "toggle": True, "state_attr": "show_battery_bars"},
        ]

        # Per-AGV path preview toggles (dynamic — depends on fleet size)
        path_start = 12
        agv_colors: Dict[int, Tuple[int, int, int]] = {
            a.id: _AGV_STATUS_COLORS[a.status] for a in self._last_agvs
        }
        for i, agv in enumerate(self._last_agvs):
            self._buttons.append({
                "label":        f"AGV {agv.id} path",
                "rect":         pygame.Rect(px, y(path_start + i), bw, bh),
                "action":       f"path_agv_{agv.id}",
                "toggle":       True,
                "agv_id":       agv.id,
                "active_color": agv_colors[agv.id],
            })

    # ------------------------------------------------------------------
    # Dividers
    # ------------------------------------------------------------------

    def _near_left_divider(self, mx: int) -> bool:
        return abs(mx - self._left_w) <= _DIVIDER_ZONE

    def _near_right_divider(self, mx: int) -> bool:
        return abs(mx - self._right_x) <= _DIVIDER_ZONE

    def _draw_dividers(self) -> None:
        mx = pygame.mouse.get_pos()[0]
        for x in (self._left_w, self._right_x):
            color = _COLOR_DIVIDER_HOVER if abs(mx - x) <= _DIVIDER_ZONE else _COLOR_DIVIDER
            pygame.draw.line(self._screen, color, (x, 0), (x, self._win_h), 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    def handle_events(self) -> str:
        mx = pygame.mouse.get_pos()[0]
        for event in pygame.event.get():
            result = self._process_event(event, mx)
            if result in ("quit", "reset"):
                return result
        self._update_cursor(mx)
        return "ok"

    def _process_event(self, event: pygame.event.Event, mx: int) -> Optional[str]:
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            return self._handle_keydown(event)
        if event.type == pygame.WINDOWRESIZED:
            self._win_w = event.x
            self._win_h = event.y
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self._handle_mouse_button_down(mx)
        if event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(mx)
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._drag_left   = False
            self._drag_right  = False
            self._held_action = None
        return None

    def _handle_keydown(self, event: pygame.event.Event) -> Optional[str]:
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
        return None

    def _handle_mouse_button_down(self, mx: int) -> Optional[str]:
        if self._near_left_divider(mx):
            self._drag_left = True
        elif self._near_right_divider(mx):
            self._drag_right = True
        else:
            pos = pygame.mouse.get_pos()
            self._handle_grid_click(pos)
            result = self._handle_button_click(pos)
            if result == "reset":
                return "reset"
            self._start_hold(pos)
        return None

    def _handle_mouse_motion(self, mx: int) -> None:
        if self._drag_left:
            self._left_w = max(_LEFT_W_MIN,
                               min(mx, self._win_w - self._right_w - _RIGHT_W_MIN))
        elif self._drag_right:
            self._right_w = max(_RIGHT_W_MIN,
                                min(self._win_w - mx,
                                    self._win_w - self._left_w - _LEFT_W_MIN))

    def _update_cursor(self, mx: int) -> None:
        near = (self._drag_left or self._drag_right
                or self._near_left_divider(mx) or self._near_right_divider(mx))
        cursor = pygame.SYSTEM_CURSOR_SIZEWE if near else pygame.SYSTEM_CURSOR_ARROW
        pygame.mouse.set_cursor(cursor)

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
                setattr(self, _toggle_attrs[action],
                        not getattr(self, _toggle_attrs[action]))
            elif action.startswith("path_agv_"):
                agv_id = btn["agv_id"]
                if agv_id in self._selected_agvs:
                    self._selected_agvs.discard(agv_id)
                else:
                    self._selected_agvs.add(agv_id)
        return None

    def _start_hold(self, pos: Tuple[int, int]) -> None:
        """Record which fps button (if any) is being held down."""
        for btn in self._buttons:
            if btn["action"] in ("fps_up", "fps_down") and btn["rect"].collidepoint(pos):
                self._held_action = btn["action"]
                self._held_time   = 0.0
                self._held_next   = 0.4   # initial delay before first repeat
                return
        self._held_action = None

    def _apply_fps_action(self, action: str) -> None:
        if action == "fps_up":
            self.fps = min(self.fps + 1, 60)
        elif action == "fps_down":
            self.fps = max(self.fps - 1, 1)

    def render(self, env: AGVFleetEnv) -> None:
        self._last_agvs = env.agvs          # cache for _handle_grid_click
        self._refresh_sprites()
        self._build_buttons()
        self._screen.fill(_COLOR_BG)
        self._draw_left_panel()
        self._draw_grid(env.plant)
        if self.show_task_markers:
            self._draw_task_markers(env.tasks)
        self._draw_path_preview(env.agvs)
        self._draw_agvs(env.agvs)
        self._draw_particles()
        self._draw_right_panel(env)
        self._draw_dividers()
        if self._paused:
            self._draw_paused_overlay()
        pygame.display.flip()

    def tick(self, fps: Optional[int] = None) -> None:
        self._clock.tick(fps or RENDER_FPS)
        self._dt = self._clock.get_time() / 1000.0

        # Advance anim_t for all AGVs
        step_dur = 1.0 / max(1, self.fps)
        for agv_id in self._anim_t:
            self._anim_t[agv_id] = min(1.0, self._anim_t[agv_id] + self._dt / step_dur)

        # Accumulate time for sim stepping
        if not self._paused:
            self._time_since_step += self._dt

        # Hold-repeat for fps +/- buttons
        if self._held_action:
            self._held_time += self._dt
            self._held_next -= self._dt
            if self._held_next <= 0:
                self._apply_fps_action(self._held_action)
                # Slow repeat (0–1.5 s held) → every 100 ms; fast after → every 40 ms
                self._held_next = 0.04 if self._held_time > 1.5 else 0.1

        self._update_particles()

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    def _draw_left_panel(self) -> None:
        pygame.draw.rect(self._screen, _COLOR_PANEL_BG,
                         pygame.Rect(0, 0, self._left_w, self._win_h))
        mouse_pos = pygame.mouse.get_pos()
        bh = 28

        self._screen.blit(
            self._font_sm.render("CONTROLS", True, _COLOR_TEXT_DIM), (10, 16))
        self._screen.blit(
            self._font_md.render(f"SIM: {self.fps:>2}/s", True, _COLOR_TEXT),
            (10, 16 + bh + 8))

        section_rows = {4: "SIMULATION", 7: "DISPLAY", 12: "PATH PREVIEW"}
        for row, label in section_rows.items():
            sy = 16 + row * (bh + 8) - 14
            self._screen.blit(
                self._font_xs.render(label, True, _COLOR_TEXT_DIM), (10, sy))

        for btn in self._buttons:
            rect    = btn["rect"]
            hovered = rect.collidepoint(mouse_pos)
            # Determine active state
            if btn["toggle"]:
                if "agv_id" in btn:
                    active = btn["agv_id"] in self._selected_agvs
                else:
                    active = getattr(self, btn.get("state_attr", ""), False)
            else:
                active = False
            active_color = btn.get("active_color", _COLOR_BTN_ACTIVE)
            if active:
                bg = active_color
            elif hovered:
                bg = _COLOR_BTN_HOVER
            else:
                bg = _COLOR_BTN_BG
            pygame.draw.rect(self._screen, bg, rect, border_radius=4)
            pygame.draw.rect(self._screen, _COLOR_BTN_BORDER, rect, 1, border_radius=4)
            lbl = self._font_sm.render(btn["label"], True, _COLOR_TEXT)
            self._screen.blit(lbl, lbl.get_rect(center=rect.center))

        hints = ["SPACE pause", "R  reset", "ESC quit"]
        hy = self._win_h - len(hints) * 16 - 8
        for hint in hints:
            self._screen.blit(
                self._font_xs.render(hint, True, _COLOR_TEXT_DIM), (10, hy))
            hy += 16

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------

    def _draw_grid(self, plant: PlantMap) -> None:
        for row in range(plant.GRID_SIZE):
            for col in range(plant.GRID_SIZE):
                cell_type = CellType(plant.grid[row, col])
                rect      = self._cell_rect(row, col)
                surf      = self._cell_sprites.get(cell_type)
                if surf:
                    self._screen.blit(surf, rect.topleft)
                else:
                    pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], rect)
                if self.show_grid_lines and cell_type != CellType.OBSTACLE:
                    pygame.draw.rect(self._screen, _COLOR_GRID_LINE, rect, 1)

    def _draw_cell_sprite_scaled(self, cell_type: CellType, rect: pygame.Rect) -> None:
        raw = self._raw_cell.get(cell_type)
        if raw:
            self._screen.blit(pygame.transform.scale(raw, (rect.width, rect.height)),
                               rect.topleft)
        else:
            pygame.draw.rect(self._screen, _CELL_COLORS[cell_type], rect)

    # ------------------------------------------------------------------
    # Path preview
    # ------------------------------------------------------------------

    def _handle_grid_click(self, pos: Tuple[int, int]) -> None:
        """Toggle path preview for the AGV under the click."""
        px, py = pos
        cs = self._cell_size
        gx = self._grid_x
        gp = self._grid_pixels
        if not (gx <= px < gx + gp and _TOP_MARGIN <= py < _TOP_MARGIN + gp):
            return
        col = (px - gx) // cs
        row = (py - _TOP_MARGIN) // cs
        if not (0 <= row < PlantMap.GRID_SIZE and 0 <= col < PlantMap.GRID_SIZE):
            return
        for agv in self._last_agvs:
            if agv.position == (row, col):
                if agv.id in self._selected_agvs:
                    self._selected_agvs.discard(agv.id)
                else:
                    self._selected_agvs.add(agv.id)
                return

    def _draw_path_preview(self, agvs: List[AGV]) -> None:
        """Draw the planned path overlay for every AGV whose preview is active."""
        if not self._selected_agvs:
            return

        line_surf = pygame.Surface((self._win_w, self._win_h), pygame.SRCALPHA)
        cs        = self._cell_size
        thickness = max(2, cs // 12)
        dot_r     = max(2, cs // 8)
        ring_r    = max(4, cs // 4)

        for agv in agvs:
            if agv.id not in self._selected_agvs:
                continue
            if not agv.path:
                continue

            color = _AGV_STATUS_COLORS[agv.status]
            start_px, start_py = self._agv_pixel_center(agv)
            waypoints: List[Tuple[float, float]] = [(start_px, start_py)]
            for cell in agv.path:
                r = self._cell_rect(*cell)
                waypoints.append((float(r.centerx), float(r.centery)))

            # Semi-transparent line segments
            for i in range(len(waypoints) - 1):
                x1, y1 = int(waypoints[i][0]),     int(waypoints[i][1])
                x2, y2 = int(waypoints[i + 1][0]), int(waypoints[i + 1][1])
                pygame.draw.line(line_surf, (*color, 110), (x1, y1), (x2, y2), thickness)

            # Dots at each waypoint (skip index 0 = AGV itself)
            for i, (wpx, wpy) in enumerate(waypoints[1:]):
                alpha    = max(60, 210 - i * 18)
                dot_surf = pygame.Surface((dot_r * 2 + 2, dot_r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(dot_surf, (*color, alpha), (dot_r + 1, dot_r + 1), dot_r)
                line_surf.blit(dot_surf, (int(wpx) - dot_r - 1, int(wpy) - dot_r - 1))

            # Destination ring at final waypoint
            if len(waypoints) > 1:
                dest_x, dest_y = int(waypoints[-1][0]), int(waypoints[-1][1])
                ring_surf = pygame.Surface((ring_r * 2 + 4, ring_r * 2 + 4), pygame.SRCALPHA)
                pygame.draw.circle(ring_surf, (*color, 210), (ring_r + 2, ring_r + 2), ring_r, 2)
                line_surf.blit(ring_surf, (dest_x - ring_r - 2, dest_y - ring_r - 2))

        self._screen.blit(line_surf, (0, 0))

    # ------------------------------------------------------------------
    # Task markers & AGVs
    # ------------------------------------------------------------------

    def _draw_task_markers(self, tasks: List[Task]) -> None:
        cs = self._cell_size
        for task in tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                pygame.draw.circle(self._screen, _COLOR_TASK_PICKUP,
                                   self._cell_rect(*task.pickup).center, cs // 4)
            if task.status == TaskStatus.IN_PROGRESS:
                pygame.draw.circle(self._screen, _COLOR_TASK_DELIVERY,
                                   self._cell_rect(*task.delivery).center, cs // 4, 2)

    def _draw_agvs(self, agvs: List[AGV]) -> None:
        for agv in agvs:
            px, py = self._agv_pixel_center(agv)
            self._draw_agv_sprite(agv, int(px), int(py))
            if self.show_battery_bars:
                self._draw_battery_bar(int(px), int(py), agv.battery)

    def _draw_agv_sprite(self, agv: AGV, cx: int, cy: int) -> None:
        if self._agv_sprite is None:
            return
        color   = _AGV_STATUS_COLORS[agv.status]
        heading = self._agv_heading.get(agv.id, (1, 0))
        angle   = self._heading_angle.get(heading, 0)

        tinted = self._agv_sprite.copy()
        tinted.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        rotated  = pygame.transform.rotate(tinted, angle)
        rot_rect = rotated.get_rect(center=(cx, cy))
        self._screen.blit(rotated, rot_rect.topleft)

        cs = self._cell_size
        if agv.task_id is not None:
            cw = cs - 14
            ch = max(4, cs // 6)
            pygame.draw.rect(self._screen, (255, 220, 60),
                             pygame.Rect(cx - cw // 2, cy - ch // 2, cw, ch),
                             border_radius=2)

        if self.show_agv_ids:
            lbl = self._font_sm.render(str(agv.id), True, (255, 255, 255))
            self._screen.blit(lbl, lbl.get_rect(center=(cx, cy + 4)))

    def _draw_battery_bar(self, cx: int, cy: int, battery: float) -> None:
        cs    = self._cell_size
        bar_w = cs - 6
        bar_h = max(3, cs // 10)
        bar_x = cx - bar_w // 2
        bar_y = cy + cs // 2 - bar_h - 2
        pygame.draw.rect(self._screen, (60, 60, 60),
                         pygame.Rect(bar_x, bar_y, bar_w, bar_h))
        if battery > 0.5:
            fill = (80, 200, 80)
        elif battery > 0.2:
            fill = (220, 180, 0)
        else:
            fill = (220, 60, 60)
        pygame.draw.rect(self._screen, fill,
                         pygame.Rect(bar_x, bar_y, int(bar_w * battery), bar_h))

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _draw_right_panel(self, env: AGVFleetEnv) -> None:
        rx = self._right_x
        pygame.draw.rect(self._screen, _COLOR_PANEL_BG,
                         pygame.Rect(rx, 0, self._right_w, self._win_h))

        info = env._get_info()
        y    = 16

        def text(msg: str, color=_COLOR_TEXT, font=None) -> None:
            nonlocal y
            surf = (font or self._font_md).render(msg, True, color)
            self._screen.blit(surf, (rx + 12, y))
            y += surf.get_height() + 4

        def separator() -> None:
            nonlocal y
            pygame.draw.line(self._screen, (60, 60, 60),
                             (rx + 8, y), (rx + self._right_w - 8, y))
            y += 10

        text("AGV Fleet Sim", color=_COLOR_ACCENT, font=self._font_lg)
        text(f"  Episode   {self.episode}")
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
            color = _AGV_STATUS_COLORS[agv.status]
            surf  = self._font_sm.render(
                f"  AGV {agv.id}  {agv.status.name.replace('_', ' '):<20}", True, color)
            self._screen.blit(surf, (rx + 12, y))
            y += surf.get_height() + 3
            bat = self._font_sm.render(f"         bat {agv.battery:.0%}", True, _COLOR_TEXT_DIM)
            self._screen.blit(bat, (rx + 12, y))
            y += bat.get_height() + 4
        separator()

        text("CELL LEGEND", color=_COLOR_TEXT_DIM)
        icon_size = 20
        for cell_type, label in [
            (CellType.ENTRY,    "Entry (raw material)"),
            (CellType.STAMPING, "Stamping press"),
            (CellType.BUFFER,   "Buffer / WIP"),
            (CellType.WELDING,  "Welding station"),
            (CellType.EXIT,     "Exit (finished)"),
            (CellType.CHARGING, "AGV Charging"),
            (CellType.OBSTACLE, "Obstacle"),
        ]:
            icon_rect = pygame.Rect(rx + 12, y, icon_size, icon_size)
            self._draw_cell_sprite_scaled(cell_type, icon_rect)
            pygame.draw.rect(self._screen, (80, 80, 80), icon_rect, 1, border_radius=2)
            self._screen.blit(
                self._font_sm.render(f"  {label}", True, _COLOR_TEXT),
                (rx + 12 + icon_size + 4, y + 3))
            y += icon_size + 4
        separator()

        text("AGV STATUS", color=_COLOR_TEXT_DIM)
        for status, color in _AGV_STATUS_COLORS.items():
            pygame.draw.circle(self._screen, color, (rx + 18, y + 6), 5)
            self._screen.blit(
                self._font_sm.render(f"      {status.name.replace('_', ' ').title()}",
                                     True, _COLOR_TEXT),
                (rx + 12, y))
            y += 16
        separator()

        pygame.draw.circle(self._screen, _COLOR_TASK_PICKUP, (rx + 18, y + 6), 5)
        self._screen.blit(self._font_sm.render("      Task pickup", True, _COLOR_TEXT),
                          (rx + 12, y))
        y += 16
        pygame.draw.circle(self._screen, _COLOR_TASK_DELIVERY, (rx + 18, y + 6), 5, 2)
        self._screen.blit(self._font_sm.render("      Task delivery", True, _COLOR_TEXT),
                          (rx + 12, y))

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def _draw_paused_overlay(self) -> None:
        overlay = pygame.Surface((self._grid_pixels, self._win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self._screen.blit(overlay, (self._grid_x, 0))
        surf = self._font_lg.render("PAUSED — press SPACE", True, (255, 255, 255))
        self._screen.blit(surf, surf.get_rect(
            center=(self._grid_x + self._grid_pixels // 2, self._win_h // 2)
        ))
