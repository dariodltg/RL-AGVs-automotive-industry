"""
Export cell type sprites as PNG files for use in documentation.

Renders each cell sprite at 64x64px using the same drawing logic as
PygameRenderer, then saves them to docs/imgs/sprites/.

Usage:
    python scripts/export_sprites.py
"""

import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"   # headless — no window needed
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

from src.env.plant_map import CellType
from src.rendering.pygame_renderer import _CELL_COLORS

SPRITE_SIZE = 64
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "imgs", "sprites")

# ---------------------------------------------------------------------------
# Sprite drawing functions (copied from PygameRenderer, adapted for surface)
# ---------------------------------------------------------------------------

def _draw_entry(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    cx = size // 2
    x, y = 0, 0

    arch_rect = pygame.Rect(x + 5, y + 4, size - 10, size - 8)
    pygame.draw.rect(surf, (20, 80, 40), arch_rect, border_radius=6)
    pygame.draw.rect(surf, (100, 230, 130), arch_rect, 2, border_radius=6)
    inner = pygame.Rect(x + 9, y + 8, size - 18, size - 14)
    pygame.draw.rect(surf, (10, 40, 20), inner, border_radius=3)

    arrow_color = (120, 255, 150)
    dn_tip = (cx, y + size - 8)
    pygame.draw.polygon(surf, arrow_color, [
        dn_tip, (cx - 4, y + size - 15), (cx + 4, y + size - 15)
    ])
    pygame.draw.line(surf, arrow_color, (cx, y + 10), (cx, y + size - 15), 2)

    dash_y = size - 3
    for i in range(3):
        dx = x + 7 + i * 8
        pygame.draw.line(surf, (100, 230, 130), (dx, dash_y), (dx + 4, dash_y), 2)


def _draw_stamping(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    cx = size // 2
    x, y = 0, 0

    frame = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
    pygame.draw.rect(surf, (25, 55, 100), frame, border_radius=2)
    pygame.draw.rect(surf, (120, 170, 240), frame, 1, border_radius=2)

    pygame.draw.rect(surf, (80, 130, 200),
                     pygame.Rect(x + 5, y + 6, size - 10, 4))

    plate_y = y + size // 2
    pygame.draw.rect(surf, (100, 150, 220),
                     pygame.Rect(x + 6, plate_y, size - 12, 5))

    for ax in (cx - 5, cx + 2):
        pygame.draw.polygon(surf, (180, 210, 255), [
            (ax, plate_y - 6), (ax + 3, plate_y - 2), (ax + 6, plate_y - 6)
        ])


def _draw_buffer(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    x, y = 0, 0

    frame = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
    pygame.draw.rect(surf, (90, 60, 20), frame, border_radius=2)
    pygame.draw.rect(surf, (220, 170, 60), frame, 1, border_radius=2)

    for i in range(2):
        shelf_y = y + 11 + i * 9
        pygame.draw.line(surf, (200, 150, 50),
                         (x + 6, shelf_y), (x + size - 6, shelf_y), 1)
        for bx in (x + 7, x + 13, x + 19):
            pygame.draw.rect(surf, (240, 190, 80),
                             pygame.Rect(bx, shelf_y - 5, 4, 5))


def _draw_welding(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    cx = size // 2
    cy = size // 2
    x, y = 0, 0

    frame = pygame.Rect(x + 4, y + 4, size - 8, size - 8)
    pygame.draw.rect(surf, (100, 25, 25), frame, border_radius=2)
    pygame.draw.rect(surf, (240, 100, 100), frame, 1, border_radius=2)

    torch_start = (x + 7, y + 7)
    torch_end   = (cx + 3, cy + 3)
    pygame.draw.line(surf, (200, 80, 80), torch_start, torch_end, 3)
    pygame.draw.circle(surf, (220, 100, 80), torch_end, 3)

    spark_color = (255, 220, 60)
    for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4), (3, 3), (-3, -3)):
        sx, sy = torch_end[0] + dx, torch_end[1] + dy
        pygame.draw.line(surf, spark_color, torch_end, (sx, sy), 1)


def _draw_exit(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    cx = size // 2
    x, y = 0, 0

    arch_rect = pygame.Rect(x + 5, y + 4, size - 10, size - 8)
    pygame.draw.rect(surf, (60, 25, 110), arch_rect, border_radius=6)
    pygame.draw.rect(surf, (180, 100, 255), arch_rect, 2, border_radius=6)
    inner = pygame.Rect(x + 9, y + 8, size - 18, size - 14)
    pygame.draw.rect(surf, (30, 10, 60), inner, border_radius=3)

    arrow_color = (200, 150, 255)
    up_tip = (cx, y + 8)
    pygame.draw.polygon(surf, arrow_color, [
        up_tip, (cx - 4, y + 15), (cx + 4, y + 15)
    ])
    pygame.draw.line(surf, arrow_color, (cx, y + 15), (cx, y + size - 8), 2)

    dash_y = size - 3
    for i in range(3):
        dx = x + 7 + i * 8
        pygame.draw.line(surf, (180, 100, 255), (dx, dash_y), (dx + 4, dash_y), 2)


def _draw_charging(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    cx = size // 2
    cy = size // 2
    x, y = 0, 0

    platform = pygame.Rect(x + 4, size - 9, size - 8, 6)
    pygame.draw.rect(surf, (60, 50, 20), platform, border_radius=2)
    pygame.draw.rect(surf, (180, 140, 40), platform, 1, border_radius=2)

    post_rect = pygame.Rect(cx - 2, y + 5, 4, size - 14)
    pygame.draw.rect(surf, (120, 100, 30), post_rect, border_radius=2)

    glow_surf = pygame.Surface((18, 18), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (255, 210, 0, 60), (9, 9), 9)
    surf.blit(glow_surf, (cx - 9, cy - 11))

    bx, by = cx, cy - 9
    bolt_points = [
        (bx + 3, by),      (bx,     by + 6),
        (bx + 3, by + 6),  (bx - 3, by + 13),
        (bx,     by + 7),  (bx - 3, by + 7),
    ]
    pygame.draw.polygon(surf, (255, 220, 0), bolt_points)
    pygame.draw.polygon(surf, (200, 160, 0), bolt_points, 1)


def _draw_obstacle(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    # Just the solid background — obstacles have no sprite
    pygame.draw.rect(surf, (45, 45, 45), pygame.Rect(0, 0, size, size))


def _draw_free(surf: pygame.Surface) -> None:
    size = SPRITE_SIZE
    pygame.draw.rect(surf, (210, 210, 210), pygame.Rect(0, 0, size, size))


# ---------------------------------------------------------------------------

SPRITES = {
    "entry":    (CellType.ENTRY,    _draw_entry),
    "stamping": (CellType.STAMPING, _draw_stamping),
    "buffer":   (CellType.BUFFER,   _draw_buffer),
    "welding":  (CellType.WELDING,  _draw_welding),
    "exit":     (CellType.EXIT,     _draw_exit),
    "charging": (CellType.CHARGING, _draw_charging),
    "obstacle": (CellType.OBSTACLE, _draw_obstacle),
    "free":     (CellType.FREE,     _draw_free),
}


def main() -> None:
    pygame.init()
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, (cell_type, draw_fn) in SPRITES.items():
        surf = pygame.Surface((SPRITE_SIZE, SPRITE_SIZE))
        bg   = _CELL_COLORS[cell_type]
        surf.fill(bg)
        draw_fn(surf)
        path = os.path.join(OUT_DIR, f"{name}.png")
        pygame.image.save(surf, path)
        print(f"  Saved {path}")

    pygame.quit()
    print(f"\nAll sprites exported to {OUT_DIR}/")


if __name__ == "__main__":
    main()
