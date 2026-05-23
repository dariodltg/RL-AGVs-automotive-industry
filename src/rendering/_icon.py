"""
Programmatic 32×32 app icon: top-down view of an AGV.

  ┌─[W]─────[W]─┐
  │              │
  │      ◉       │  ← blue status light
  │              │
  │      ▲       │  ← direction mark
  │              │
  └─[W]─────[W]─┘
"""

import pygame


def make_app_icon() -> pygame.Surface:
    """Return a 32×32 RGBA surface drawn as a top-down AGV silhouette."""
    s = pygame.Surface((32, 32), pygame.SRCALPHA)

    # Body
    pygame.draw.rect(s, (175, 188, 205), pygame.Rect(7, 3, 18, 26), border_radius=3)
    # Outline
    pygame.draw.rect(s, (100, 115, 135), pygame.Rect(7, 3, 18, 26), 1, border_radius=3)

    # Wheels — two pairs (left / right), dark blue-grey
    _W = (38, 43, 60)
    pygame.draw.rect(s, _W, pygame.Rect(3,  6, 5, 7), border_radius=1)   # top-left
    pygame.draw.rect(s, _W, pygame.Rect(24, 6, 5, 7), border_radius=1)   # top-right
    pygame.draw.rect(s, _W, pygame.Rect(3, 19, 5, 7), border_radius=1)   # bottom-left
    pygame.draw.rect(s, _W, pygame.Rect(24,19, 5, 7), border_radius=1)   # bottom-right

    # Status light (blue, centre-top)
    pygame.draw.circle(s, (90, 160, 255), (16, 12), 3)
    pygame.draw.circle(s, (160, 210, 255), (16, 12), 1)   # bright core

    # Direction arrow (small triangle pointing up, near bottom of body)
    arrow = [(16, 18), (13, 23), (19, 23)]
    pygame.draw.polygon(s, (130, 145, 165), arrow)

    return s
