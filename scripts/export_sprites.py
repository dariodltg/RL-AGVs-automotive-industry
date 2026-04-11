"""
Export cell type sprites to docs/imgs/sprites/ for use in documentation.

The canonical sprites live in src/rendering/assets/cells/.
This script copies them (optionally resizing) to the docs folder so the
README always shows the sprites actually used in the renderer.

Usage:
    python scripts/export_sprites.py
    python scripts/export_sprites.py --size 64   # resize to 64x64 (default)
    python scripts/export_sprites.py --size 128  # larger for high-DPI README
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "rendering", "assets", "cells",
)
DST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "imgs", "sprites",
)

SPRITE_NAMES = [
    "entry", "stamping", "buffer", "welding",
    "exit", "charging", "obstacle", "free",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    pygame.init()
    os.makedirs(DST_DIR, exist_ok=True)

    for name in SPRITE_NAMES:
        src = os.path.join(SRC_DIR, f"{name}.png")
        dst = os.path.join(DST_DIR, f"{name}.png")
        img     = pygame.image.load(src).convert()
        scaled  = pygame.transform.scale(img, (args.size, args.size))
        pygame.image.save(scaled, dst)
        print(f"  {name}.png  ({args.size}x{args.size})  -> {dst}")

    pygame.quit()
    print(f"\nDone. {len(SPRITE_NAMES)} sprites exported to {DST_DIR}/")


if __name__ == "__main__":
    main()
