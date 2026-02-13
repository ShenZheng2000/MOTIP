#!/usr/bin/env python3
"""
Make a simple grid / chessboard-like image (white background + black grid lines)
similar to the attached example.

Usage:
  python make_chessboard_grid.py --out debug/grid.png --size 512 --step 48 --line 3
"""

import argparse
from PIL import Image, ImageDraw

def make_grid_image(size: int, step: int, line: int, margin: int = 0) -> Image.Image:
    """
    White background with black grid lines.
    - size: output image is size x size
    - step: spacing between grid lines (pixels)
    - line: line thickness (pixels)
    - margin: optional empty border around the grid
    """
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # draw vertical lines
    x = margin
    while x < size - margin:
        draw.rectangle([x, margin, x + line - 1, size - margin - 1], fill=(0, 0, 0))
        x += step

    # draw horizontal lines
    y = margin
    while y < size - margin:
        draw.rectangle([margin, y, size - margin - 1, y + line - 1], fill=(0, 0, 0))
        y += step

    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="debug/grid.png", help="output path, e.g. debug/grid.png")
    ap.add_argument("--size", type=int, default=512, help="image size (square)")
    ap.add_argument("--step", type=int, default=48, help="grid spacing in pixels")
    ap.add_argument("--line", type=int, default=3, help="grid line thickness in pixels")
    ap.add_argument("--margin", type=int, default=0, help="optional border margin in pixels")
    args = ap.parse_args()

    img = make_grid_image(size=args.size, step=args.step, line=args.line, margin=args.margin)
    img.save(args.out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
