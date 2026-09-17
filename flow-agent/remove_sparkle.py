"""Remove the small gray sparkle/star artifact Flow's gem_pix_2 model keeps
adding to a corner of generated images, without touching nearby text.

Background: even when explicitly told "no sparkles, no icons, no
decorative graphics," gem_pix_2 image-to-image generations on a solid-color
background regularly still add a small 4-point sparkle glyph, almost always
in the bottom-right corner. Regenerating sometimes avoids it, sometimes
doesn't, and it can land close enough to burned-in text that a naive crop/
patch clips letters.

Method (fully deterministic, no AI/inpainting):
  1. Sample the true background color near the target corner.
  2. Build a mask of "candidate" pixels: not background, not near-pure
     foreground-text-bright (adjust --text-thresh for non-white text).
  3. Dilate the bright-text mask outward first and subtract it from the
     candidates, so nothing touching/adjacent to a letter is ever touched.
  4. Connected-component filter what's left, dropping specks smaller than
     --min-area (sensor noise) but keeping the sparkle-sized blob(s).
  5. Flat-fill the surviving pixels with the sampled background color.

Works well when the artifact sits on a flat, solid-color background (e.g.
the black-backdrop split-screen egg carousel). Will not help on a textured
or gradient background -- the flat-fill step assumes a uniform local color.

Usage:
    python remove_sparkle.py IMAGE.png
    python remove_sparkle.py IMAGE.png -o OUT.png --corner bottom-right
    python remove_sparkle.py IMAGE.png --region 850,860,1024,1000
    python remove_sparkle.py *.png --in-place   # batch, overwrite each file
"""

from __future__ import annotations

import argparse
import glob
import sys

import cv2
import numpy as np
from PIL import Image

CORNER_SIZE_FRAC = 0.20  # default corner region size as a fraction of image dims


def region_for_corner(w: int, h: int, corner: str) -> tuple[int, int, int, int]:
    rw, rh = int(w * CORNER_SIZE_FRAC), int(h * CORNER_SIZE_FRAC)
    if corner == "bottom-right":
        return w - rw, h - rh, w, h
    if corner == "bottom-left":
        return 0, h - rh, rw, h
    if corner == "top-right":
        return w - rw, 0, w, rh
    if corner == "top-left":
        return 0, 0, rw, rh
    raise ValueError(f"unknown corner: {corner}")


def sample_background_color(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int]:
    """Median color of the corner region's darkest quartile of pixels by
    luminance, as a robust stand-in for "the flat background color" even
    if some text/artifact pixels are mixed in."""
    region = arr[y0:y1, x0:x1].reshape(-1, 3)
    lum = region.mean(axis=1)
    dark_quartile = region[lum <= np.percentile(lum, 25)]
    if len(dark_quartile) == 0:
        return 0, 0, 0
    med = np.median(dark_quartile, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def remove_sparkle(
    arr: np.ndarray,
    region: tuple[int, int, int, int],
    text_thresh: int = 200,
    min_area: int = 5,
    max_area: int = 2000,
) -> tuple[np.ndarray, int]:
    """Returns (result_array, num_pixels_removed)."""
    x0, y0, x1, y1 = region
    bg_color = sample_background_color(arr, x0, y0, x1, y1)
    bg_lum = sum(bg_color) / 3

    crop = arr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # candidates: anything meaningfully different from the flat background,
    # but not as bright as real foreground text
    is_dark_bg = bg_lum < 128
    if is_dark_bg:
        candidates = (gray > bg_lum + 20) & (gray < text_thresh)
    else:
        candidates = (gray < bg_lum - 20) & (gray > 255 - text_thresh)
    cand_mask = (candidates.astype(np.uint8)) * 255

    # exclude anything touching real text (near-pure foreground), with a
    # margin, so letter anti-aliasing is never eligible for removal
    if is_dark_bg:
        text_mask = (gray > text_thresh).astype(np.uint8) * 255
    else:
        text_mask = (gray < 255 - text_thresh).astype(np.uint8) * 255
    text_mask = cv2.dilate(text_mask, np.ones((7, 7), np.uint8), iterations=1)
    isolated = cv2.bitwise_and(cand_mask, cv2.bitwise_not(text_mask))

    # keep only small-to-medium isolated blobs (the sparkle), drop noise
    # specks and drop anything implausibly large (probably real content)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(isolated, 8)
    clean = np.zeros_like(isolated)
    removed_px = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean[labels == i] = 255
            removed_px += int(area)
    clean = cv2.dilate(clean, np.ones((3, 3), np.uint8), iterations=1)

    full_mask = np.zeros(arr.shape[:2], np.uint8)
    full_mask[y0:y1, x0:x1] = clean

    out = arr.copy()
    out[full_mask > 0] = bg_color
    return out, removed_px


def process_file(path: str, out_path: str, corner: str, region_arg: str | None,
                  text_thresh: int, min_area: int, max_area: int) -> None:
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    h, w = arr.shape[:2]

    if region_arg:
        x0, y0, x1, y1 = (int(v) for v in region_arg.split(","))
        region = (x0, y0, x1, y1)
    else:
        region = region_for_corner(w, h, corner)

    result, removed_px = remove_sparkle(arr, region, text_thresh, min_area, max_area)
    Image.fromarray(result).save(out_path)

    status = f"removed ~{removed_px}px artifact" if removed_px else "nothing found to remove"
    print(f"{path} -> {out_path}  ({status})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image file(s) or glob pattern(s)")
    ap.add_argument("-o", "--output", help="output path (single-file mode only)")
    ap.add_argument("--in-place", action="store_true", help="overwrite each input file")
    ap.add_argument("--corner", default="bottom-right",
                     choices=["bottom-right", "bottom-left", "top-right", "top-left"])
    ap.add_argument("--region", help="explicit x0,y0,x1,y1 pixel box instead of --corner")
    ap.add_argument("--text-thresh", type=int, default=200,
                     help="brightness (on a dark bg) treated as real text, protected from removal")
    ap.add_argument("--min-area", type=int, default=5, help="drop candidate blobs smaller than this (noise)")
    ap.add_argument("--max-area", type=int, default=2000, help="drop candidate blobs larger than this (real content)")
    args = ap.parse_args()

    files: list[str] = []
    for pattern in args.images:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])

    if len(files) > 1 and args.output and not args.in_place:
        print("error: --output only works with a single input file; use --in-place for batches", file=sys.stderr)
        sys.exit(1)

    for f in files:
        if args.in_place:
            out = f
        elif args.output:
            out = args.output
        else:
            dot = f.rfind(".")
            out = f[:dot] + "_clean" + f[dot:] if dot > 0 else f + "_clean"
        process_file(f, out, args.corner, args.region, args.text_thresh, args.min_area, args.max_area)


if __name__ == "__main__":
    main()
