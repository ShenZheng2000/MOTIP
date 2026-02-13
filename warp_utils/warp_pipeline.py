"""
warp_pipeline.py
Reusable warp–unwarp utilities for Pix2Pix-Turbo (used in both train & inference).
"""

import numpy as np
import torch
from .warping_layers import PlainKDEGrid, warp, invert_grid

# optional import (only used for face detection)
from torchvision import transforms
import torch.nn.functional as F
import os
import json
from PIL import Image, ImageOps
from pathlib import Path


def get_gt_bbox(base_name, img_pil, bbox_map, device="cuda"):
    """
    Given:
        base_name : image filename (e.g. 0001.jpg)
        img_pil   : PIL image
        bbox_map  : dict { filename: [ [x,y,w,h], ... ] }
        device    : torch device for tensor output

    Returns:
        bbox tensor of shape [N, 4] with (x1, y1, x2, y2)
        or full-image box if missing.
    """
    if bbox_map is None:
        # fallback: full image
        w_img, h_img = img_pil.size
        return torch.tensor([[0., 0., float(w_img), float(h_img)]], device=device)

    if base_name not in bbox_map:
        # fallback: full image
        w_img, h_img = img_pil.size
        return torch.tensor([[0., 0., float(w_img), float(h_img)]], device=device)

    valid = []
    for (x, y, w, h) in bbox_map[base_name]:
        if w <= 1 or h <= 1 or any(v != v for v in (x, y, w, h)):
            continue
        valid.append([x, y, x + w, y + h])

    if not valid:
        # no valid GT → fallback to full image
        w_img, h_img = img_pil.size
        return torch.tensor([[0., 0., float(w_img), float(h_img)]], device=device)

    return torch.tensor(valid, device=device)


def visualize_bbox(img_pil, bbox_tensor, base_name, save_dir):
    """
    Draw green rectangle(s) on the image for debugging.

    Args:
        img_pil      : input PIL image
        bbox_tensor  : [N, 4] torch tensor
        base_name    : file name for output
        save_dir     : directory where the bbox visualization is saved
    """
    try:
        import cv2
        import numpy as np
        os.makedirs(save_dir, exist_ok=True)

        img_cv = np.array(img_pil)[:, :, ::-1].copy()   # PIL RGB → OpenCV BGR
        for box in bbox_tensor:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        out_path = os.path.join(save_dir, base_name)
        cv2.imwrite(out_path, img_cv)
    except Exception as e:
        print(f"[WARN] Failed to visualize bbox for {base_name}: {e}")


def load_bbox_map(json_list):
    """
    Load and merge MANY COCO JSON bbox maps.
    Keyed by basename.
    Crashes on any basename collision.
    """
    if not json_list:
        return None

    merged = {}

    for path in json_list:
        if not path or not os.path.exists(path):
            continue

        with open(path, "r") as f:
            coco = json.load(f)

        # build id -> basename
        id2name = {}
        seen = set()
        for img in coco["images"]:
            base = os.path.basename(img["file_name"])
            if base in seen:
                raise RuntimeError(f"❌ Basename collision INSIDE {path}: {base}")
            seen.add(base)
            id2name[img["id"]] = base

        # collect bboxes
        part = {}
        for ann in coco["annotations"]:
            fn = id2name[ann["image_id"]]
            part.setdefault(fn, []).append(ann["bbox"])

        print(f"📘 Loaded {len(part)} entries from {path}")

        # merge, but forbid overwriting
        dup = set(merged).intersection(part)
        if dup:
            raise RuntimeError(
                f"❌ Basename collision ACROSS JSONs: {sorted(list(dup))[:5]}"
            )

        merged.update(part)

    print(f"✅ Total merged bbox entries: {len(merged)}")

    # ---- DEBUG PREVIEW ----
    print("🔎 [DEBUG] bbox_map preview:")
    keys = list(merged.keys())

    for i, k in enumerate(keys[:4]):
        print(f"  [TOP {i}] {k} -> {merged[k]}")

    if len(keys) > 4:
        print("  ...")

    for i, k in enumerate(keys[-4:]):
        print(f"  [BOT {len(keys)-4+i}] {k} -> {merged[k]}")

    return merged


def resize_longest_side(img_pil, cropped_size, target_size):
    cw, ch = cropped_size
    aspect_ratio = cw / ch
    if aspect_ratio >= 1:
        new_w, new_h = target_size, int(target_size / aspect_ratio)
    else:
        new_w, new_h = int(target_size * aspect_ratio), target_size
    return img_pil.resize((new_w, new_h), Image.LANCZOS)


def find_fg_mask_path(input_path):
    """
    ONLY responsible for finding mask_path.
    Priority matches inference image selection:
    flat > legacy
    """
    input_path = Path(input_path)

    # -------------------------
    # Flat layout (HIGH PRIORITY)
    #   input_dir/images/xxx.jpg
    #   input_dir/fg_masks/xxx.png
    # -------------------------
    if input_path.parent.name == "image":
        root = input_path.parent.parent
        fg_dir = root / "fg_masks"
        if fg_dir.exists():
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = fg_dir / f"{input_path.stem}{ext}"
                if candidate.exists():
                    return candidate

    # -------------------------
    # Legacy layout (FALLBACK)
    #   seq_x/image.jpg
    #   seq_x/pre_processing/black_fg_mask_groundedsam2.png
    # -------------------------
    legacy_mask = (
        input_path.parent / "pre_processing" / "black_fg_mask_groundedsam2.png"
    )
    if legacy_mask.exists():
        return legacy_mask

    return None


def crop_to_foreground(input_path):
    # input_root = input_path.parent  # e.g. .../8seconds_men_shirts_034
    # mask_path = input_root / "pre_processing" / "black_fg_mask_groundedsam2.png"

    # print(f"\n🟢 Image: {input_path}")
    # print(f"🔍 Mask:  {mask_path}")

    mask_path = find_fg_mask_path(input_path)

    img = Image.open(input_path).convert("RGB")

    if mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        inverted_mask = ImageOps.invert(mask)
        bbox = inverted_mask.getbbox()
        if bbox:
            img = img.crop(bbox)
            # print(f"✅ Cropped to bbox {bbox}")
        else:
            print("⚠️ Empty mask, using full image.")
    else:
        print("⚠️ Mask not found, using full image.")

    return img, img.size


# ===============================================================
# ✅ Forward warp (returns warped tensor + warp grid)
# ===============================================================
def apply_forward_warp(image_tensor,
                        bbox_tensor,
                        bw,
                        separable=True,
                        output_shape=None,
                        return_saliency=False,
                        warp_smooth_alpha=0.0):   # 0.0 = no smoothing
    """
    Applies KDE-based forward warp around bbox region.
    Returns (warped_tensor, warp_grid).
    """
    device = image_tensor.device
    bbox_tensor = bbox_tensor.to(device)

    _, _, H, W = image_tensor.shape

    # --- NEW LOGIC ---
    if output_shape is None:
        out_H, out_W = H, W
    else:
        out_H, out_W = output_shape

    # ✅ cache grid_net so self._sal_prev persists across frames
    global _PLAIN_KDEGRID_CACHE
    key = (H, W, out_H, out_W, separable, bw,
           return_saliency, float(warp_smooth_alpha), device.type)

    if "_PLAIN_KDEGRID_CACHE" not in globals() or _PLAIN_KDEGRID_CACHE[0] != key:
        grid_net = PlainKDEGrid(
            input_shape=(H, W),
            output_shape=(out_H, out_W),
            separable=separable,
            bandwidth_scale=bw,
            amplitude_scale=1.0,
            return_saliency=return_saliency,
            warp_smooth_alpha=warp_smooth_alpha,
        ).to(device)
        _PLAIN_KDEGRID_CACHE = (key, grid_net)
    else:
        grid_net = _PLAIN_KDEGRID_CACHE[1]

    warp_outputs = grid_net(image_tensor, gt_bboxes=bbox_tensor.unsqueeze(0))

    if return_saliency:
        warp_grid, saliency = warp_outputs
    else:
        warp_grid = warp_outputs

    warped = warp(warp_grid, image_tensor)

    if return_saliency:
        return warped, warp_grid, saliency
    else:
        return warped, warp_grid

# ===============================================================
# ✅ Unwarp (returns restored tensor)
# ===============================================================
def apply_unwarp(warp_grid, warped_output, separable=True):
    """
    Applies inverse KDE grid to unwarp the warped output tensor.
    """
    device = warped_output.device  # make sure we stay on the same GPU
    _, _, H, W = warped_output.shape

    # NOTE: resize warp_grid to have same height and width as warped_output
    grid = warp_grid.to(device)
    grid = grid.permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(warped_output.shape[-2], warped_output.shape[-1]), mode='bilinear', align_corners=True)
    grid = grid.permute(0, 2, 3, 1)    

    # inv_grid = invert_grid(warp_grid.to(device), (1, 3, H, W), separable=separable)
    inv_grid = invert_grid(grid.to(device), (1, 3, H, W), separable=separable)
    restored = F.grid_sample(warped_output, inv_grid.to(device), align_corners=True)
    return restored