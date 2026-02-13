import glob
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from warp_utils.warp_pipeline import apply_forward_warp
from warp_utils.warping_layers import invert_grid, unwarp_bboxes

# ===============================================================
# MOT GT reader: frame,id,x,y,w,h,conf,cls,vis  -> (x1,y1,x2,y2)
# ===============================================================
def load_mot_gt_by_frame(gt_path: str):
    """
    Returns dict: frame_idx -> list of (track_id, x1,y1,x2,y2)  (float)
    MOT format: frame,id,x,y,w,h,conf,cls,vis
    """
    frame_to = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            frame = int(float(parts[0]))
            tid   = int(float(parts[1]))
            x = float(parts[2]); y = float(parts[3])
            w = float(parts[4]); h = float(parts[5])

            x1, y1, x2, y2 = x, y, x + w, y + h
            frame_to.setdefault(frame, []).append((tid, x1, y1, x2, y2))
    return frame_to


def smooth_boxes_ema(box_list, state, alpha=0.1):
    """
    box_list: list of (tid, x1,y1,x2,y2)
    state: dict tid -> (w_smooth, h_smooth)
    returns: Tensor[N,4] xyxy with center from current frame, wh smoothed
    """
    out = []
    for tid, x1, y1, x2, y2 in box_list:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w  = (x2 - x1)
        h  = (y2 - y1)

        if tid not in state:
            ws, hs = w, h
        else:
            w_prev, h_prev = state[tid]
            ws = alpha * w + (1 - alpha) * w_prev
            hs = alpha * h + (1 - alpha) * h_prev

        state[tid] = (ws, hs)

        # rebuild xyxy using current center + smoothed wh
        nx1 = cx - 0.5 * ws
        ny1 = cy - 0.5 * hs
        nx2 = cx + 0.5 * ws
        ny2 = cy + 0.5 * hs
        out.append([nx1, ny1, nx2, ny2])

    return torch.tensor(out, dtype=torch.float32)


def visualize_bbox(img_pil, orig_bbox, smooth_bbox, base_name, save_dir):
    """
    Draw original bbox in red and smoothed bbox in blue.
    """
    try:
        import cv2
        import numpy as np
        import os

        os.makedirs(save_dir, exist_ok=True)

        img_cv = np.array(img_pil)[:, :, ::-1].copy()  # RGB → BGR

        # 🔴 Original (red in BGR = (0,0,255))
        if orig_bbox is not None:
            for box in orig_bbox:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 🔵 Smoothed (blue in BGR = (255,0,0))
        if smooth_bbox is not None:
            for box in smooth_bbox:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)

        out_path = os.path.join(save_dir, base_name)
        cv2.imwrite(out_path, img_cv)

    except Exception as e:
        print(f"[WARN] Failed to visualize bbox for {base_name}: {e}")


def main():
    img_dir = Path("/ssd0/shenzhen/Datasets/tracking/DanceTrack/val/dancetrack0019/img1")
    gt_path = "/ssd0/shenzhen/Datasets/tracking/DanceTrack/val/dancetrack0019/gt/gt.txt"

    alpha = 0.3  # TODO: adjust this alpha later [0.3, 0.5, 0.7] etc. 
    bw = 128

    # 🔥 Save inside debug_warped/
    out_dir = Path(f"debug_warped/dancetrack0019_bw{bw}_alpha{alpha:.1f}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    frame_to_boxes = load_mot_gt_by_frame(gt_path)
    smooth_state = {}   # tid -> (w_smooth, h_smooth)

    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))

    for p in img_paths:
        frame_idx = int(Path(p).stem)

        pil = Image.open(p).convert("RGB")
        img = to_tensor(pil).unsqueeze(0).to(device)

        box_list = frame_to_boxes.get(frame_idx, None)
        if box_list is None:
            orig_bboxes = None
            bboxes = None
        else:
            # original boxes (xyxy)
            orig_bboxes = torch.tensor([b[1:] for b in box_list], dtype=torch.float32).to(device)
            # smoothed boxes (xyxy)
            bboxes = smooth_boxes_ema(box_list, smooth_state, alpha=alpha).to(device)

        # ✅ DEBUG PRINT
        print(f"\nFrame {frame_idx}")
        if bboxes is None:
            print("  No bbox found")
        else:
            print(f"  Num boxes: {len(bboxes)}")
            print(f"  First box (xyxy): {bboxes[0]}")

        if bboxes is None or bboxes.numel() == 0:
            save_image(img[0].cpu(), str(out_dir / Path(p).name), normalize=False)
            continue

        # IMPORTANT:
        # PlainKDEGrid.bbox2sal() mutates bbox_tensor in-place (xyxy -> xywh),
        # so keep a copy for visualization, and pass a clone into the warp.
        bboxes_vis = bboxes.clone()
        orig_vis = orig_bboxes.clone() if orig_bboxes is not None else None

        with torch.no_grad():
            warped, warp_grid = apply_forward_warp(
                image_tensor=img,
                bbox_tensor=bboxes.clone(),
                bw=bw,
                separable=True,
                output_shape=None
            )

        # compute inverse grid (original -> warped)
        _, _, H, W = img.shape
        inverse_grid = invert_grid(warp_grid, (1, 3, H, W), separable=True)  # [1,H,W,2]

        # warp bboxes (original coords -> warped coords)
        orig_on_warped = unwarp_bboxes(orig_vis, inverse_grid[0], output_shape=(H, W)) if orig_vis is not None else None
        smooth_on_warped = unwarp_bboxes(bboxes_vis, inverse_grid[0], output_shape=(H, W))

        # save warped image
        save_image(warped[0].cpu(), str(out_dir / Path(p).name), normalize=False)

        # 🔥 visualize on warped image (orig=red, smooth=blue)
        warped_pil = to_pil_image(warped[0].cpu().clamp(0, 1))
        visualize_bbox(
            warped_pil,
            orig_on_warped.cpu() if orig_on_warped is not None else None,
            smooth_on_warped.cpu() if smooth_on_warped is not None else None,
            base_name=f"{Path(p).stem}_vis.jpg",
            save_dir=str(out_dir)
        )

    print(f"\nDone. Saved warped frames to: {out_dir}")


if __name__ == "__main__":
    main()