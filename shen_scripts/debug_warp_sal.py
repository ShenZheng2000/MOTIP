import glob
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image

from warp_utils.warp_pipeline import apply_forward_warp
from warp_utils.warping_layers import invert_grid, unwarp_bboxes
from shen_scripts.debug_warp import load_mot_gt_by_frame

'''
python -m shen_scripts.debug_warp_sal
'''

def main():
    img_dir = Path("/ssd0/shenzhen/Datasets/tracking/DanceTrack/val/dancetrack0019/img1")
    gt_path = "/ssd0/shenzhen/Datasets/tracking/DanceTrack/val/dancetrack0019/gt/gt.txt"

    alpha = 0.3
    bw = 128

    RETURN_SALIENCY = True
    SMOOTH_SALIENCY = True

    sal_smooth_tag = f"salEMA{alpha:.1f}" if SMOOTH_SALIENCY else "salRaw"
    out_dir = Path(f"debug_warped/dancetrack0019_bw{bw}_{sal_smooth_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    frame_to_boxes = load_mot_gt_by_frame(gt_path)

    img_paths = sorted(glob.glob(str(img_dir / "*.jpg")))

    for p in img_paths:
        frame_idx = int(Path(p).stem)

        pil = Image.open(p).convert("RGB")
        img = to_tensor(pil).unsqueeze(0).to(device)

        box_list = frame_to_boxes.get(frame_idx, None)
        if box_list is None:
            bboxes = None
        else:
            # raw GT boxes (xyxy) -- NO SMOOTHING
            bboxes = torch.tensor([b[1:] for b in box_list], dtype=torch.float32).to(device)

        print(f"\nFrame {frame_idx}")
        if bboxes is None:
            print("  No bbox found")
        else:
            print(f"  Num boxes: {len(bboxes)}")
            print(f"  First box (xyxy): {bboxes[0]}")

        if bboxes is None or bboxes.numel() == 0:
            save_image(img[0].cpu(), str(out_dir / Path(p).name), normalize=False)
            continue

        # IMPORTANT: bbox2sal mutates bbox in-place, so keep a copy for visualization
        bboxes_vis = bboxes.clone()

        with torch.no_grad():
            if RETURN_SALIENCY:
                warped, warp_grid, sal = apply_forward_warp(
                    image_tensor=img,
                    bbox_tensor=bboxes.clone(),
                    bw=bw,
                    separable=True,
                    output_shape=None,
                    return_saliency=True,
                    smooth_saliency=SMOOTH_SALIENCY,
                    smooth_alpha=alpha,
                )
                save_image(sal[0].cpu(), str(out_dir / f"{Path(p).stem}_sal.png"), normalize=True)

            else:
                raise NotImplementedError("Set RETURN_SALIENCY=True to save warped image.")

        # save warped image
        save_image(warped[0].cpu(), str(out_dir / Path(p).name), normalize=False)

    print(f"\nDone. Saved warped frames to: {out_dir}")


if __name__ == "__main__":
    main()