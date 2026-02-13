# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
from PIL import Image
from torchvision.transforms import v2

from torch.utils.data import Dataset
from utils.nested_tensor import nested_tensor_from_tensor_list


class SeqDataset(Dataset):
    def __init__(
            self,
            seq_info,
            image_paths,
            annotations,
            max_shorter: int = 800,
            max_longer: int = 1536,
            size_divisibility: int = 0,
            dtype=torch.float32,
    ):
        self.seq_info = seq_info
        self.image_paths = image_paths
        self.annotations = annotations   # NOTE: add gt annotations
        self.max_shorter = max_shorter
        self.max_longer = max_longer
        self.size_divisibility = size_divisibility
        self.dtype = dtype

        self.transform = v2.Compose([
            v2.Resize(size=self.max_shorter, max_size=self.max_longer),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # NOTE: reset saliency smoothing state for new sequence
        try:
            from warp_utils.warp_pipeline import _PLAIN_KDEGRID_CACHE
            _PLAIN_KDEGRID_CACHE[1]._sal_prev = None
            print("[data/seq_dataset.py] Reset _PLAIN_KDEGRID_CACHE._sal_prev for new sequence.")
        except:
            pass

        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = self._load(self.image_paths[item])
        transformed_image = self.transform(image)

        # NOTE: getting original and new widths for GT resizing
        orig_w = image.size[0] # (width, height)
        new_w = transformed_image.shape[-1] # (3, height, width)
        
        if self.dtype != torch.float32:
            transformed_image = transformed_image.to(self.dtype)
        transformed_image = nested_tensor_from_tensor_list([transformed_image], self.size_divisibility)
        
        # NOTE: get, resize GT annotations and convert to (x1,y1,x2,y2) format, then attach to the transformed image
        gt = self.annotations[item] if self.annotations is not None else None
        gt = self.resize_gt_xywh_to_xyxy(gt, orig_w, new_w)
        transformed_image.gt = gt
        
        return transformed_image, self.image_paths[item]

    def seq_hw(self):
        return self.seq_info["height"], self.seq_info["width"]

    @staticmethod
    def _load(path):
        image = Image.open(path)
        return image
            
    def resize_gt_xywh_to_xyxy(self, gt, orig_w, new_w):
        if gt is None or gt["bbox"].numel() == 0:
            return gt

        scale = new_w / orig_w   # uniform scale (aspect ratio preserved)

        gt = gt.copy()
        bbox = gt["bbox"].clone() * scale   # (x,y,w,h) scaled

        # convert (x,y,w,h) → (x1,y1,x2,y2)
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        gt["bbox"] = bbox
        return gt