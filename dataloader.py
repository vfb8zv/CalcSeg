import torch
import numpy as np
import os
from torch.utils.data import Dataset
from scipy import ndimage


def std_img(tens):
    t_ = (tens-tens.min())/(tens.max()-tens.min()+1e-14)
    return t_


def resize_volume(img, ex=224, order=1):
    current_depth = img.shape[0]
    current_width = img.shape[1]            

    depth_factor = ex / current_depth
    width_factor = ex / current_width

    factors = (depth_factor, width_factor)

    return ndimage.zoom(img, factors, order=order)


class DataLoading_3D(Dataset):
    def __init__(self, data_dir, test_flag, image_size=224, subs=None):
        if not test_flag:
            self.data_dir = os.path.join(data_dir, "train")
        else:
            self.data_dir = os.path.join(data_dir, "test")

        self.image_size = image_size

        all_files = [f for f in os.listdir(self.data_dir) if "raw_" in f]

        subjects = {}
        for f in all_files:
            subject_id = "_".join(f.split("_")[:-1])
            subjects.setdefault(subject_id, []).append(f)

        self.subjects = []
        for subject_id, file_list in subjects.items():
            if subs is not None:
                if subject_id not in subs:
                    continue
            file_list = sorted(
                file_list,
                key=lambda x: int(x[:-4].split("_")[-1])
            )
            self.subjects.append((subject_id, file_list))

        self.labels = []
        for subject_id, file_list in self.subjects:
            if subs is not None:
                if subject_id not in subs:
                    continue
            lge_slices = []
            for f in file_list:
                lge_np = np.load(
                    os.path.join(self.data_dir, f.replace("raw_", "lge_"))
                )
                lge_np = resize_volume(lge_np, ex=self.image_size)
                lge_slices.append(lge_np)

            lge_vol = np.stack(lge_slices, axis=0)  # (D, H, W)
            label = 1 if np.sum(lge_vol) >= 1 else 0
            self.labels.append(label)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject, slice_files = self.subjects[idx]

        raw_slices = []
        lge_slices = []

        for f in slice_files:
            raw_np = np.load(os.path.join(self.data_dir, f))
            lge_np = np.load(
                os.path.join(self.data_dir, f.replace("raw_", "lge_"))
            )

            raw_slices.append(
                resize_volume(raw_np, ex=self.image_size, order=1)
            )
            lge_slices.append(
                resize_volume(lge_np, ex=self.image_size, order=0)
            )

        raw_vol = np.stack(raw_slices, axis=0)
        lge_vol = np.stack(lge_slices, axis=0)

        image = torch.from_numpy(raw_vol).float()
        seg = torch.from_numpy(np.nan_to_num(lge_vol)).float()

        image = std_img(image)
        seg = std_img(seg)

        seg[seg >= 0.5] = 1
        seg[seg < 0.5] = 0

        target_D = 12  ## change based on dataset
        D, H, W = image.shape

        pad_mask = torch.ones(D)

        if D < target_D:
            pad = target_D - D
            pad_img = torch.zeros((pad, H, W), dtype=image.dtype)
            pad_seg = torch.zeros((pad, H, W), dtype=seg.dtype)

            image = torch.cat([image, pad_img], dim=0)
            seg = torch.cat([seg, pad_seg], dim=0)
            pad_mask = torch.cat(
                [pad_mask, torch.zeros(pad, dtype=pad_mask.dtype)],
                dim=0
            )
        else:
            image = image[:target_D]
            seg = seg[:target_D]
            pad_mask = pad_mask[:target_D]
            # print('Truncated', subject)


        return (
            image.unsqueeze(1),             
            seg.unsqueeze(1),                   
            torch.tensor(self.labels[idx], dtype=torch.long),
            pad_mask,                           
            subject,  
        )
