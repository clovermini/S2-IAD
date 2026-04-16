"""Dataset loaders used by the anomaly-detection training and evaluation code.

This module provides a unified dataset interface for metal anomaly datasets
that are indexed by a shared `meta.json`-style file.
"""

import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os



CLSNAMES = ["casting_billet", "steel_pipe", "KolektorSDD", "KolektorSDD2"]


CLSNAMES_map_index = {}
for k, index in zip(CLSNAMES, range(len(CLSNAMES))):
    CLSNAMES_map_index[k] = index


class MetalDataset(data.Dataset):
    """Dataset wrapper for metal anomaly datasets described by metadata files."""

    def __init__(
        self,
        root,
        meta_path,
        transform,
        target_transform,
        mode="test",
        k_shot=0,
        save_dir=None,
        obj_name=None,
    ):
        """Initialize the dataset from root paths and a metadata description.

        Args:
            root: Dataset root directory.
            meta_path: Path to the metadata JSON file.
            transform: Transform applied to the input image.
            target_transform: Transform applied to the mask.
            mode: Either `train` or `test`.
            k_shot: Number of samples to draw in training mode.
            save_dir: Optional directory used to save sampled support-image paths.
            obj_name: Target object name for training mode.
        """

        self.transform = transform
        self.target_transform = target_transform
        self.dataset_dir = root

        self.data_all = []
        meta_info = json.load(open(meta_path, "r"))

        # Select the requested split from the metadata file.
        meta_info = meta_info[mode]

        if mode == "train":
            self.cls_names = [obj_name]
        else:
            self.cls_names = CLSNAMES  # list(meta_info.keys())

        if mode == "train":
            save_dir = os.path.join(save_dir, "k_shot.txt")

        # In training mode, randomly sample k support images.
        # In test mode, load all samples from the selected classes.
        for cls_name in self.cls_names:
            if mode == "train":
                data_tmp = meta_info[cls_name]
                indices = torch.randint(0, len(data_tmp), (k_shot,))
                for i in range(len(indices)):
                    self.data_all.append(data_tmp[indices[i]])
                    with open(save_dir, "a") as f:
                        f.write(data_tmp[indices[i]]["img_path"] + "\n")
            else:
                print("cls_name ", cls_name)
                self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

    def __len__(self):
        """Return the number of loaded samples."""
        return self.length

    def get_cls_names(self):
        """Return the class names currently used by this dataset instance."""
        return self.cls_names

    def __getitem__(self, index):
        """Load one image-mask pair and return the model-ready sample dictionary."""
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = (
            data["img_path"],
            data["mask_path"],
            data["cls_name"],
            data["specie_name"],
            data["anomaly"],
        )

        # Load the image first, then build a zero mask for normal samples.
        img = Image.open(os.path.join(self.dataset_dir, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode="L")
        else:
            # Fall back to an empty mask if the expected mask file is missing.
            if not os.path.exists(os.path.join(self.dataset_dir, mask_path)):
                img_mask = Image.fromarray(
                    np.zeros((img.size[0], img.size[1])), mode="L"
                )
                raise
            else:
                # Convert the raw grayscale mask into a binary foreground mask.
                img_mask = (
                    np.array(
                        Image.open(os.path.join(self.dataset_dir, mask_path)).convert(
                            "L"
                        )
                    )
                    > 0
                )
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode="L")
        # Apply image and mask transforms after loading.
        img = self.transform(img) if self.transform is not None else img
        img_mask = (
            self.target_transform(img_mask)
            if self.target_transform is not None and img_mask is not None
            else img_mask
        )
        img_mask = [] if img_mask is None else img_mask

        # Return both image-level and class-level information for evaluation.
        return {
            "img": img,
            "img_mask": img_mask,
            "cls_name": cls_name,
            "anomaly": anomaly,
            "img_path": os.path.join(self.dataset_dir, img_path),
            "cls_id": CLSNAMES_map_index[cls_name],
        }
