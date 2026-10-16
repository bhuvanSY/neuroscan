"""PyTorch Dataset and DataLoader abstractions for NeuroScan AI."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.data.transforms import get_transforms
from src.data.constants import CLASS_TO_IDX, IDX_TO_CLASS
from src.utils.seed import seed_worker


class BrainMRIDataset(Dataset):
    """PyTorch Dataset for Brain Tumor MRI Scans with fixed label mapping."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            records: List of sample metadata dicts (must contain 'path' and 'label').
            transform: Optional torchvision transform pipeline.
            class_to_idx: Fixed mapping from string class label to integer index.
        """
        self.records = records
        self.transform = transform
        self.class_to_idx = class_to_idx or CLASS_TO_IDX
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.records[idx]
        img_path = item["path"]
        label_str = item["label"]
        
        if label_str not in self.class_to_idx:
            raise KeyError(f"Label '{label_str}' not in fixed class_to_idx mapping: {list(self.class_to_idx.keys())}")
            
        label_idx = self.class_to_idx[label_str]

        # Always convert to 3-channel RGB for neural network architectures
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            
        if self.transform is not None:
            image_tensor = self.transform(rgb_img)
        else:
            from torchvision.transforms import ToTensor
            image_tensor = ToTensor()(rgb_img)

        return image_tensor, label_idx, img_path


def create_dataloaders(
    records: List[Dict[str, Any]],
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    image_size: Tuple[int, int] = (224, 224),
    crop_contour: bool = True,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    seed: int = 42,
) -> Dict[str, Any]:
    """Create deterministic PyTorch DataLoaders for train, val, and test splits.
    
    Guarantees:
    - Fixed class-to-index mapping across all splits
    - Mild augmentation applied strictly to train split
    - Deterministic worker seeding
    """
    splits = {
        "train": [r for r in records if r.get("split") == "train"],
        "val": [r for r in records if r.get("split") == "val"],
        "test": [r for r in records if r.get("split") == "test"],
    }

    g = torch.Generator()
    g.manual_seed(seed)

    if not torch.cuda.is_available():
        pin_memory = False

    loaders = {}
    for split_name, split_records in splits.items():
        if not split_records:
            loaders[split_name] = None
            continue

        transform = get_transforms(
            split=split_name,
            image_size=image_size,
            crop_contour=crop_contour,
            mean=mean,
            std=std,
        )
        
        dataset = BrainMRIDataset(
            records=split_records,
            transform=transform,
            class_to_idx=CLASS_TO_IDX,
        )

        shuffle = (split_name == "train")
        
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=g if shuffle else None,
            drop_last=(split_name == "train" and len(dataset) > batch_size),
        )

    return {
        "loaders": loaders,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "dataset_splits": {k: len(v) for k, v in splits.items()},
    }
