"""Duplicate and near-duplicate detection algorithms for NeuroScan AI."""

import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from PIL import Image


def compute_md5(image_path: Path) -> str:
    """Compute MD5 hash of raw file content."""
    hasher = hashlib.md5()
    with open(image_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for perceptual similarity comparison.
    
    Args:
        image_path: Path to image file.
        hash_size: Grid width (generates hash of length hash_size * hash_size bits).
        
    Returns:
        Integer representing the perceptual hash.
    """
    with Image.open(image_path) as img:
        # Convert to grayscale and resize to (hash_size + 1, hash_size)
        gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = np.array(gray, dtype=np.int32)
        
    # Compare adjacent horizontal pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    
    # Pack bits into integer
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
        
    return hash_int


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculate the Hamming distance between two integer perceptual hashes."""
    x = hash1 ^ hash2
    return bin(x).count("1")


def detect_duplicates(
    records: List[Dict[str, Any]],
    check_near_duplicates: bool = True,
    hamming_threshold: int = 4,
    hash_size: int = 8
) -> Dict[str, Any]:
    """Detect both exact duplicates and potential near-duplicates across dataset splits.
    
    Args:
        records: List of image metadata dicts containing at least 'path', 'split', 'label'.
        check_near_duplicates: Whether to compute and compare perceptual hashes.
        hamming_threshold: Max hamming distance to consider as near-duplicate (0-64).
        hash_size: Dimension for dHash calculation.
        
    Returns:
        Structured dictionary of duplicate statistics, grouped pairs, and cross-split leakage warnings.
    """
    exact_hash_map = defaultdict(list)
    dhash_entries = []

    # 1. Exact Duplicate Detection via MD5
    for item in records:
        path = Path(item["path"])
        try:
            md5_val = item.get("md5") or compute_md5(path)
            item["md5"] = md5_val
            exact_hash_map[md5_val].append(item)
            
            if check_near_duplicates:
                phash = compute_dhash(path, hash_size=hash_size)
                dhash_entries.append((phash, item))
        except Exception as e:
            item["error"] = str(e)

    exact_duplicate_groups = [group for group in exact_hash_map.values() if len(group) > 1]
    
    exact_leakage = []
    exact_pairs = []
    for group in exact_duplicate_groups:
        splits = set(item.get("split") for item in group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                item_a, item_b = group[i], group[j]
                exact_pairs.append({
                    "path_a": str(item_a["path"]),
                    "path_b": str(item_b["path"]),
                    "split_a": item_a.get("split"),
                    "split_b": item_b.get("split"),
                    "label_a": item_a.get("label"),
                    "label_b": item_b.get("label"),
                    "md5": item_a["md5"],
                })
                if item_a.get("split") != item_b.get("split"):
                    exact_leakage.append({
                        "item_a": item_a["path"],
                        "item_b": item_b["path"],
                        "split_a": item_a.get("split"),
                        "split_b": item_b.get("split"),
                        "md5": item_a["md5"],
                    })

    # 2. Near Duplicate Detection via Perceptual dHash
    near_duplicate_pairs = []
    near_leakage = []
    if check_near_duplicates and len(dhash_entries) > 1:
        n = len(dhash_entries)
        # Compare pairs
        for i in range(n):
            hash_a, item_a = dhash_entries[i]
            for j in range(i + 1, n):
                hash_b, item_b = dhash_entries[j]
                
                # Skip if already identified as exact duplicate
                if item_a.get("md5") == item_b.get("md5"):
                    continue
                    
                dist = hamming_distance(hash_a, hash_b)
                if dist <= hamming_threshold:
                    pair_info = {
                        "path_a": str(item_a["path"]),
                        "path_b": str(item_b["path"]),
                        "split_a": item_a.get("split"),
                        "split_b": item_b.get("split"),
                        "label_a": item_a.get("label"),
                        "label_b": item_b.get("label"),
                        "hamming_distance": dist,
                    }
                    near_duplicate_pairs.append(pair_info)
                    if item_a.get("split") != item_b.get("split"):
                        near_leakage.append(pair_info)

    summary = {
        "total_images_analyzed": len(records),
        "exact_duplicate_groups_count": len(exact_duplicate_groups),
        "exact_duplicate_pairs_count": len(exact_pairs),
        "exact_cross_split_leakage_count": len(exact_leakage),
        "near_duplicate_pairs_count": len(near_duplicate_pairs),
        "near_cross_split_leakage_count": len(near_leakage),
        "exact_duplicate_pairs": exact_pairs[:50],  # cap for summary readability
        "exact_cross_split_leakage": exact_leakage,
        "near_duplicate_pairs": near_duplicate_pairs[:50],
        "near_cross_split_leakage": near_leakage,
    }
    return summary
