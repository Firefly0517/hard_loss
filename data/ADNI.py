import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class ADNIT1AgeDataset(Dataset):
    """
    root/
      ADNI/*.npy  -> T1 volume, shape (182,218,182)
      Age/*.npy   -> age scalar (or 0-d / 1-d array)
    Pairing is done by same basename (without extension).
    """
    def __init__(
        self,
        root_dir: str,
        t1_subdir: str = "ADNI",
        age_subdir: str = "Age",
        transform=None,
        mmap: bool = True,
        expected_shape=(182, 218, 182),
        add_channel_dim: bool = True,
        return_name: bool = False,
    ):
        self.root_dir = root_dir
        self.t1_dir = os.path.join(root_dir, t1_subdir)
        self.age_dir = os.path.join(root_dir, age_subdir)

        self.transform = transform
        self.mmap = mmap
        self.expected_shape = tuple(expected_shape)
        self.add_channel_dim = add_channel_dim
        self.return_name = return_name

        if not os.path.isdir(self.t1_dir):
            raise FileNotFoundError(f"T1 folder not found: {self.t1_dir}")
        if not os.path.isdir(self.age_dir):
            raise FileNotFoundError(f"Age folder not found: {self.age_dir}")

        # list all T1 files
        t1_files = sorted(glob.glob(os.path.join(self.t1_dir, "*.npy")))
        if len(t1_files) == 0:
            raise FileNotFoundError(f"No .npy files found in: {self.t1_dir}")

        # keep only those with matching age file
        self.items = []
        missing_age = 0
        for t1_path in t1_files:
            name = os.path.splitext(os.path.basename(t1_path))[0]
            age_path = os.path.join(self.age_dir, name + ".npy")
            if os.path.isfile(age_path):
                self.items.append((name, t1_path, age_path))
            else:
                missing_age += 1

        if len(self.items) == 0:
            raise RuntimeError("No matched (T1, Age) pairs found. Check filenames.")
        if missing_age > 0:
            print(f"[WARN] {missing_age} T1 files have no matching age file. They are skipped.")
        print(f"[INFO] Matched pairs: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _load_t1(self, path: str) -> np.ndarray:
        arr = np.load(path, mmap_mode="r" if self.mmap else None)

        # allow (182,218,182) or (1,182,218,182)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]

        if tuple(arr.shape) != self.expected_shape:
            raise ValueError(f"Bad T1 shape {arr.shape} in {path}, expected {self.expected_shape}")

        arr = np.asarray(arr, dtype=np.float32)
        if self.add_channel_dim:
            arr = arr[None, ...]  # (1,D,H,W)
        return arr

    def _load_age(self, path: str) -> float:
        a = np.load(path, mmap_mode="r" if self.mmap else None)

        # handle scalar / (1,) / (1,1) etc.
        a = np.asarray(a).reshape(-1)
        if a.size < 1:
            raise ValueError(f"Empty age file: {path}")
        return float(a[0])

    def __getitem__(self, idx):
        name, t1_path, age_path = self.items[idx]

        x = self._load_t1(t1_path)          # numpy (1,182,218,182)
        age = self._load_age(age_path)      # float

        if self.transform is not None:
            x = self.transform(x)

        x = torch.from_numpy(x)
        y = torch.tensor(age, dtype=torch.float32)

        if self.return_name:
            return x, y, name
        return x, y
