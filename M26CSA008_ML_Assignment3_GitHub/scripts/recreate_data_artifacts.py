"""Recreate deterministic dataset/index artifacts from the original assignment setup.
Run from the repository root after installing requirements.
This does not retrain models or regenerate model checkpoints.
"""
import os
import numpy as np
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

ROLL_NO = "M26CSA008"
SEED = int("".join(ch for ch in ROLL_NO if ch.isdigit()))
os.makedirs("results", exist_ok=True)

train_data = CIFAR10(root="./data", train=True, download=True)
y = np.asarray(train_data.targets)
train_idx, val_idx = train_test_split(
    np.arange(len(y)), test_size=5000, stratify=y, random_state=SEED
)
np.savez(f"results/split_seed{SEED}.npz", train_idx=train_idx, val_idx=val_idx)

# The notebook chooses these subsets from positions within the 45,000-item
# training split, not from the original 50,000-item CIFAR10 index space.
y_train = y[train_idx]
train_positions = np.arange(len(y_train))

mem_idx, _ = train_test_split(
    train_positions, train_size=10000, stratify=y_train, random_state=SEED
)
np.save("results/q1b_memorisation_subset_idx.npy", mem_idx)

perm = np.random.RandomState(SEED).permutation(32 * 32)
np.save("results/pixel_permutation.npy", perm)

svm_idx, _ = train_test_split(
    train_positions, train_size=6000, stratify=y_train, random_state=SEED
)
np.save("results/q4_svm_subset_idx.npy", svm_idx)

print("Created split, memorisation subset, pixel permutation and SVM subset.")
print("Train:", len(train_idx), "Validation:", len(val_idx))
print("Seed:", SEED)
