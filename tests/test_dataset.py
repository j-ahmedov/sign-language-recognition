"""Dataset tests against the real extracted LSA64 cache.

Marked ``needs_data`` because they read the cache; CI runs without the dataset and skips
them. The synthetic, always-run half of the data tests lives in ``test_splits.py`` and
``test_normalize.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signadapt.data.dataset import KeypointDataset, load_records, loso_folds, make_splits
from signadapt.utils.config import load_config
from signadapt.utils.seeding import seed_everything, torch_generator

CACHE_DIR = "data/cache/lsa64"

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/data.yaml")


@pytest.fixture(scope="module")
def records(cfg):
    return load_records(cfg["dataset"]["cache_dir"])


def test_dataset_item_shape_and_dtype(records, cfg):
    data = KeypointDataset(records, list(range(4)), cfg)
    x, y = data[0]

    assert x.shape == (cfg["temporal"]["n_frames"], cfg["landmarks"]["total"], 4)
    assert x.dtype == torch.float32
    assert isinstance(y, int)
    assert 0 <= y < cfg["dataset"]["n_classes"]


def test_dataset_never_yields_nan(records, cfg):
    """NaN reaching the optimizer would poison every weight in one step."""
    data = KeypointDataset(records, list(range(0, len(records), 337)), cfg)
    for index in range(len(data)):
        x, _ = data[index]
        assert not torch.isnan(x).any(), f"NaN in {data.records[data.indices[index]].clip_id}"


def test_validity_channel_is_binary_and_informative(records, cfg):
    data = KeypointDataset(records, list(range(0, len(records), 337)), cfg)
    seen = set()
    for index in range(len(data)):
        x, _ = data[index]
        seen |= set(torch.unique(x[..., 3]).tolist())
    assert seen <= {0.0, 1.0}
    assert 0.0 in seen, "some clip should have a missing landmark group; none did"


def test_normalized_coordinates_are_in_a_sane_range(records, cfg):
    """Units are shoulder-widths from the mid-shoulder: a hand 8 widths away is a bug."""
    data = KeypointDataset(records, list(range(0, len(records), 337)), cfg)
    biggest = 0.0
    for index in range(len(data)):
        x, _ = data[index]
        valid = x[..., 3] > 0.5
        biggest = max(biggest, float(x[..., :2][valid.unsqueeze(-1).expand(-1, -1, 2)].abs().max()))
    assert biggest < 8.0, f"largest normalized coordinate is {biggest:.1f} shoulder widths"


def test_dataset_is_deterministic(records, cfg):
    a = KeypointDataset(records, [5], cfg, cache_in_memory=False)[0][0]
    b = KeypointDataset(records, [5], cfg, cache_in_memory=False)[0][0]
    assert torch.equal(a, b)


def test_mirroring_changes_the_tensor(records, cfg):
    plain = KeypointDataset(records, [5], cfg)[0][0]
    flipped = KeypointDataset(records, [5], cfg, mirror=True)[0][0]
    assert not torch.equal(plain, flipped)


def test_dataloader_batches(records, cfg):
    seed_everything(0)
    split = make_splits(records, cfg)
    data = KeypointDataset(records, split.val[:16], cfg)
    loader = DataLoader(data, batch_size=8, shuffle=True, generator=torch_generator(0))

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (8, cfg["temporal"]["n_frames"], cfg["landmarks"]["total"], 4)
    assert batch_y.shape == (8,)


def test_split_datasets_have_disjoint_signers(records, cfg):
    split = make_splits(records, cfg)
    parts = {
        name: KeypointDataset(records, getattr(split, name), cfg).signers
        for name in ("train", "val", "test")
    }
    assert not set(parts["train"]) & set(parts["test"])
    assert not set(parts["train"]) & set(parts["val"])
    assert not set(parts["val"]) & set(parts["test"])


def test_every_class_appears_in_every_loso_training_set(records):
    """A fold that never sees a class cannot be scored on it."""
    for fold in loso_folds(records):
        labels = {records[i].label for i in fold.train}
        assert len(labels) == 64, f"{fold.name} trains on only {len(labels)} classes"


def test_split_sizes_are_as_expected(records, cfg):
    split = make_splits(records, cfg)
    sizes = split.sizes()
    assert sum(sizes.values()) == len(records) == 3200
    # 7 train / 1 val / 2 test signers x 64 signs x 5 repetitions
    assert sizes == {"train": 7 * 320, "val": 320, "test": 2 * 320}


def test_class_distribution_is_balanced(records):
    counts = np.bincount([r.label for r in records], minlength=64)
    assert counts.min() == counts.max() == 50, "LSA64 is 10 signers x 5 repetitions per sign"
