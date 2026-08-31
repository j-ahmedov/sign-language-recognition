"""torch Dataset over cached keypoints, plus signer-aware split construction.

Signer leakage -- the same signer appearing in both training and evaluation data -- silently
inflates every number this project reports and invalidates the thesis (PLAN.md sections 5
and 10). The defence here is structural rather than procedural: every split is built by
partitioning the *signer* set first and then selecting clips, so a split with overlapping
signers cannot be expressed by this API. :func:`assert_disjoint` re-checks it anyway, and
``tests/test_splits.py`` checks it again on real metadata.

LSA64 filename convention: ``NNN_SSS_RRR.mp4`` -> sign ``NNN`` (1-64), signer ``SSS``
(1-10), repetition ``RRR`` (1-5). Labels are converted to 0-based class indices.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from signadapt.data.normalize import config_kwargs, normalize_clip
from signadapt.utils.seeding import temporary_seed

__all__ = [
    "ClipRecord",
    "KeypointDataset",
    "Split",
    "assert_disjoint",
    "kshot_indices",
    "load_records",
    "loso_folds",
    "make_splits",
    "parse_clip_id",
]

_LSA64_RE = re.compile(r"^(?P<sign>\d{3})_(?P<signer>\d{3})_(?P<rep>\d{3})$")

#: LSA64 was recorded in two sessions: signs 1-23 outdoors in natural light, signs 24-64
#: indoors under artificial light. Sign 23 (0-based label 22) is the last of session 1.
SESSION_1_LAST_LABEL = 22

#: The one signer id that does not identify one person. LSA64's documentation states that
#: subject 10 was unavailable for the second recording session and was replaced by a
#: different person, and this is plainly visible in the video: signer 10 is a bearded man in
#: signs 1-23 and a long-haired woman in signs 24-64. Every other signer is the same person
#: in both sessions (verified by eye on signers 3 and 7).
#:
#: This matters here more than it did for the dataset's original purpose. One client per
#: signer means client 10 is two people sharing a private classifier head, and the
#: leave-one-signer-out fold for signer 10 measures generalization to two unseen people at
#: once. Set ``splits.exclude_signers: [10]`` in configs/data.yaml to drop it.
SPLIT_IDENTITY_SIGNERS: tuple[int, ...] = (10,)

# Clips per class reserved from a held-out signer as the fixed evaluation set for the k-shot
# sweep. LSA64 records exactly 5 repetitions of every sign by every signer, so reserving 1
# caps k at 4 -- PLAN.md section 6's sweep up to k=20 is not realizable on this dataset.
# See :func:`kshot_indices` and :func:`max_k`.
QUERY_REPETITIONS = 1

#: The 64 LSA64 sign glosses, in dataset order, and whether each is signed with one hand
#: ("R", all subjects are right-handed) or both ("B"). Transcribed from the sign table at
#: https://facundoq.github.io/datasets/lsa64/. Needed for two things beyond documentation:
#: measuring whether the hands a sign *actually uses* were detected, and captioning the live
#: demo. Index 0 is sign 01.
SIGN_TABLE: tuple[tuple[str, str], ...] = (
    ("Opaque", "R"), ("Red", "R"), ("Green", "R"), ("Yellow", "R"),
    ("Bright", "R"), ("Light-blue", "R"), ("Colors", "R"), ("Pink", "R"),
    ("Women", "R"), ("Enemy", "R"), ("Son", "R"), ("Man", "R"),
    ("Away", "R"), ("Drawer", "R"), ("Born", "R"), ("Learn", "R"),
    ("Call", "R"), ("Skimmer", "R"), ("Bitter", "R"), ("Sweet milk", "R"),
    ("Milk", "R"), ("Water", "R"), ("Food", "R"), ("Argentina", "R"),
    ("Uruguay", "R"), ("Country", "R"), ("Last name", "R"), ("Where", "R"),
    ("Mock", "B"), ("Birthday", "R"), ("Breakfast", "B"), ("Photo", "B"),
    ("Hungry", "R"), ("Map", "B"), ("Coin", "B"), ("Music", "B"),
    ("Ship", "R"), ("None", "R"), ("Name", "R"), ("Patience", "R"),
    ("Perfume", "R"), ("Deaf", "R"), ("Trap", "B"), ("Rice", "B"),
    ("Barbecue", "B"), ("Candy", "R"), ("Chewing-gum", "R"), ("Spaghetti", "B"),
    ("Yogurt", "B"), ("Accept", "B"), ("Thanks", "B"), ("Shut down", "R"),
    ("Appear", "B"), ("To land", "B"), ("Catch", "B"), ("Help", "B"),
    ("Dance", "B"), ("Bathe", "B"), ("Buy", "R"), ("Copy", "B"),
    ("Run", "B"), ("Realize", "R"), ("Give", "B"), ("Find", "R"),
)  # fmt: skip

#: Class names indexed by 0-based label.
SIGN_NAMES: tuple[str, ...] = tuple(name for name, _ in SIGN_TABLE)

#: 0-based labels of the 22 signs performed with both hands.
TWO_HANDED_LABELS: frozenset[int] = frozenset(
    index for index, (_, hands) in enumerate(SIGN_TABLE) if hands == "B"
)

assert len(SIGN_TABLE) == 64, "the LSA64 sign table must have 64 entries"
assert len(TWO_HANDED_LABELS) == 22, "LSA64 documents 22 two-handed signs"


def is_two_handed(label: int) -> bool:
    """Return whether a 0-based class label is a two-handed sign.

    Args:
        label: 0-based class index.

    Returns:
        True if the sign is performed with both hands.
    """
    return label in TWO_HANDED_LABELS


@dataclass(frozen=True)
class ClipRecord:
    """One cached clip and its metadata.

    Attributes:
        clip_id: Filename stem, e.g. ``"001_007_003"``.
        path: Path to the cached ``.npy``.
        label: 0-based class index.
        signer: 1-based signer id as given by the dataset.
        repetition: 1-based repetition index.
        n_frames: Frames in the cached array.
        aspect: ``width / height`` of the source video, needed for normalization.
    """

    clip_id: str
    path: Path
    label: int
    signer: int
    repetition: int
    n_frames: int
    aspect: float

    @property
    def session(self) -> int:
        """Return the recording session (1 or 2) this clip belongs to."""
        return 1 if self.label <= SESSION_1_LAST_LABEL else 2

    @property
    def participant(self) -> str:
        """Return an id that identifies a *person*, unlike :attr:`signer`.

        For every signer but 10 this is just the signer id. Signer 10 becomes ``"10a"`` in
        session 1 and ``"10b"`` in session 2 -- see :data:`SPLIT_IDENTITY_SIGNERS`. Use this
        wherever the question is "is this the same human", and :attr:`signer` wherever the
        question is "is this the same federated client".
        """
        if self.signer in SPLIT_IDENTITY_SIGNERS:
            return f"{self.signer}{'a' if self.session == 1 else 'b'}"
        return str(self.signer)


@dataclass(frozen=True)
class Split:
    """A train/val/test partition, identified by signer sets as well as clip indices.

    Attributes:
        name: Human-readable name, e.g. ``"signer_independent"`` or ``"loso-signer07"``.
        train: Indices into the record list.
        val: Indices into the record list.
        test: Indices into the record list.
        signers: Signer ids per split part, used by :func:`assert_disjoint`.
    """

    name: str
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    signers: dict[str, tuple[int, ...]]

    def sizes(self) -> dict[str, int]:
        """Return the clip count of each part."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def parse_clip_id(clip_id: str) -> tuple[int, int, int]:
    """Parse an LSA64 clip id into ``(label, signer, repetition)``.

    Args:
        clip_id: Filename stem such as ``"042_003_005"``.

    Returns:
        ``(label, signer, repetition)`` with a 0-based label and 1-based signer/repetition.

    Raises:
        ValueError: If the id does not match the LSA64 convention. Failing loudly matters:
            a silently mis-parsed signer id is exactly how leakage gets in.
    """
    match = _LSA64_RE.match(clip_id)
    if not match:
        raise ValueError(f"clip id {clip_id!r} does not match LSA64's NNN_SSS_RRR convention")
    return (
        int(match["sign"]) - 1,
        int(match["signer"]),
        int(match["rep"]),
    )


def load_records(cache_dir: str | Path, *, exclude_signers: Iterable[int] = ()) -> list[ClipRecord]:
    """Load the extraction manifest and build one :class:`ClipRecord` per cached clip.

    Args:
        cache_dir: Directory holding ``manifest.json`` and the ``.npy`` files.
        exclude_signers: Signer ids to drop entirely. Typically ``[10]`` -- see
            :data:`SPLIT_IDENTITY_SIGNERS`.

    Returns:
        Records sorted by clip id, so the order is deterministic across machines.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """
    cache = Path(cache_dir)
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run `python -m signadapt.data.keypoints` first"
        )
    manifest = json.loads(manifest_path.read_text())

    dropped = set(exclude_signers)
    records: list[ClipRecord] = []
    for clip_id, entry in sorted(manifest["clips"].items()):
        label, signer, repetition = parse_clip_id(clip_id)
        if signer in dropped:
            continue
        records.append(
            ClipRecord(
                clip_id=clip_id,
                path=cache / entry["path"],
                label=label,
                signer=signer,
                repetition=repetition,
                n_frames=int(entry["n_frames"]),
                aspect=float(entry["width"]) / float(entry["height"]),
            )
        )
    return records


def signers_of(records: Sequence[ClipRecord], indices: Iterable[int]) -> tuple[int, ...]:
    """Return the sorted distinct signer ids covered by a set of record indices.

    Args:
        records: All records.
        indices: Indices into ``records``.

    Returns:
        Sorted unique signer ids.
    """
    return tuple(sorted({records[i].signer for i in indices}))


def assert_disjoint(split: Split) -> None:
    """Raise if any two parts of a split share a signer.

    This is the guard that PLAN.md section 10 calls non-negotiable. It is cheap, so it runs
    on construction of every split rather than only in the test suite.

    Args:
        split: The split to check.

    Raises:
        AssertionError: If any pair of parts shares a signer id.
    """
    parts = {k: set(v) for k, v in split.signers.items() if v}
    names = sorted(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = parts[a] & parts[b]
            if overlap:
                raise AssertionError(
                    f"signer leakage in split {split.name!r}: signers {sorted(overlap)} "
                    f"appear in both {a!r} and {b!r}"
                )


def _indices_for(records: Sequence[ClipRecord], signers: Iterable[int]) -> tuple[int, ...]:
    """Return record indices belonging to the given signers."""
    wanted = set(signers)
    return tuple(i for i, r in enumerate(records) if r.signer in wanted)


def make_splits(
    records: Sequence[ClipRecord],
    cfg: dict[str, Any],
    *,
    seed: int = 0,
) -> Split:
    """Build the split described by ``splits.mode`` in ``configs/data.yaml``.

    Two modes exist and they answer different questions (PLAN.md section 6):

    * ``signer_independent`` (E2): train, val and test signers are disjoint sets. This is
      the setting the thesis is about.
    * ``signer_dependent`` (E1): every signer appears in every part, split by *repetition*
      so that no individual clip is shared. This is the optimistic ceiling, and its signer
      sets deliberately overlap -- :func:`assert_disjoint` is not applied to it.

    Args:
        records: All records.
        cfg: Loaded ``configs/data.yaml``.
        seed: Seed for the repetition shuffle in signer-dependent mode.

    Returns:
        The constructed :class:`Split`.

    Raises:
        ValueError: On an unknown mode, or when a configured signer has no clips.
    """
    spec = cfg["splits"]
    mode = spec["mode"]

    if mode == "signer_independent":
        parts = {name: tuple(spec[f"{name}_signers"]) for name in ("train", "val", "test")}
        available = {r.signer for r in records}
        for name, signers in parts.items():
            missing = set(signers) - available
            if missing:
                raise ValueError(f"{name} signers {sorted(missing)} have no clips in the cache")
        split = Split(
            name="signer_independent",
            train=_indices_for(records, parts["train"]),
            val=_indices_for(records, parts["val"]),
            test=_indices_for(records, parts["test"]),
            signers={k: tuple(sorted(v)) for k, v in parts.items()},
        )
        assert_disjoint(split)
        return split

    if mode == "signer_dependent":
        # Every signer is in every part; hold out whole repetitions so that no clip and no
        # (signer, sign, repetition) triple is shared between parts.
        reps = sorted({r.repetition for r in records})
        with temporary_seed(seed):
            order = list(np.random.permutation(reps))
        n_test = max(1, len(order) // 5)
        test_reps, val_reps = set(order[:n_test]), set(order[n_test : n_test + 1])
        train_reps = set(order[n_test + 1 :])
        by_rep = {
            "train": tuple(i for i, r in enumerate(records) if r.repetition in train_reps),
            "val": tuple(i for i, r in enumerate(records) if r.repetition in val_reps),
            "test": tuple(i for i, r in enumerate(records) if r.repetition in test_reps),
        }
        return Split(
            name="signer_dependent",
            **by_rep,
            signers={k: signers_of(records, v) for k, v in by_rep.items()},
        )

    raise ValueError(f"unknown splits.mode: {mode!r}")


def loso_folds(
    records: Sequence[ClipRecord],
    *,
    val_signers: int = 1,
    seed: int = 0,
) -> list[Split]:
    """Build one leave-one-signer-out fold per signer.

    In each fold exactly one signer is held out for test; ``val_signers`` further signers are
    held out for validation, and the rest train. Validation signers are drawn deterministically
    from the seed so the folds are reproducible, and they are never the test signer.

    Args:
        records: All records.
        val_signers: How many signers to reserve for validation in each fold.
        seed: Seed for choosing the validation signers.

    Returns:
        One :class:`Split` per signer, each already checked for disjointness.

    Raises:
        ValueError: If there are too few signers to form a fold.
    """
    all_signers = sorted({r.signer for r in records})
    if len(all_signers) < val_signers + 2:
        raise ValueError(
            f"need at least {val_signers + 2} signers for LOSO with {val_signers} "
            f"validation signers, have {len(all_signers)}"
        )

    folds: list[Split] = []
    for held_out in all_signers:
        rest = [s for s in all_signers if s != held_out]
        with temporary_seed(seed + held_out):
            chosen = np.random.permutation(rest)[:val_signers]
        val = tuple(sorted(int(s) for s in chosen))
        train = tuple(s for s in rest if s not in set(val))
        split = Split(
            name=f"loso-signer{held_out:02d}",
            train=_indices_for(records, train),
            val=_indices_for(records, val),
            test=_indices_for(records, [held_out]),
            signers={"train": train, "val": val, "test": (held_out,)},
        )
        assert_disjoint(split)
        folds.append(split)
    return folds


def kshot_indices(
    records: Sequence[ClipRecord],
    candidates: Sequence[int],
    k: int,
    *,
    seed: int = 0,
    query_repetitions: int = QUERY_REPETITIONS,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Draw a k-shot support set per class against a query set that does not depend on k.

    Two properties matter for the adaptation curve in PLAN.md section 6, and one permutation
    per class delivers both:

    * **The query set is fixed across k.** ``query_repetitions`` clips per class are reserved
      before any support is drawn, so top-1 at k=1 and at k=3 are measured on the same
      examples. Letting the query set shrink as k grows -- the obvious implementation -- means
      each point on the curve is measured on a different test set, and a rising curve could
      then be an easier test set rather than a better model.
    * **The support set is nested across k.** The support for k=3 starts with the same clips
      as the one for k=2 (same seed), so a rising curve reflects more data and not a luckier
      draw.

    Args:
        records: All records.
        candidates: Indices of the held-out signer's clips to draw from.
        k: Support examples per class; ``k=0`` returns an empty support set.
        seed: Seed for the per-class permutation.
        query_repetitions: Clips per class reserved for evaluation.

    Returns:
        ``(support, query)`` index tuples, disjoint by construction. Clips that are neither
        reserved nor drawn are in neither -- they are the budget k has not spent yet.

    Raises:
        ValueError: If some class cannot supply ``query_repetitions + k`` clips. Silently
            truncating would put a point on the adaptation curve labelled ``k=10`` that was
            actually measured at k=4, which is worse than not having the point.
    """
    by_class: dict[int, list[int]] = {}
    for index in candidates:
        by_class.setdefault(records[index].label, []).append(index)

    smallest = min((len(pool) for pool in by_class.values()), default=0)
    if by_class and query_repetitions + k > smallest:
        raise ValueError(
            f"k={k} with {query_repetitions} reserved query clips needs "
            f"{query_repetitions + k} clips per class, but the smallest class has {smallest}. "
            f"On LSA64 every (signer, sign) pair has exactly 5 clips, so k is capped at "
            f"{smallest - query_repetitions}."
        )

    support: list[int] = []
    query: list[int] = []
    for label in sorted(by_class):
        pool = sorted(by_class[label])
        # One permutation, seeded independently of k: the first `query_repetitions` entries
        # are the fixed query set, and the support is a nested prefix of what remains.
        with temporary_seed(seed * 1000 + label):
            order = np.random.permutation(len(pool))
        query.extend(pool[i] for i in order[:query_repetitions])
        support.extend(pool[i] for i in order[query_repetitions : query_repetitions + k])

    return tuple(sorted(support)), tuple(sorted(query))


def max_k(records: Sequence[ClipRecord], *, query_repetitions: int = QUERY_REPETITIONS) -> int:
    """Return the largest k that :func:`kshot_indices` can honour for every signer and class.

    Args:
        records: All records.
        query_repetitions: Clips per class reserved for evaluation.

    Returns:
        ``min clips per (signer, class) - query_repetitions``; 4 on LSA64.
    """
    counts: dict[tuple[int, int], int] = {}
    for record in records:
        counts[(record.signer, record.label)] = counts.get((record.signer, record.label), 0) + 1
    return min(counts.values(), default=0) - query_repetitions


def feasible_k_values(
    records: Sequence[ClipRecord],
    k_values: Iterable[int],
    *,
    query_repetitions: int = QUERY_REPETITIONS,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a configured k sweep into the values this dataset can supply and those it cannot.

    Args:
        records: All records.
        k_values: The configured sweep, e.g. ``configs/fl.yaml``'s ``[0, 1, 2, 3, 5, 10, 20]``.
        query_repetitions: Clips per class reserved for evaluation.

    Returns:
        ``(feasible, skipped)``, both sorted.
    """
    limit = max_k(records, query_repetitions=query_repetitions)
    wanted = sorted(set(k_values))
    return tuple(k for k in wanted if k <= limit), tuple(k for k in wanted if k > limit)


class KeypointDataset(Dataset):
    """Serves normalized fixed-length keypoint tensors from the ``.npy`` cache.

    Each item is ``(x, y)`` where ``x`` has shape ``(T, 115, 4)`` -- channels
    ``[x, y, z, valid]`` -- and ``y`` is the class index. The validity channel is carried all
    the way to the model on purpose: it is what lets the encoder distinguish "hand absent"
    from "hand at the origin" (see :mod:`signadapt.data.normalize`).

    Attributes:
        records: The records this dataset serves.
        indices: Indices into ``records`` that are actually served.
    """

    def __init__(
        self,
        records: Sequence[ClipRecord],
        indices: Sequence[int],
        cfg: dict[str, Any],
        *,
        mirror: bool = False,
        cache_in_memory: bool = True,
    ) -> None:
        """Build a dataset view over a subset of records.

        Args:
            records: All records.
            indices: Which records this view serves.
            cfg: Loaded ``configs/data.yaml``.
            mirror: Apply handedness mirroring to every clip.
            cache_in_memory: Keep normalized tensors in RAM. All of LSA64 at T=64 is about
                3200 x 64 x 115 x 4 x 4 B = 377 MB, which fits the 16 GB budget comfortably.
        """
        self.records = list(records)
        self.indices = list(indices)
        self._cfg = cfg
        self._kwargs = config_kwargs(cfg)
        self._mirror = mirror
        self._cache: dict[int, torch.Tensor] | None = {} if cache_in_memory else None

    def __len__(self) -> int:
        """Return the number of clips served."""
        return len(self.indices)

    @property
    def signers(self) -> tuple[int, ...]:
        """Return the sorted distinct signer ids served by this dataset."""
        return signers_of(self.records, self.indices)

    @property
    def labels(self) -> list[int]:
        """Return the class index of every served clip, in order."""
        return [self.records[i].label for i in self.indices]

    def _load(self, record_index: int) -> torch.Tensor:
        """Load, normalize and cache one clip."""
        if self._cache is not None and record_index in self._cache:
            return self._cache[record_index]
        record = self.records[record_index]
        raw = np.load(record.path)
        array = normalize_clip(raw, aspect=record.aspect, do_mirror=self._mirror, **self._kwargs)
        tensor = torch.from_numpy(array)
        if self._cache is not None:
            self._cache[record_index] = tensor
        return tensor

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        """Return ``(keypoints, label)`` for the ``item``-th served clip."""
        record_index = self.indices[item]
        return self._load(record_index), self.records[record_index].label
