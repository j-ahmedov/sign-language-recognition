"""Signer-leakage guard. Non-negotiable (PLAN.md sections 5, 7 and 10).

Asserts that the signer-id intersection between any two of train/val/test is empty, and the
same for every leave-one-signer-out fold. Signer leakage silently inflates every number in
the thesis, so these tests run on *synthetic* metadata too -- they must pass in CI, where
the 1.5 GB dataset does not exist, not only on the machine that happens to have it.
"""

from __future__ import annotations

import itertools

import pytest

from conftest import synthetic_records
from signadapt.data.dataset import (
    SIGN_NAMES,
    SIGN_TABLE,
    ClipRecord,
    Split,
    assert_disjoint,
    feasible_k_values,
    is_two_handed,
    kshot_indices,
    load_records,
    loso_folds,
    make_splits,
    max_k,
    parse_clip_id,
)
from signadapt.utils.config import load_config

CACHE_DIR = "data/cache/lsa64"


def signer_sets(records, split):
    return {
        part: {records[i].signer for i in getattr(split, part)} for part in ("train", "val", "test")
    }


# ------------------------------------------------------- the guard, on synthetic metadata


def test_signer_independent_split_has_no_signer_overlap():
    records = synthetic_records()
    cfg = load_config("configs/data.yaml")
    split = make_splits(records, cfg)
    sets = signer_sets(records, split)

    for a, b in itertools.combinations(sets, 2):
        assert not sets[a] & sets[b], f"signer leakage between {a} and {b}: {sets[a] & sets[b]}"


def test_signer_independent_split_covers_every_signer_exactly_once():
    records = synthetic_records()
    cfg = load_config("configs/data.yaml")
    split = make_splits(records, cfg)
    sets = signer_sets(records, split)
    assert sets["train"] | sets["val"] | sets["test"] == {r.signer for r in records}
    assert sum(len(s) for s in sets.values()) == len({r.signer for r in records})


def test_every_loso_fold_has_no_signer_overlap():
    records = synthetic_records()
    folds = loso_folds(records)
    assert len(folds) == 10

    for fold in folds:
        sets = signer_sets(records, fold)
        assert len(sets["test"]) == 1, "a LOSO fold must hold out exactly one signer"
        for a, b in itertools.combinations(sets, 2):
            assert not sets[a] & sets[b], f"{fold.name}: leakage between {a} and {b}"


def test_loso_holds_out_each_signer_exactly_once():
    records = synthetic_records()
    held_out = [next(iter(signer_sets(records, f)["test"])) for f in loso_folds(records)]
    assert sorted(held_out) == sorted({r.signer for r in records})


def test_loso_train_never_contains_the_held_out_signer():
    records = synthetic_records()
    for fold in loso_folds(records):
        test_signer = next(iter(signer_sets(records, fold)["test"]))
        for index in fold.train:
            assert records[index].signer != test_signer, (
                f"{fold.name} trains on the held-out signer"
            )


def test_no_clip_appears_in_two_parts():
    """Disjoint signers imply disjoint clips, but assert it directly rather than infer it."""
    records = synthetic_records()
    cfg = load_config("configs/data.yaml")
    for split in [make_splits(records, cfg), *loso_folds(records)]:
        parts = [set(split.train), set(split.val), set(split.test)]
        for a, b in itertools.combinations(parts, 2):
            assert not a & b


def test_signer_dependent_split_shares_signers_but_not_clips():
    """E1 is the optimistic ceiling: signers overlap on purpose, clips must not."""
    records = synthetic_records()
    cfg = load_config("configs/data.yaml")
    cfg["splits"]["mode"] = "signer_dependent"
    split = make_splits(records, cfg, seed=0)
    sets = signer_sets(records, split)

    assert sets["train"] == sets["test"], "E1 is meant to share signers across parts"
    parts = [set(split.train), set(split.val), set(split.test)]
    for a, b in itertools.combinations(parts, 2):
        assert not a & b, "E1 must still never reuse the same clip"
    assert sum(len(p) for p in parts) == len(records)


def test_assert_disjoint_actually_catches_leakage():
    """A guard that cannot fail is not a guard."""
    leaky = Split(
        name="leaky",
        train=(0,),
        val=(1,),
        test=(2,),
        signers={"train": (1, 2), "val": (3,), "test": (2,)},
    )
    with pytest.raises(AssertionError, match="signer leakage"):
        assert_disjoint(leaky)


# ------------------------------------------------------------------------- id parsing


@pytest.mark.parametrize(
    ("clip_id", "expected"),
    [("001_001_001", (0, 1, 1)), ("064_010_005", (63, 10, 5)), ("042_003_002", (41, 3, 2))],
)
def test_parse_clip_id(clip_id, expected):
    assert parse_clip_id(clip_id) == expected


@pytest.mark.parametrize("bad", ["1_1_1", "001_001", "abc_001_001", "001-001-001", ""])
def test_parse_clip_id_rejects_malformed(bad):
    """Silently mis-parsing a signer id is how leakage gets in; it must raise."""
    with pytest.raises(ValueError, match="LSA64"):
        parse_clip_id(bad)


# ------------------------------------------------------- signer 10 is two different people


def _record(clip_id) -> ClipRecord:
    label, signer, rep = parse_clip_id(clip_id)
    return ClipRecord(
        clip_id=clip_id,
        path=f"/nonexistent/{clip_id}.npy",
        label=label,
        signer=signer,
        repetition=rep,
        n_frames=64,
        aspect=16 / 9,
    )


@pytest.mark.parametrize(
    ("clip_id", "session"),
    [("001_010_001", 1), ("023_010_001", 1), ("024_010_001", 2), ("064_010_001", 2)],
)
def test_session_boundary_is_between_signs_23_and_24(clip_id, session):
    assert _record(clip_id).session == session


def test_signer_10_gets_two_participant_ids():
    """The recorded subject changed between sessions; participant must reflect that."""
    assert _record("001_010_001").participant == "10a"
    assert _record("064_010_001").participant == "10b"


def test_other_signers_keep_one_participant_id():
    assert _record("001_003_001").participant == _record("064_003_001").participant == "3"


# ------------------------------------------------------------------- LSA64 sign table


def test_sign_table_matches_the_documented_session_structure():
    """LSA64 documents 23 one-handed signs in session 1, then 22 two-handed + 19 one-handed."""
    assert len(SIGN_TABLE) == len(SIGN_NAMES) == 64
    session_1 = [label for label in range(64) if label <= 22]
    assert not any(is_two_handed(label) for label in session_1), "session 1 is all one-handed"
    session_2_two = [label for label in range(23, 64) if is_two_handed(label)]
    assert len(session_2_two) == 22
    assert len(range(23, 64)) - len(session_2_two) == 19


def test_sign_names_are_unique_and_non_empty():
    assert len(set(SIGN_NAMES)) == 64
    assert all(name.strip() for name in SIGN_NAMES)


@pytest.mark.parametrize(
    ("label", "expected"),
    [(0, False), (22, False), (28, True), (29, False), (56, True), (63, False)],
)
def test_is_two_handed(label, expected):
    assert is_two_handed(label) is expected


# ------------------------------------------------------------------------- k-shot draw


def test_kshot_is_nested_across_k():
    """k=4's support set must extend k=2's, or the adaptation curve measures luck."""
    records = synthetic_records()
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    for small_k, large_k in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        small, _ = kshot_indices(records, candidates, small_k, seed=0)
        large, _ = kshot_indices(records, candidates, large_k, seed=0)
        assert set(small) <= set(large), f"k={small_k} is not a prefix of k={large_k}"


def test_kshot_query_set_is_identical_for_every_k():
    """Every point on the adaptation curve must be measured on the same examples."""
    records = synthetic_records()
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    queries = {k: kshot_indices(records, candidates, k, seed=0)[1] for k in range(5)}
    assert len(set(queries.values())) == 1, "the query set moved as k changed"


def test_kshot_query_set_depends_on_the_seed():
    records = synthetic_records()
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    assert (
        kshot_indices(records, candidates, 1, seed=0)[1]
        != kshot_indices(records, candidates, 1, seed=1)[1]
    )


def test_kshot_is_balanced_and_disjoint_from_query():
    records = synthetic_records(n_classes=8, n_reps=5)
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    support, query = kshot_indices(records, candidates, 2, seed=0)

    counts: dict[int, int] = {}
    for index in support:
        counts[records[index].label] = counts.get(records[index].label, 0) + 1
    assert set(counts.values()) == {2}, "k-shot must draw k examples of every class"
    assert not set(support) & set(query), "support and query must never share a clip"
    assert len(query) == 8, "one reserved query clip per class"
    assert set(support) | set(query) <= set(candidates)


def test_kshot_zero_has_no_support_but_still_has_a_query_set():
    """k=0 is the zero-shot point of the curve; it still needs something to evaluate on."""
    records = synthetic_records()
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    support, query = kshot_indices(records, candidates, 0, seed=0)
    assert support == ()
    assert len(query) == 8


def test_kshot_support_comes_only_from_the_held_out_signer():
    records = synthetic_records()
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    support, query = kshot_indices(records, candidates, 3, seed=0)
    assert {records[i].signer for i in support} == {7}
    assert {records[i].signer for i in query} == {7}


def test_kshot_refuses_a_k_the_data_cannot_supply():
    """Silently truncating would label a k=10 point that was really measured at k=4."""
    records = synthetic_records(n_reps=5)
    candidates = [i for i, r in enumerate(records) if r.signer == 7]
    with pytest.raises(ValueError, match="smallest class has 5"):
        kshot_indices(records, candidates, 5, seed=0)


def test_max_k_reflects_the_repetitions_available():
    assert max_k(synthetic_records(n_reps=5)) == 4
    assert max_k(synthetic_records(n_reps=5), query_repetitions=2) == 3


def test_feasible_k_values_splits_the_configured_sweep():
    """PLAN.md section 6 sweeps to k=20; LSA64 has 5 clips per signer and sign."""
    feasible, skipped = feasible_k_values(synthetic_records(n_reps=5), [0, 1, 2, 3, 5, 10, 20])
    assert feasible == (0, 1, 2, 3)
    assert skipped == (5, 10, 20)


@pytest.mark.needs_data
def test_the_real_dataset_caps_k_at_four():
    records = load_records(CACHE_DIR)
    assert max_k(records) == 4, "LSA64 records exactly 5 repetitions per signer and sign"


# ------------------------------------------------- the same guard, on the real dataset


@pytest.mark.needs_data
def test_real_dataset_splits_have_no_signer_overlap():
    """Re-run the guard against the actual extracted LSA64 manifest."""
    records = load_records(CACHE_DIR)
    cfg = load_config("configs/data.yaml")
    assert len(records) > 0

    for split in [make_splits(records, cfg), *loso_folds(records)]:
        sets = signer_sets(records, split)
        for a, b in itertools.combinations(sets, 2):
            assert not sets[a] & sets[b], f"{split.name}: leakage between {a} and {b}"


@pytest.mark.needs_data
def test_real_dataset_is_complete_and_well_formed():
    records = load_records(CACHE_DIR)
    assert len(records) == 3200, f"LSA64 has 3200 clips, manifest has {len(records)}"
    assert sorted({r.signer for r in records}) == list(range(1, 11))
    assert sorted({r.label for r in records}) == list(range(64))
    assert len({r.clip_id for r in records}) == len(records)
