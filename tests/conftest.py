"""Shared test fixtures.

``synthetic_records`` lives here rather than in one test module because three suites need
LSA64-shaped metadata without the dataset on disk, and a test that quietly requires the cache
passes on the machine that built it and fails everywhere else -- which is how the pretraining
cache test in ``test_personalize.py`` stayed red in CI from phase 5 onward.
"""

from __future__ import annotations

from signadapt.data.dataset import ClipRecord


def synthetic_records(n_signers=10, n_classes=8, n_reps=5):
    """Build LSA64-shaped metadata without needing the dataset on disk."""
    records = []
    for signer in range(1, n_signers + 1):
        for label in range(n_classes):
            for rep in range(1, n_reps + 1):
                clip_id = f"{label + 1:03d}_{signer:03d}_{rep:03d}"
                records.append(
                    ClipRecord(
                        clip_id=clip_id,
                        path=f"/nonexistent/{clip_id}.npy",  # never read by these tests
                        label=label,
                        signer=signer,
                        repetition=rep,
                        n_frames=64,
                        aspect=16 / 9,
                    )
                )
    return records
