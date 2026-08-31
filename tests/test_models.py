"""Phase 2: the encoder/head split, the parameter budget, and augmentation safety.

The parameter-group tests here are the precondition for `tests/test_fedper.py` in phase 4:
FedPer can only aggregate the encoder and not the head if that partition is unambiguous.
"""

from __future__ import annotations

import pytest
import torch

from signadapt.data.augment import augment_batch
from signadapt.models.encoder import TemporalTransformerEncoder, build_encoder
from signadapt.models.head import LinearHead, build_head
from signadapt.models.model import (
    ENCODER_PREFIX,
    HEAD_PREFIX,
    SignAdaptModel,
    build_model,
    group_state_dict,
)
from signadapt.train.loop import cosine_schedule, param_groups, resolve_device
from signadapt.utils.config import load_config
from signadapt.utils.metrics import accuracy, mean_std, per_group_accuracy, topk_correct
from signadapt.utils.seeding import seed_everything, torch_generator

T, L, C = 64, 115, 4


@pytest.fixture
def cfg():
    return load_config("configs/model.yaml")


@pytest.fixture
def model(cfg):
    seed_everything(0)
    return build_model(cfg)


def batch(n=4, valid_fraction=0.7, seed=0):
    """A batch shaped like KeypointDataset output, with realistic missing landmarks."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, L, C, generator=generator)
    x[..., 3] = (torch.rand(n, T, L, generator=generator) < valid_fraction).float()
    x[..., :3] *= x[..., 3:]
    return x


# ------------------------------------------------------------------------ shapes and config


def test_forward_shape(model):
    assert model(batch()).shape == (4, 64)


def test_embed_shape(model, cfg):
    assert model.embed(batch()).shape == (4, cfg["encoder"]["d_model"])


def test_encoder_accepts_flattened_input(model):
    model.eval()  # dropout is active in train mode, so the two calls would differ
    x = batch()
    torch.testing.assert_close(model.encoder(x), model.encoder(x.flatten(2)))


def test_input_dim_matches_the_dataset_layout(cfg):
    data = load_config("configs/data.yaml")
    assert cfg["encoder"]["input_dim"] == data["landmarks"]["total"] * C


def test_max_len_matches_temporal_resampling(cfg):
    assert cfg["encoder"]["max_len"] == load_config("configs/data.yaml")["temporal"]["n_frames"]


def test_encoder_rejects_a_clip_longer_than_max_len(model):
    with pytest.raises(ValueError, match="max_len"):
        model.encoder(torch.zeros(1, T + 1, L, C))


def test_cls_pooling_also_works(cfg):
    spec = dict(cfg["encoder"], pooling="cls")
    spec.pop("type")
    assert TemporalTransformerEncoder(**spec)(batch()).shape == (4, spec["d_model"])


def test_unknown_pooling_is_rejected(cfg):
    spec = dict(cfg["encoder"], pooling="max")
    spec.pop("type")
    with pytest.raises(ValueError, match="pooling"):
        TemporalTransformerEncoder(**spec)


def test_unknown_types_are_rejected(cfg):
    with pytest.raises(ValueError, match="encoder.type"):
        build_encoder({"encoder": dict(cfg["encoder"], type="lstm")})
    with pytest.raises(ValueError, match="head.type"):
        build_head({"head": dict(cfg["head"], type="prototypical")})


def test_mismatched_encoder_and_head_are_rejected():
    encoder = TemporalTransformerEncoder(input_dim=L * C, d_model=128)
    with pytest.raises(ValueError, match="embeddings"):
        SignAdaptModel(encoder, LinearHead(in_dim=64, n_classes=64))


# ------------------------------------------------------------------- the parameter budget


def test_parameter_count_is_within_the_plan_budget(model):
    """PLAN.md section 4 budgets 0.5-2 M parameters for the shared encoder."""
    counts = model.n_parameters()
    assert 0.5e6 <= counts["encoder"] <= 2e6, counts
    assert counts["total"] == counts["encoder"] + counts["head"]


def test_head_is_a_negligible_fraction_of_the_model(model):
    """The private head is what a client keeps; the shared encoder is what it downloads."""
    counts = model.n_parameters()
    assert counts["head"] < 0.05 * counts["encoder"]


# --------------------------------------------------------------------- parameter groups


def test_every_parameter_belongs_to_exactly_one_group(model):
    """FedPer aggregates by prefix, so the partition must be total and non-overlapping."""
    names = set(model.state_dict())
    encoder, head = set(model.encoder_state_dict()), set(model.head_state_dict())
    assert encoder | head == names
    assert not encoder & head
    assert all(n.startswith(ENCODER_PREFIX) for n in encoder)
    assert all(n.startswith(HEAD_PREFIX) for n in head)


def test_parameter_iterators_match_the_state_dicts(model):
    assert sum(p.numel() for p in model.encoder_parameters()) == sum(
        v.numel() for v in model.encoder_state_dict().values()
    )
    assert sum(p.numel() for p in model.head_parameters()) == sum(
        v.numel() for v in model.head_state_dict().values()
    )


def test_loading_an_encoder_payload_leaves_the_head_alone(cfg):
    seed_everything(0)
    a = build_model(cfg)
    seed_everything(1)
    b = build_model(cfg)
    head_before = {k: v.clone() for k, v in b.head_state_dict().items()}

    b.load_encoder_state_dict(a.encoder_state_dict())

    for key, value in a.encoder_state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key])
    for key, value in head_before.items():
        torch.testing.assert_close(value, b.state_dict()[key])


def test_loading_a_head_tensor_as_an_encoder_payload_raises(model):
    """Silently ignoring a stray key is how a private parameter gets overwritten."""
    payload = model.encoder_state_dict() | model.head_state_dict()
    with pytest.raises(ValueError, match="not encoder parameters"):
        model.load_encoder_state_dict(payload)


def test_state_dict_slices_are_detached_copies(model):
    """A captured payload must not keep changing while training continues."""
    captured = model.encoder_state_dict()
    with torch.no_grad():
        model.encoder.input_proj.weight.add_(1.0)
    assert not torch.allclose(
        captured["encoder.input_proj.weight"], model.encoder.input_proj.weight
    )
    assert not any(v.requires_grad for v in captured.values())


def test_freeze_encoder_leaves_only_the_head_trainable(model):
    model.freeze_encoder(True)
    assert not any(p.requires_grad for p in model.encoder_parameters())
    assert all(p.requires_grad for p in model.head_parameters())
    trainable = sum(
        g["params"] and sum(p.numel() for p in g["params"]) or 0 for g in param_groups(model, 0.01)
    )
    assert trainable == model.n_parameters()["head"]
    model.freeze_encoder(False)
    assert all(p.requires_grad for p in model.encoder_parameters())


def test_group_state_dict_on_a_bare_module(model):
    assert set(group_state_dict(model, HEAD_PREFIX)) == set(model.head_state_dict())


def test_head_reset_changes_its_weights(model):
    before = model.head.linear.weight.clone()
    seed_everything(7)
    model.head.reset_parameters()
    assert not torch.allclose(before, model.head.linear.weight)


# --------------------------------------------------------------------------- augmentation

AUG = {
    "enabled": True,
    "rotate_deg": 10.0,
    "scale_jitter": 0.1,
    "translate": 0.05,
    "time_mask": 8,
    "dropout_landmarks": 0.05,
}


def test_augmentation_never_resurrects_a_missing_landmark():
    """The core invariant: augmentation may destroy a landmark, never create one."""
    x = batch(n=8, valid_fraction=0.6)
    out = augment_batch(x, AUG, torch_generator(0))
    missing = out[..., 3] == 0
    assert (out[..., :3][missing] == 0).all()


def test_augmentation_only_turns_validity_off():
    x = batch(n=8, valid_fraction=0.6)
    out = augment_batch(x, AUG, torch_generator(0))
    assert (out[..., 3] <= x[..., 3]).all()
    assert (out[..., 3] < x[..., 3]).any(), "time_mask/dropout should drop something"


def test_augmentation_preserves_shape_and_dtype():
    x = batch()
    out = augment_batch(x, AUG, torch_generator(0))
    assert out.shape == x.shape and out.dtype == x.dtype


def test_augmentation_is_reproducible_from_the_generator():
    x = batch()
    a = augment_batch(x, AUG, torch_generator(3))
    b = augment_batch(x, AUG, torch_generator(3))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, augment_batch(x, AUG, torch_generator(4)))


def test_augmentation_disabled_is_the_identity():
    x = batch()
    assert augment_batch(x, dict(AUG, enabled=False), torch_generator(0)) is x


def test_rotation_preserves_distance_from_the_anchor():
    """Rotation is about the mid-shoulder origin, so radii in the xy plane are invariant."""
    x = batch(valid_fraction=1.0)
    only_rotate = {"enabled": True, "rotate_deg": 10.0}
    out = augment_batch(x, only_rotate, torch_generator(0))
    torch.testing.assert_close(
        out[..., :2].norm(dim=-1), x[..., :2].norm(dim=-1), rtol=1e-4, atol=1e-4
    )


def test_rotation_leaves_z_untouched():
    x = batch(valid_fraction=1.0)
    out = augment_batch(x, {"enabled": True, "rotate_deg": 10.0}, torch_generator(0))
    torch.testing.assert_close(out[..., 2], x[..., 2])


def test_time_mask_drops_a_contiguous_span_of_frames():
    x = batch(n=1, valid_fraction=1.0)
    out = augment_batch(x, {"enabled": True, "time_mask": 8}, torch_generator(0))
    dropped = (out[0, :, :, 3] == 0).all(dim=1).nonzero().flatten().tolist()
    assert dropped, "a span should have been masked"
    assert dropped == list(range(dropped[0], dropped[-1] + 1)), "the span must be contiguous"
    assert len(dropped) <= 8


def test_landmark_dropout_removes_a_landmark_for_the_whole_clip():
    x = batch(n=4, valid_fraction=1.0)
    out = augment_batch(x, {"enabled": True, "dropout_landmarks": 0.5}, torch_generator(0))
    per_landmark = (out[..., 3] == 0).float().mean(dim=1)
    assert set(per_landmark.unique().tolist()) <= {0.0, 1.0}, "dropout is per clip, not per frame"


# ------------------------------------------------------------------------------- metrics


def test_topk_matches_a_hand_computed_case():
    logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 1.0]])
    targets = torch.tensor([2, 1])
    assert topk_correct(logits, targets, 1).tolist() == [False, True]
    assert topk_correct(logits, targets, 2).tolist() == [True, True]
    assert accuracy(logits, targets, 1) == 0.5


def test_topk_clips_k_to_the_number_of_classes():
    logits, targets = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
    assert accuracy(logits, targets, 5) == 1.0


def test_accuracy_of_an_empty_batch_is_nan_not_zero():
    """A zero would silently drag down a mean over signers."""
    import math

    assert math.isnan(accuracy(torch.empty(0, 3), torch.empty(0, dtype=torch.long)))


def test_per_group_accuracy_splits_correctly():
    correct = torch.tensor([True, False, True, True])
    assert per_group_accuracy(correct, [9, 9, 10, 10]) == {"9": 0.5, "10": 1.0}


def test_per_group_accuracy_rejects_misaligned_keys():
    with pytest.raises(ValueError, match="group keys"):
        per_group_accuracy(torch.tensor([True, False]), [1])


def test_mean_std_is_the_sample_standard_deviation():
    summary = mean_std([0.2, 0.4, 0.6])
    assert summary["mean"] == pytest.approx(0.4)
    assert summary["std"] == pytest.approx(0.2)
    assert summary["n"] == 3


def test_mean_std_of_one_value_has_nan_std_not_zero():
    import math

    assert math.isnan(mean_std([0.5])["std"])


# ------------------------------------------------------------------------ training pieces


def test_cosine_schedule_warms_up_then_decays_to_zero():
    factor = cosine_schedule(epochs=10, warmup_epochs=3)
    values = [factor(e) for e in range(10)]
    assert values[0] < values[1] < values[2] <= values[3], "warmup should ramp up"
    assert values[3:] == sorted(values[3:], reverse=True), "then decay monotonically"
    assert values[-1] < 0.1


def test_cosine_schedule_handles_zero_warmup():
    assert cosine_schedule(epochs=5, warmup_epochs=0)(0) == pytest.approx(1.0)


def test_weight_decay_excludes_norms_biases_and_positions(model):
    decay, no_decay = param_groups(model, 0.01)
    assert decay["weight_decay"] == 0.01
    assert no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert any(p.ndim == 3 for p in no_decay["params"]), "pos_embedding must not be decayed"


def test_resolve_device_honours_an_explicit_name():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
