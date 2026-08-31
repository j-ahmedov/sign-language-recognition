"""Shared encoder, private head, and their assembly."""

from signadapt.models.encoder import TemporalTransformerEncoder, build_encoder
from signadapt.models.head import LinearHead, build_head
from signadapt.models.model import (
    ENCODER_PREFIX,
    HEAD_PREFIX,
    SignAdaptModel,
    build_model,
    group_state_dict,
)

__all__ = [
    "ENCODER_PREFIX",
    "HEAD_PREFIX",
    "LinearHead",
    "SignAdaptModel",
    "TemporalTransformerEncoder",
    "build_encoder",
    "build_head",
    "build_model",
    "group_state_dict",
]
