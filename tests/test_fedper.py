"""Asserts that head parameters are never included in federated aggregation (PLAN.md E5).

Written in phase 4 alongside ``federated/strategy.py``. Skips loudly until then.
"""

import pytest

pytest.skip(
    "phase 4: implemented together with signadapt.federated.strategy",
    allow_module_level=True,
)
