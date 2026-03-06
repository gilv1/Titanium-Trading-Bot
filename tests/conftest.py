"""
Shared pytest fixtures and configuration for the Titanium Warrior test suite.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolated_risk_state(tmp_path):
    """
    Redirect the risk-manager state file to a temporary directory for every test
    so that tests neither read stale state from disk nor write to the real
    ``data/risk_state.json``.

    ``tmp_path`` is a built-in pytest fixture that provides a unique temporary
    directory for each test invocation.
    """
    import core.risk_manager as rm_module

    tmp_state = os.path.join(str(tmp_path), "risk_state.json")
    with patch.object(rm_module, "_RISK_STATE_PATH", tmp_state):
        yield
