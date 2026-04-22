"""
Shared pytest configuration for the AgentXAI test suite.

Adds a --run-slow flag; tests marked @pytest.mark.slow are skipped unless
that flag is passed.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow integration tests (requires model downloads).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow/integration (skipped unless --run-slow is passed)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="Pass --run-slow to run slow integration tests.")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
