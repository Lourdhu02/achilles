"""Pytest plumbing for the labs.

Every lab directory holds:
    solution.py   reference implementation -- read it only after an honest attempt
    exercise.py   the same API with the bodies removed -- this is where you work
    test_*.py     tests that import whichever implementation you select

Choose the implementation with --impl (default: exercise):
    pytest labs/01_autograd                    # grades YOUR code
    pytest labs/01_autograd --impl=solution    # checks the reference
"""

import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--impl",
        action="store",
        default=None,
        help="Which implementation the lab tests import: exercise (default) or solution.",
    )

def pytest_configure(config):
    impl = config.getoption("--impl")
    if impl:
        os.environ["LABS_IMPL"] = impl

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and os.environ.get("ACHILLES_STRICT_TIME") == "1":
        if report.duration > 10.0 and not item.get_closest_marker("slow"):
            report.outcome = "failed"
            report.longrepr = f"Test exceeded 10s budget ({report.duration:.2f}s) but lacks @pytest.mark.slow."
