"""Load a lab implementation by file path.

Lab directories start with digits (``01_autograd``), so they are not importable
as normal packages. Tests call ``load(__file__)`` to import the sibling
``exercise.py`` (default) or ``solution.py`` (``--impl=solution`` or
``LABS_IMPL=solution``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load(test_file: str, impl: str | None = None) -> ModuleType:
    lab_dir = Path(test_file).resolve().parent
    target = impl or os.environ.get("LABS_IMPL", "exercise")
    path = lab_dir / f"{target}.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate stubs with `python tools/make_exercises.py`."
        )
    name = f"_labs_{lab_dir.name}_{target}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
