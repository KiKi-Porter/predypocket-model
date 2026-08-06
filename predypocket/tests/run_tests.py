"""Dependency-free runner for the isolated function-style test suite."""

from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
import traceback
from types import ModuleType


class _Raises:
    def __init__(self, exception_type, match: str | None = None):
        self.exception_type = exception_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback_object):
        del traceback_object
        if exception is None:
            raise AssertionError(f"Expected {self.exception_type.__name__} was not raised")
        if not isinstance(exception, self.exception_type):
            return False
        if self.match is not None and re.search(self.match, str(exception)) is None:
            raise AssertionError(
                f"Exception {exception!r} does not match pattern {self.match!r}"
            )
        return True


def _install_pytest_compatibility() -> None:
    if "pytest" in sys.modules:
        return
    module = ModuleType("pytest")

    def fixture(*, scope: str | None = None):
        del scope

        def decorator(function):
            return function

        return decorator

    module.fixture = fixture
    module.raises = lambda exception_type, match=None: _Raises(exception_type, match)
    sys.modules["pytest"] = module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tests",
        help="Comma-separated numeric test IDs; omitted means the complete suite",
    )
    args = parser.parse_args(argv)
    requested = (
        {int(value) for value in args.tests.split(",")}
        if args.tests
        else None
    )
    _install_pytest_compatibility()
    conftest = importlib.import_module("predypocket.tests.conftest")
    fixtures = {}

    def fixture(name):
        if name not in fixtures:
            if name == "loaded_model":
                fixtures[name] = conftest.loaded_model()
            elif name == "model_artifacts":
                fixtures[name] = conftest.model_artifacts(fixture("loaded_model"))
            elif name == "backward_artifacts":
                fixtures[name] = conftest.backward_artifacts(fixture("loaded_model"))
            elif name == "protocol_v2_models":
                fixtures[name] = conftest.protocol_v2_models(fixture("loaded_model"))
            elif name == "protocol_v2_split_artifacts":
                fixtures[name] = conftest.protocol_v2_split_artifacts()
            else:
                raise KeyError(name)
        return fixtures[name]
    module_names = (
        "predypocket.tests.test_model",
        "predypocket.tests.test_data_loss",
        "predypocket.tests.test_metrics_folds",
        "predypocket.tests.test_backward_safety",
        "predypocket.tests.test_readiness",
        "predypocket.tests.test_protocol_v2",
    )
    tests = []
    for module_name in module_names:
        module = importlib.import_module(module_name)
        tests.extend(
            (module_name, name, function)
            for name, function in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_")
        )
    tests.sort(key=lambda item: int(item[1].split("_", 2)[1]))
    if requested is not None:
        tests = [
            item for item in tests if int(item[1].split("_", 2)[1]) in requested
        ]
    failures = []
    for module_name, name, function in tests:
        arguments = {
            parameter: fixture(parameter)
            for parameter in inspect.signature(function).parameters
        }
        try:
            function(**arguments)
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - test runner must aggregate failures
            failures.append((module_name, name, exc, traceback.format_exc()))
            print(f"FAIL {name}: {exc}")
    print(
        f"SUMMARY passed={len(tests) - len(failures)} failed={len(failures)} "
        f"total={len(tests)}"
    )
    for module_name, name, _, details in failures:
        print(f"\n--- {module_name}.{name} ---\n{details}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
