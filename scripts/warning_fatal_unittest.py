#!/usr/bin/env python3
"""Run unittest with spawn-safe discovery and the Python 3.13 SWIG cleanup."""

from __future__ import annotations

import argparse
import atexit
import sys
import unittest
import warnings


def _cleanup_sentencepiece_swig_capsule() -> None:
    # Python 3.13 can otherwise destroy SentencePiece's shared SWIG capsule
    # after module teardown and segfault after a completely successful suite.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = sys.modules.get("swig_runtime_data4")
        if module is not None and hasattr(module, "type_pointer_capsule"):
            delattr(module, "type_pointer_capsule")


atexit.register(_cleanup_sentencepiece_swig_capsule)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="*")
    parser.add_argument("--start-directory", default="tests")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loader = unittest.defaultTestLoader
    suite = (
        loader.loadTestsFromNames(args.tests)
        if args.tests
        else loader.discover(args.start_directory, pattern=args.pattern)
    )
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
