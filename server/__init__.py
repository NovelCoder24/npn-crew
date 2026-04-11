# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hypertrophy Env environment server components."""

__all__ = ["HypertrophyEnvironment"]


def __getattr__(name):
    """Lazy import to avoid circular/failing imports during validator checks."""
    if name == "HypertrophyEnvironment":
        from server.hypertrophy_env_environment import HypertrophyEnvironment
        return HypertrophyEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
