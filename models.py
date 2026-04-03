# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hypertrophy Env Environment.

The hypertrophy_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class HypertrophyAction(Action):
    """Action for the Hypertrophy Env environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class HypertrophyObservation(Observation):
    """Observation from the Hypertrophy Env environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
