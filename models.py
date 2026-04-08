# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hypertrophy Environment.

A 12-week (84-day) hypertrophy simulator where an AI agent learns to
maximize muscle growth by balancing training intensity/volume with
recovery while managing fatigue.
""" 

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class HypertrophyAction(Action):
    """Agent's daily training decision."""

    intensity: int = Field(
        ..., ge=1, le=10,
        description="Training intensity (1=light warm-up, 10=max effort)"
    )
    volume: int = Field(
        ..., ge=1, le=10,
        description="Training volume (1=minimal sets/reps, 10=maximum volume)"
    )
    recovery_strategy: int = Field(
        ..., ge=1, le=10,
        description="Recovery effort (1=poor sleep/nutrition, 10=optimal recovery protocol)"
    )


class HypertrophyObservation(Observation):
    """Observable state after each training day.

    NOTE: `done` and `reward` are inherited from the Observation base class.
    Do NOT redeclare them here or serialization will break.
    """

    day: int = Field(default=0, description="Current training day (0-84)")
    muscle_size: float = Field(default=50.0, description="Muscle size score (50.0-100.0)")
    strength: float = Field(default=50.0, description="Strength score (50.0-100.0)")
    fatigue: float = Field(default=0.0, description="Fatigue level (0.0=fresh, 1.0=overtrained)")
    status_message: str = Field(default="", description="Human-readable status for the agent")
