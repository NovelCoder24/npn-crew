# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hypertrophy Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import HypertrophyAction, HypertrophyObservation
except ImportError:
    from models import HypertrophyAction, HypertrophyObservation


class HypertrophyEnv(
    EnvClient[HypertrophyAction, HypertrophyObservation, State]
):
    """
    Client for the Hypertrophy Training Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with HypertrophyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.muscle_size)  # 50.0
        ...
        ...     result = client.step(HypertrophyAction(intensity=7, volume=6, recovery_strategy=8))
        ...     print(result.observation.muscle_size)  # ~50.84

    Example with Docker:
        >>> client = HypertrophyEnv.from_docker_image("hypertrophy_env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HypertrophyAction(intensity=5, volume=5, recovery_strategy=7))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: HypertrophyAction) -> Dict:
        """
        Convert HypertrophyAction to JSON payload for step message.

        Args:
            action: HypertrophyAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "intensity": action.intensity,
            "volume": action.volume,
            "recovery_strategy": action.recovery_strategy,
        }

    def _parse_result(self, payload: Dict) -> StepResult[HypertrophyObservation]:
        """
        Parse server response into StepResult[HypertrophyObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with HypertrophyObservation
        """
        obs_data = payload.get("observation", {})
        observation = HypertrophyObservation(
            day=obs_data.get("day", 0),
            muscle_size=obs_data.get("muscle_size", 50.0),
            strength=obs_data.get("strength", 50.0),
            fatigue=obs_data.get("fatigue", 0.0),
            status_message=obs_data.get("status_message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
