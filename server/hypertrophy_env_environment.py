# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hypertrophy Environment Implementation.

A 12-week (84-day) hypertrophy simulator with fatigue management.
The agent must balance training intensity and volume against recovery
to maximize muscle growth while avoiding overtraining.

MDP Formalization:
    State:  (day, muscle_size, strength, fatigue)
    Action: (intensity, volume, recovery_strategy) ∈ [1,10]³
    Transition:
        effective_stimulus = intensity × volume × (1 - fatigue²)
        muscle_size' = clamp(muscle + stimulus × 0.02, 50, 100)
        strength'    = clamp(strength + stimulus × 0.012, 50, 100)
        fatigue'     = clamp(fatigue + i×v×0.008 - recovery×0.06, 0, 1)
    Reward:
        base         = (muscle_delta) × 10.0
        penalty      = -20.0 × max(0, fatigue' - 0.8)²
        bonus        = +1.0 if recovery ≥ 7 and fatigue' < 0.3
    Terminal: day ≥ 84
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import HypertrophyAction, HypertrophyObservation

MAX_DAYS = 84


class HypertrophyEnvironment(Environment):
    """
    12-week hypertrophy simulator with fatigue management.

    The agent trains daily for 84 days (12 weeks). Each day it chooses
    training intensity, volume, and recovery effort. The environment
    simulates muscle growth, strength gains, and fatigue accumulation.

    Overtraining (fatigue > 0.8) is penalized quadratically.
    Smart recovery (high recovery + low fatigue) earns bonus reward.

    Example:
        >>> env = HypertrophyEnvironment()
        >>> obs = env.reset()
        >>> print(obs.muscle_size)  # 50.0
        >>> obs = env.step(HypertrophyAction(intensity=7, volume=6, recovery_strategy=8))
        >>> print(obs.muscle_size)  # ~50.84
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the hypertrophy environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._day = 0
        self._muscle_size = 50.0
        self._strength = 50.0
        self._fatigue = 0.0
        # Episode-level tracking for graders
        self._total_fatigue = 0.0
        self._overtrain_days = 0

    def reset(self) -> HypertrophyObservation:
        """
        Reset the environment to day 0 with baseline stats.

        Returns:
            HypertrophyObservation with initial state
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._day = 0
        self._muscle_size = 50.0
        self._strength = 50.0
        self._fatigue = 0.0
        self._total_fatigue = 0.0
        self._overtrain_days = 0

        return HypertrophyObservation(
            day=0,
            muscle_size=50.0,
            strength=50.0,
            fatigue=0.0,
            status_message="Week 1 begins. You are fresh and ready to train.",
            done=False,
            reward=0.0,
        )

    def step(self, action: HypertrophyAction) -> HypertrophyObservation:  # type: ignore[override]
        """
        Execute one training day.

        Args:
            action: HypertrophyAction with intensity, volume, recovery_strategy

        Returns:
            HypertrophyObservation with updated state, reward, and done flag
        """
        self._state.step_count += 1
        self._day += 1

        i = action.intensity
        v = action.volume
        r = action.recovery_strategy

        # --- Transition dynamics ---
        # Fatigue reduces training effectiveness quadratically
        fatigue_penalty_factor = 1.0 - (self._fatigue ** 2)
        effective_stimulus = i * v * fatigue_penalty_factor

        old_muscle = self._muscle_size

        # Muscle and strength growth (clamped to [50, 100])
        self._muscle_size = min(self._muscle_size + effective_stimulus * 0.02, 100.0)
        self._strength = min(self._strength + effective_stimulus * 0.012, 100.0)

        # Fatigue accumulation and recovery (clamped to [0, 1])
        fatigue_gain = i * v * 0.008
        fatigue_recovery = r * 0.06
        self._fatigue = max(0.0, min(self._fatigue + fatigue_gain - fatigue_recovery, 1.0))

        # --- Episode tracking for graders ---
        self._total_fatigue += self._fatigue
        if self._fatigue > 0.8:
            self._overtrain_days += 1

        # --- Reward computation ---
        muscle_delta = self._muscle_size - old_muscle
        base_reward = muscle_delta * 10.0

        # Quadratic overtraining penalty (kicks in above 0.8)
        fatigue_penalty = 0.0
        if self._fatigue > 0.8:
            fatigue_penalty = -20.0 * ((self._fatigue - 0.8) ** 2)

        # Recovery bonus for smart rest
        recovery_bonus = 1.0 if (r >= 7 and self._fatigue < 0.3) else 0.0

        reward = base_reward + fatigue_penalty + recovery_bonus

        # --- Terminal check ---
        done = self._day >= MAX_DAYS

        # --- Status message ---
        week = (self._day - 1) // 7 + 1
        if self._fatigue > 0.8:
            status = (
                f"Day {self._day} (Week {week}): OVERTRAINING! "
                f"Fatigue critical at {self._fatigue:.2f}"
            )
        elif self._fatigue > 0.5:
            status = (
                f"Day {self._day} (Week {week}): High fatigue ({self._fatigue:.2f}). "
                f"Consider recovery."
            )
        elif done:
            status = (
                f"Program complete! Final muscle: {self._muscle_size:.1f}, "
                f"strength: {self._strength:.1f}"
            )
        else:
            status = (
                f"Day {self._day} (Week {week}): "
                f"Muscle {self._muscle_size:.1f} | "
                f"Strength {self._strength:.1f} | "
                f"Fatigue {self._fatigue:.2f}"
            )

        return HypertrophyObservation(
            day=self._day,
            muscle_size=round(self._muscle_size, 2),
            strength=round(self._strength, 2),
            fatigue=round(self._fatigue, 4),
            status_message=status,
            done=done,
            reward=round(reward, 4),
            metadata={
                "week": week,
                "effective_stimulus": round(effective_stimulus, 2),
                "muscle_delta": round(muscle_delta, 4),
                "fatigue_penalty": round(fatigue_penalty, 4),
                "avg_fatigue": round(self._total_fatigue / self._day, 4) if self._day > 0 else 0.0,
                "overtrain_days": self._overtrain_days,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
