"""
Inference Script — Hypertrophy Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=muscle_gain env=hypertrophy_env model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action={"intensity":7,"volume":6,"recovery_strategy":8} reward=0.84 done=false error=null
    [STEP] step=84 action={"intensity":5,"volume":5,"recovery_strategy":9} reward=0.72 done=true error=null
    [END] success=true steps=84 score=0.650 rewards=0.84,...,0.72
"""

import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import textwrap
from typing import Callable, Dict, List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from client import HypertrophyEnv
    from models import HypertrophyAction
except ModuleNotFoundError:
    from hypertrophy_env import HypertrophyAction, HypertrophyEnv

if load_dotenv is not None:
    load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv(
    "ENV_HTTP_URL", "https://novelcoder123-hypertrophy-env-openenv.hf.space"
)
API_KEY = os.getenv("HF_TOKEN")
ALLOW_TEMP_CONTAINERS = os.getenv("ALLOW_TEMP_CONTAINERS", "0") == "1"

API_BASE_URL = os.getenv("API_BASE_URL" , "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME" , "Qwen/Qwen2.5-72B-Instruct")

# Task selection (hackathon requires 3+ tasks with graders)
TASK_NAME = os.getenv("HYPERTROPHY_TASK", "muscle_gain")
BENCHMARK = "hypertrophy_env"

MAX_STEPS = 84
# Reproducibility controls:
# - REPRODUCIBLE_MODE=1 forces deterministic decoding settings.
# - INFERENCE_SEED controls local randomness (e.g. any fallback policy behavior).
REPRODUCIBLE_MODE = os.getenv("REPRODUCIBLE_MODE", "1") == "1"
INFERENCE_SEED = int(os.getenv("INFERENCE_SEED", "42"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = 100  # Keep small for 20-min runtime budget
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]
DEBUG_LOGS = os.getenv("DEBUG_LOGS", "0") == "1"
SAVE_TRAJECTORY = os.getenv("SAVE_TRAJECTORY", "1") == "1"
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

if REPRODUCIBLE_MODE:
    random.seed(INFERENCE_SEED)
    TEMPERATURE = 0.0


# ──────────────────────────────────────────────
# Task-specific graders (all return score ∈ [0.0, 1.0])
# ──────────────────────────────────────────────
def grade_muscle_gain(obs, rewards: List[float], metadata: Dict) -> float:
    """Primary task: how much muscle did we gain over 84 days?"""
    return max(0.0, min(1.0, (obs.muscle_size - 50.0) / 50.0))


def grade_fatigue_management(obs, rewards: List[float], metadata: Dict) -> float:
    """Balance task: grow muscle while keeping average fatigue low."""
    muscle_score = (obs.muscle_size - 50.0) / 50.0
    avg_fatigue = metadata.get("avg_fatigue", 0.5)
    return max(0.0, min(1.0, (1.0 - avg_fatigue) * muscle_score))


def grade_periodization(obs, rewards: List[float], metadata: Dict) -> float:
    """Efficiency task: achieve gains with minimal overtraining days."""
    muscle_score = (obs.muscle_size - 50.0) / 50.0
    overtrain_ratio = metadata.get("overtrain_days", 0) / MAX_STEPS
    no_overtrain = 1.0 - overtrain_ratio
    return max(0.0, min(1.0, muscle_score * 0.6 + no_overtrain * 0.4))


TASK_SPECS: Dict[str, Dict[str, object]] = {
    "muscle_gain": {
        "grader": grade_muscle_gain,
        "difficulty": "easy",
        "description": "Maximize final muscle size over 84 days.",
    },
    "fatigue_management": {
        "grader": grade_fatigue_management,
        "difficulty": "medium",
        "description": "Grow muscle while controlling cumulative fatigue.",
    },
    "periodization": {
        "grader": grade_periodization,
        "difficulty": "hard",
        "description": "Maximize gains while minimizing overtraining days.",
    },
}

if TASK_NAME not in TASK_SPECS:
    valid = ", ".join(sorted(TASK_SPECS.keys()))
    raise ValueError(f"Unknown HYPERTROPHY_TASK='{TASK_NAME}'. Expected one of: {valid}")


def get_task_grader(task_name: str) -> Callable:
    return TASK_SPECS[task_name]["grader"]  # type: ignore[return-value]


def get_task_difficulty(task_name: str) -> str:
    return str(TASK_SPECS[task_name]["difficulty"])


# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI strength coach managing a 12-week (84-day) hypertrophy program.

    Each day you choose three values (integers 1-10):
    - intensity: How hard to train (1=light warm-up, 10=maximum effort)
    - volume: How much to train (1=few sets, 10=maximum volume)
    - recovery_strategy: Recovery effort (1=poor sleep/nutrition, 10=optimal recovery)

    You observe: muscle_size (50-100), strength (50-100), fatigue (0.0-1.0).

    KEY RULES:
    - Fatigue above 0.8 triggers overtraining penalties (quadratic)
    - High fatigue reduces training effectiveness: gains *= (1 - fatigue²)
    - Recovery reduces fatigue; you must balance training vs rest
    - Smart recovery (recovery ≥ 7 when fatigue < 0.3) earns bonus reward
    - Goal: maximize muscle_size by day 84 while avoiding overtraining

    Reply ONLY with one-line JSON in this exact structure:
    {"thought":"short reason","action":{"intensity":N,"volume":N,"recovery_strategy":N}}

    The thought must reference current fatigue and tradeoff between growth and recovery.
    Keep thought under 20 words.
    """
).strip()


# ──────────────────────────────────────────────
# Logging (strict hackathon format)
# ──────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────
def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-10:]) if history else "None"
    week = ((obs.day - 1) // 7 + 1) if obs.day > 0 else 1
    return textwrap.dedent(
        f"""
        Day: {obs.day}/84 (Week {week})
        Muscle Size: {obs.muscle_size:.1f}/100
        Strength: {obs.strength:.1f}/100
        Fatigue: {obs.fatigue:.2f}/1.0
        Last Reward: {last_reward:.2f}
        Status: {obs.status_message}
        Recent History:
        {history_block}

        Choose your training parameters for today.
        If fatigue is high, prioritize recovery to avoid penalties.
        """
    ).strip()


# ──────────────────────────────────────────────
# Action parsing (robust against malformed LLM output)
# ──────────────────────────────────────────────
def _clamp_action_from_payload(data: Dict) -> HypertrophyAction:
    return HypertrophyAction(
        intensity=max(1, min(10, int(data.get("intensity", 5)))),
        volume=max(1, min(10, int(data.get("volume", 5)))),
        recovery_strategy=max(1, min(10, int(data.get("recovery_strategy", 5)))),
    )


def parse_action_response(text: str) -> tuple[HypertrophyAction, str]:
    """Parse model output into (action, thought) with robust fallback."""
    try:
        cleaned = text.strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # LLMs may wrap JSON in markdown fences or add extra prose.
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
            if fence_match:
                data = json.loads(fence_match.group(1))
            else:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise
                data = json.loads(cleaned[start : end + 1])

        if not isinstance(data, dict):
            raise ValueError("Model output JSON must be an object")

        if isinstance(data.get("action"), dict):
            action = _clamp_action_from_payload(data["action"])
            thought = str(data.get("thought", "")).strip()
            return action, thought

        # Backward compatibility: allow flat action JSON without thought
        action = _clamp_action_from_payload(data)
        return action, ""

    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        # Fallback: moderate balanced training
        return HypertrophyAction(intensity=5, volume=5, recovery_strategy=5), ""


def action_to_str(action: HypertrophyAction) -> str:
    """Serialize action to compact JSON string for logging."""
    return json.dumps({
        "intensity": action.intensity,
        "volume": action.volume,
        "recovery_strategy": action.recovery_strategy,
    }, separators=(",", ":"))


# ──────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────
def _get_model_action_sync(
    client: OpenAI, step: int, obs, last_reward: float, history: List[str]
) -> tuple[HypertrophyAction, str]:
    """Ask the LLM for next training decision."""
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return HypertrophyAction(intensity=5, volume=5, recovery_strategy=5), ""
        return parse_action_response(text)
    except Exception as exc:
        if DEBUG_LOGS:
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return HypertrophyAction(intensity=5, volume=5, recovery_strategy=5), ""


async def get_model_action(
    client: OpenAI, step: int, obs, last_reward: float, history: List[str]
) -> tuple[HypertrophyAction, str]:
    """Run blocking model call in a worker thread to avoid blocking async websocket keepalives."""
    return await asyncio.to_thread(_get_model_action_sync, client, step, obs, last_reward, history)


# ──────────────────────────────────────────────
# Main episode loop
# ──────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if ENV_BASE_URL:
        # Reuse a running server and avoid creating a temporary Docker container.
        env = HypertrophyEnv(base_url=ENV_BASE_URL)
    elif ALLOW_TEMP_CONTAINERS:
        if not IMAGE_NAME:
            raise ValueError("IMAGE_NAME is required when ALLOW_TEMP_CONTAINERS=1")
        env = await HypertrophyEnv.from_docker_image(IMAGE_NAME)
    else:
        raise ValueError(
            "Temporary container startup is disabled. Set ENV_BASE_URL (or ENV_HTTP_URL) to an "
            "existing environment server, or set ALLOW_TEMP_CONTAINERS=1 to enable Docker fallback."
        )

    history: List[str] = []
    trajectory: List[Dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_obs = None
    last_metadata: Dict = {}

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    if DEBUG_LOGS:
        print(
            f"[DEBUG] task={TASK_NAME} difficulty={get_task_difficulty(TASK_NAME)} "
            f"reproducible_mode={str(REPRODUCIBLE_MODE).lower()} seed={INFERENCE_SEED} "
            f"temperature={TEMPERATURE}",
            flush=True,
        )

    try:
        result = await env.reset()
        last_obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, thought = await get_model_action(client, step, last_obs, last_reward, history)
            if DEBUG_LOGS:
                print(f"[DEBUG] LLM chose: {action} | thought={thought}", flush=True)
            action_str = action_to_str(action)

            try:
                result = await env.step(action)
            except Exception as exc:
                # Transport/socket failure should end the episode cleanly with a visible step error.
                steps_taken = step
                log_step(step=step, action=action_str, reward=0.0, done=True, error=str(exc))
                if DEBUG_LOGS:
                    print(f"[DEBUG] env.step() failed: {exc}", flush=True)
                break
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_obs = obs
            last_reward = reward
            last_metadata = obs.metadata or {}

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Day {obs.day}: i={action.intensity} v={action.volume} r={action.recovery_strategy} "
                f"-> muscle={obs.muscle_size:.1f} fatigue={obs.fatigue:.2f} reward={reward:+.2f}"
            )
            trajectory.append(
                {
                    "day": obs.day,
                    "action": {
                        "intensity": action.intensity,
                        "volume": action.volume,
                        "recovery_strategy": action.recovery_strategy,
                    },
                    "thought": thought,
                    "reward": round(float(reward), 4),
                    "muscle_size": round(float(obs.muscle_size), 4),
                    "strength": round(float(obs.strength), 4),
                    "fatigue": round(float(obs.fatigue), 4),
                    "status": obs.status_message,
                }
            )

            if done:
                break

        # Score via task-specific grader (NOT sum of rewards)
        if last_obs is not None:
            grader = get_task_grader(TASK_NAME)
            score = grader(last_obs, rewards, last_metadata)
            score = max(0.0, min(1.0, score))  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        if SAVE_TRAJECTORY and trajectory:
            artifact_path = Path(ARTIFACT_DIR)
            artifact_path.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = artifact_path / f"trajectory_{TASK_NAME}_{stamp}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "task": TASK_NAME,
                        "task_difficulty": get_task_difficulty(TASK_NAME),
                        "model": MODEL_NAME,
                        "score": round(float(score), 4),
                        "steps": steps_taken,
                        "reproducible_mode": REPRODUCIBLE_MODE,
                        "inference_seed": INFERENCE_SEED,
                        "temperature": TEMPERATURE,
                        "trajectory": trajectory,
                    },
                    f,
                    indent=2,
                )
        try:
            await env.close()
        except Exception as e:
            if DEBUG_LOGS:
                print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())