"""
Evaluate agent behavior in the hypertrophy RL-style environment.

This script compares multiple policies across episodes so you can quantify
whether your agent behavior is improving relative to baselines.

Policies:
- random: uniformly random action each step
- fixed_5_5_5: constant action (5,5,5)
- heuristic: trains harder when fatigue is low, recovers when high
- llm: LLM-driven action policy (optional, enable with ENABLE_LLM_EVAL=1)

Environment connection:
- If ENV_BASE_URL is set, connects to an already running server.
- Otherwise, starts a temporary container from IMAGE_NAME.
"""

import asyncio
import csv
import json
import os
from pathlib import Path
import random
import re
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from hypertrophy_env import HypertrophyAction, HypertrophyEnv
except ModuleNotFoundError:
    from client import HypertrophyEnv
    from models import HypertrophyAction

if load_dotenv is not None:
    load_dotenv()

IMAGE_NAME = os.getenv("IMAGE_NAME", "hypertrophy_env:latest")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("ENV_HTTP_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ALLOW_TEMP_CONTAINERS = os.getenv("ALLOW_TEMP_CONTAINERS", "0") == "1"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("HYPERTROPHY_TASK", "muscle_gain")
EVAL_EPISODES = int(os.getenv("EVAL_EPISODES", "10"))
EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))
MAX_STEPS = 84
ENABLE_LLM_EVAL = os.getenv("ENABLE_LLM_EVAL", "0") == "1"
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

TASK_DIFFICULTY = {
    "muscle_gain": "easy",
    "fatigue_management": "medium",
    "periodization": "hard",
}

if TASK_NAME not in TASK_DIFFICULTY:
    valid = ", ".join(sorted(TASK_DIFFICULTY.keys()))
    raise ValueError(f"Unknown HYPERTROPHY_TASK='{TASK_NAME}'. Expected one of: {valid}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass
class EpisodeResult:
    policy: str
    task: str
    task_difficulty: str
    steps: int
    total_reward: float
    final_muscle: float
    final_strength: float
    avg_fatigue: float
    overtrain_days: int
    score: float
    llm_failures: int
    trajectory: List[Dict]


def score_muscle_gain(final_muscle: float) -> float:
    return max(0.0, min(1.0, (final_muscle - 50.0) / 50.0))


def score_fatigue_management(final_muscle: float, avg_fatigue: float) -> float:
    muscle_score = score_muscle_gain(final_muscle)
    return max(0.0, min(1.0, (1.0 - avg_fatigue) * muscle_score))


def score_periodization(final_muscle: float, overtrain_days: int, steps: int) -> float:
    muscle_score = score_muscle_gain(final_muscle)
    overtrain_ratio = (overtrain_days / steps) if steps else 1.0
    no_overtrain = 1.0 - overtrain_ratio
    return max(0.0, min(1.0, muscle_score * 0.6 + no_overtrain * 0.4))


def choose_task_score(task: str, final_muscle: float, avg_fatigue: float, overtrain_days: int, steps: int) -> float:
    if task == "fatigue_management":
        return score_fatigue_management(final_muscle, avg_fatigue)
    if task == "periodization":
        return score_periodization(final_muscle, overtrain_days, steps)
    return score_muscle_gain(final_muscle)


def random_policy(_obs) -> HypertrophyAction:
    return HypertrophyAction(
        intensity=random.randint(1, 10),
        volume=random.randint(1, 10),
        recovery_strategy=random.randint(1, 10),
    )


def fixed_policy(_obs) -> HypertrophyAction:
    return HypertrophyAction(intensity=5, volume=5, recovery_strategy=5)


def heuristic_policy(obs) -> HypertrophyAction:
    f = float(getattr(obs, "fatigue", 0.0))
    if f >= 0.75:
        return HypertrophyAction(intensity=2, volume=2, recovery_strategy=10)
    if f >= 0.5:
        return HypertrophyAction(intensity=4, volume=4, recovery_strategy=9)
    if f >= 0.3:
        return HypertrophyAction(intensity=6, volume=5, recovery_strategy=8)
    return HypertrophyAction(intensity=8, volume=7, recovery_strategy=8)


def _parse_llm_action(text: str) -> HypertrophyAction:
    try:
        match = re.search(r"\{[^}]+\}", text)
        payload = json.loads(match.group() if match else text)
        return HypertrophyAction(
            intensity=max(1, min(10, int(payload.get("intensity", 5)))),
            volume=max(1, min(10, int(payload.get("volume", 5)))),
            recovery_strategy=max(1, min(10, int(payload.get("recovery_strategy", 5)))),
        )
    except Exception:
        return HypertrophyAction(intensity=5, volume=5, recovery_strategy=5)


async def llm_policy(obs, client: OpenAI) -> HypertrophyAction:
    prompt = (
        "Choose training action JSON for hypertrophy environment. "
        "Return one line JSON with keys intensity, volume, recovery_strategy (1..10). "
        f"Current: day={getattr(obs, 'day', 0)}, muscle={getattr(obs, 'muscle_size', 50.0):.2f}, "
        f"strength={getattr(obs, 'strength', 50.0):.2f}, fatigue={getattr(obs, 'fatigue', 0.0):.3f}."
    )

    def _sync_call() -> HypertrophyAction:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=80,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_llm_action(text)

    return await asyncio.to_thread(_sync_call)


async def create_env() -> HypertrophyEnv:
    if ENV_BASE_URL:
        return HypertrophyEnv(base_url=ENV_BASE_URL)
    if ALLOW_TEMP_CONTAINERS:
        return await HypertrophyEnv.from_docker_image(IMAGE_NAME)
    raise ValueError(
        "Temporary container startup is disabled. Set ENV_BASE_URL (or ENV_HTTP_URL) to an existing "
        "environment server, or set ALLOW_TEMP_CONTAINERS=1 to enable Docker fallback."
    )


async def run_episode(policy_name: str, policy_fn: Callable, client: Optional[OpenAI] = None) -> EpisodeResult:
    env = await create_env()
    llm_failures = 0
    rewards: List[float] = []
    trajectory: List[Dict] = []
    overtrain_days = 0
    last_obs = None
    steps = 0

    try:
        result = await env.reset()
        last_obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            if policy_name == "llm" and client is not None:
                try:
                    action = await llm_policy(last_obs, client)
                except Exception:
                    llm_failures += 1
                    action = fixed_policy(last_obs)
            else:
                action = policy_fn(last_obs)

            result = await env.step(action)
            obs = result.observation
            last_obs = obs
            steps = step
            r = float(result.reward or 0.0)
            rewards.append(r)

            meta = getattr(obs, "metadata", {}) or {}
            if meta.get("overtrain_days") is not None:
                overtrain_days = int(meta.get("overtrain_days", overtrain_days))
            trajectory.append(
                {
                    "day": int(getattr(obs, "day", step)),
                    "muscle_size": float(getattr(obs, "muscle_size", 50.0)),
                    "fatigue": float(getattr(obs, "fatigue", 0.0)),
                    "reward": r,
                    "action": {
                        "intensity": int(getattr(action, "intensity", 5)),
                        "volume": int(getattr(action, "volume", 5)),
                        "recovery_strategy": int(getattr(action, "recovery_strategy", 5)),
                    },
                }
            )

            if result.done:
                break

        if last_obs is None:
            return EpisodeResult(policy=policy_name, task=TASK_NAME, task_difficulty=TASK_DIFFICULTY[TASK_NAME],
                                 steps=0, total_reward=0.0, final_muscle=50.0,
                                 final_strength=50.0, avg_fatigue=0.0, overtrain_days=0,
                                 score=0.0, llm_failures=llm_failures, trajectory=trajectory)

        metadata = getattr(last_obs, "metadata", {}) or {}
        avg_fatigue = float(metadata.get("avg_fatigue", getattr(last_obs, "fatigue", 0.0)))
        final_muscle = float(getattr(last_obs, "muscle_size", 50.0))
        final_strength = float(getattr(last_obs, "strength", 50.0))

        score = choose_task_score(TASK_NAME, final_muscle, avg_fatigue, overtrain_days, max(steps, 1))

        return EpisodeResult(
            policy=policy_name,
            task=TASK_NAME,
            task_difficulty=TASK_DIFFICULTY[TASK_NAME],
            steps=steps,
            total_reward=sum(rewards),
            final_muscle=final_muscle,
            final_strength=final_strength,
            avg_fatigue=avg_fatigue,
            overtrain_days=overtrain_days,
            score=score,
            llm_failures=llm_failures,
            trajectory=trajectory,
        )
    finally:
        try:
            await env.close()
        except Exception:
            pass


def summarize(results: List[EpisodeResult], policy_name: str) -> None:
    subset = [r for r in results if r.policy == policy_name]
    if not subset:
        return

    def mean(values: List[float]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    def stdev(values: List[float]) -> float:
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    scores = [r.score for r in subset]
    rewards = [r.total_reward for r in subset]
    muscle = [r.final_muscle for r in subset]
    fatigue = [r.avg_fatigue for r in subset]
    overtrain = [float(r.overtrain_days) for r in subset]
    llm_fail = [float(r.llm_failures) for r in subset]

    print(f"\n=== {policy_name.upper()} ===")
    print(f"episodes={len(subset)}")
    print(f"score_mean={mean(scores):.4f} score_std={stdev(scores):.4f}")
    print(f"reward_mean={mean(rewards):.2f} reward_std={stdev(rewards):.2f}")
    print(f"final_muscle_mean={mean(muscle):.2f}")
    print(f"avg_fatigue_mean={mean(fatigue):.4f}")
    print(f"overtrain_days_mean={mean(overtrain):.2f}")
    if policy_name == "llm":
        print(f"llm_failures_mean={mean(llm_fail):.2f}")


def write_summary_csv(results: List[EpisodeResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "policy_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task",
            "task_difficulty",
            "policy",
            "steps",
            "total_reward",
            "final_muscle",
            "final_strength",
            "avg_fatigue",
            "overtrain_days",
            "score",
            "llm_failures",
        ])
        for r in results:
            writer.writerow([
                r.task,
                r.task_difficulty,
                r.policy,
                r.steps,
                f"{r.total_reward:.4f}",
                f"{r.final_muscle:.4f}",
                f"{r.final_strength:.4f}",
                f"{r.avg_fatigue:.4f}",
                r.overtrain_days,
                f"{r.score:.6f}",
                r.llm_failures,
            ])


def _pick_representative(results: List[EpisodeResult], policy_name: str) -> Optional[EpisodeResult]:
    subset = [r for r in results if r.policy == policy_name and r.trajectory]
    if not subset:
        return None
    subset.sort(key=lambda r: r.score, reverse=True)
    return subset[0]


def render_dual_axis_plot(results: List[EpisodeResult], output_dir: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib is not installed. Skipping trajectory plot.")
        return

    baseline = _pick_representative(results, "random")
    agent = _pick_representative(results, "llm") or _pick_representative(results, "heuristic")

    if baseline is None or agent is None:
        print("[WARN] Not enough trajectory data to render comparison plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    b_days = [p["day"] for p in baseline.trajectory]
    b_muscle = [p["muscle_size"] for p in baseline.trajectory]
    b_fatigue = [p["fatigue"] for p in baseline.trajectory]

    a_days = [p["day"] for p in agent.trajectory]
    a_muscle = [p["muscle_size"] for p in agent.trajectory]
    a_fatigue = [p["fatigue"] for p in agent.trajectory]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    ax1.plot(b_days, b_muscle, color="#d1495b", linewidth=2.2, linestyle="--", label="Random muscle")
    ax1.plot(a_days, a_muscle, color="#2b9348", linewidth=2.4, label=f"{agent.policy} muscle")

    ax2.plot(b_days, b_fatigue, color="#5c677d", linewidth=1.8, linestyle=":", label="Random fatigue")
    ax2.plot(a_days, a_fatigue, color="#1b4965", linewidth=1.9, label=f"{agent.policy} fatigue")

    ax1.set_xlabel("Day")
    ax1.set_ylabel("Muscle Size")
    ax2.set_ylabel("Fatigue")
    ax1.set_title("84-Day Trajectory: Baseline vs Agent")
    ax1.set_xlim(1, 84)
    ax1.set_ylim(50, 100)
    ax2.set_ylim(0, 1)
    ax1.grid(alpha=0.25)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    out_path = output_dir / "trajectory_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


async def main() -> None:
    random.seed(EVAL_SEED)
    print(
        f"[CONFIG] task={TASK_NAME} difficulty={TASK_DIFFICULTY[TASK_NAME]} "
        f"episodes={EVAL_EPISODES} eval_seed={EVAL_SEED}"
    )
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if (ENABLE_LLM_EVAL and API_KEY) else None

    policies = [
        ("random", random_policy),
        ("fixed_5_5_5", fixed_policy),
        ("heuristic", heuristic_policy),
    ]

    if ENABLE_LLM_EVAL:
        policies.append(("llm", None))

    all_results: List[EpisodeResult] = []

    for name, fn in policies:
        for ep in range(EVAL_EPISODES):
            if name == "llm":
                result = await run_episode(name, fixed_policy, client=client)
            else:
                result = await run_episode(name, fn)
            all_results.append(result)
            print(
                f"[EP] policy={name} ep={ep + 1}/{EVAL_EPISODES} "
                f"score={result.score:.4f} muscle={result.final_muscle:.2f} "
                f"fatigue={result.avg_fatigue:.4f} overtrain_days={result.overtrain_days}"
            )

    for name, _ in policies:
        summarize(all_results, name)

    artifact_dir = Path(ARTIFACT_DIR)
    write_summary_csv(all_results, artifact_dir)
    render_dual_axis_plot(all_results, artifact_dir)

    if ENABLE_LLM_EVAL:
        llm_mean = statistics.mean([r.score for r in all_results if r.policy == "llm"])
        heuristic_mean = statistics.mean([r.score for r in all_results if r.policy == "heuristic"])
        print("\n=== LEARNING SIGNAL ===")
        if llm_mean > heuristic_mean:
            print(f"LLM beats heuristic baseline by {llm_mean - heuristic_mean:.4f}")
        else:
            print(f"LLM does not beat heuristic baseline ({llm_mean:.4f} <= {heuristic_mean:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
