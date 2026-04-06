---
title: Hypertrophy Environment Server
emoji: 💪
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Hypertrophy Environment

A 12-week (84-day) hypertrophy simulator where an AI agent learns to maximize muscle growth by balancing training intensity, volume, and recovery while managing fatigue. Overtraining is penalized; smart periodization is rewarded.

## Quick Start

```python
from hypertrophy_env import HypertrophyAction, HypertrophyEnv

try:
    # Create environment from Docker image
    env = HypertrophyEnv.from_docker_image("hypertrophy_env:latest")

    # Reset to day 0
    result = env.reset()
    print(f"Day 0: muscle={result.observation.muscle_size}, fatigue={result.observation.fatigue}")

    # Train for 84 days
    for day in range(1, 85):
        result = env.step(HypertrophyAction(
            intensity=7,
            volume=6,
            recovery_strategy=8,
        ))
        obs = result.observation
        print(f"Day {obs.day}: muscle={obs.muscle_size:.1f} fatigue={obs.fatigue:.2f} reward={result.reward:.2f}")

        if result.done:
            print(f"Program complete! Final muscle: {obs.muscle_size:.1f}")
            break

finally:
    env.close()
```

## Building the Docker Image

```bash
docker build -t hypertrophy_env:latest .
```

## Environment Details

### Action

**HypertrophyAction** — the agent's daily training decision:
- `intensity` (int, 1-10) — Training intensity (1=light, 10=max effort)
- `volume` (int, 1-10) — Training volume (1=minimal sets, 10=maximum volume)
- `recovery_strategy` (int, 1-10) — Recovery effort (1=poor, 10=optimal protocol)

### Observation

**HypertrophyObservation** — state after each training day:
- `day` (int) — Current training day (0-84)
- `muscle_size` (float) — Muscle size score (50.0-100.0)
- `strength` (float) — Strength score (50.0-100.0)
- `fatigue` (float) — Fatigue level (0.0=fresh, 1.0=overtrained)
- `status_message` (str) — Human-readable status
- `reward` (float) — Step reward (inherited from base)
- `done` (bool) — Whether episode is complete (inherited from base)
- `metadata` (dict) — Additional metrics (week, effective_stimulus, avg_fatigue, overtrain_days)

### Reward

The reward is shaped to encourage sustainable training:
- **Base reward**: `muscle_delta × 10.0` — proportional to actual muscle gained
- **Fatigue penalty**: `-20.0 × max(0, fatigue - 0.8)²` — quadratic penalty for overtraining
- **Recovery bonus**: `+1.0` if recovery ≥ 7 and fatigue < 0.3

### Physics

- **Fatigue reduces effectiveness**: `effective_stimulus = intensity × volume × (1 - fatigue²)`
- **Fatigue accumulation**: `fatigue += intensity × volume × 0.008`
- **Fatigue recovery**: `fatigue -= recovery_strategy × 0.06`
- **Muscle growth**: `muscle_size += effective_stimulus × 0.02` (clamped to 100)
- **Strength growth**: `strength += effective_stimulus × 0.012` (clamped to 100)

### Tasks (3 variants with explicit difficulty)

| Difficulty | Task | Description | Score Formula |
|------------|------|-------------|---------------|
| Easy | `muscle_gain` | Maximize final muscle size | `(muscle - 50) / 50` |
| Medium | `fatigue_management` | Grow muscle while keeping fatigue low | `(1 - avg_fatigue) × muscle_score` |
| Hard | `periodization` | Achieve gains with minimal overtraining | `muscle×0.6 + no_overtrain×0.4` |

Select task via: `HYPERTROPHY_TASK=muscle_gain`

### Reproducibility Notes

- **Baseline reproducibility (strong)**: `evaluate_agent.py` is seeded (`EVAL_SEED`, default `42`) and supports deterministic baseline policies.
- **LLM reproducibility (best effort)**: external APIs can still vary slightly. For stronger reproducibility in `inference.py`, use:
    - `REPRODUCIBLE_MODE=1` (default) to force deterministic decoding settings
    - `INFERENCE_SEED=42` for local deterministic behavior
    - `TEMPERATURE=0.0` for low-variance generation
- For a fair comparison, keep model, task, and env vars fixed across repeated runs and report mean/std over multiple episodes.

## Advanced Usage

### Connecting to an Existing Server

```python
from hypertrophy_env import HypertrophyAction, HypertrophyEnv

env = HypertrophyEnv(base_url="http://localhost:8000")
result = env.reset()
result = env.step(HypertrophyAction(intensity=5, volume=5, recovery_strategy=7))
```

### Using the Context Manager

```python
from hypertrophy_env import HypertrophyAction, HypertrophyEnv

with HypertrophyEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    for day in range(1, 85):
        result = env.step(HypertrophyAction(intensity=6, volume=5, recovery_strategy=8))
        if result.done:
            break
```

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Development & Testing

## Proving Agent Adaptation

This project does not fine-tune model weights during rollout.
Adaptation happens through in-context learning: recent trajectory, reward, and fatigue history are fed back into each next-day prompt.

### 1) Run Baseline vs Agent Evaluation

```bash
# Baselines only (random/fixed/heuristic)
EVAL_EPISODES=20 ENABLE_LLM_EVAL=0 python evaluate_agent.py

# Include LLM policy (requires API key/model access)
EVAL_EPISODES=20 ENABLE_LLM_EVAL=1 python evaluate_agent.py
```

Artifacts generated in `artifacts/`:
- `policy_summary.csv` - per-episode metrics for tables/charts
- `trajectory_comparison.png` - dual-axis plot (muscle and fatigue) for baseline vs agent

### 2) Capture LLM Reasoning Trace

`inference.py` logs full day-by-day trajectory (state, action, reward, thought) to a JSON artifact file.

```bash
SAVE_TRAJECTORY=1 ARTIFACT_DIR=artifacts python inference.py
```

This provides judge-friendly evidence that the policy reacts to fatigue penalties and recovery opportunities across the 84-day horizon.

### Direct Environment Testing

```bash
python3 server/hypertrophy_env_environment.py
```

### Running Locally

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
hypertrophy_env/
├── .dockerignore          # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # HypertrophyEnv client (WebSocket)
├── models.py              # Action and Observation models
├── inference.py           # LLM agent inference script
├── implementation_plan.md # Implementation plan and checklist
└── server/
    ├── __init__.py        # Server module exports
    ├── hypertrophy_env_environment.py  # Core MDP environment
    └── app.py             # FastAPI application (HTTP + WebSocket)
```
