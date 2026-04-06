# Implementation Plan: 12-Week Hypertrophy & Fatigue Simulator

## Overview

Convert the echo environment into an AI-driven 84-day workout simulator where an LLM agent adapts through in-context history (not weight updates) to maximize muscle growth by balancing training intensity/volume with recovery while managing a fatigue meter.

### Learning Reality Check

- This project uses in-context learning during rollouts, not traditional RL weight updates.
- Qwen weights remain frozen; adaptation happens because each prompt includes recent trajectory, rewards, and fatigue outcomes.
- Learning quality is measured by trajectory and score improvements versus baselines (random/fixed/heuristic).

---

## MDP Specification

```
State Space S:
  day ∈ [0, 84]           — episode timestep
  muscle_size ∈ [50, 100]  — primary objective
  strength ∈ [50, 100]     — secondary metric
  fatigue ∈ [0.0, 1.0]     — resource to manage

Action Space A:
  intensity ∈ [1, 10]          — how hard you train
  volume ∈ [1, 10]             — how much you train
  recovery_strategy ∈ [1, 10]  — how much you recover

Transition Dynamics:
  effective_stimulus = intensity × volume × (1 - fatigue²)
  muscle_size' = clamp(muscle + stimulus × 0.02, 50, 100)
  strength'    = clamp(strength + stimulus × 0.012, 50, 100)
  fatigue'     = clamp(fatigue + i×v×0.008 - recovery×0.06, 0, 1)

Reward:
  base_reward     = (muscle_delta) × 10.0
  fatigue_penalty = -20.0 × max(0, fatigue' - 0.8)²
  recovery_bonus  = +1.0 if recovery ≥ 7 and fatigue' < 0.3
  R = base_reward + fatigue_penalty + recovery_bonus

Terminal: day ≥ 84
```

### Coefficient Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| `stimulus × 0.02` | Max gain/step = 2.0 | At moderate intensity over 84 steps, muscle grows ~50→85 |
| `fatigue × 0.008` | Max fatigue/step = 0.8 | High intensity fills fatigue fast but not instantly |
| `recovery × 0.06` | Max recovery/step = 0.6 | Recovery outpaces moderate-intensity fatigue accumulation |
| `(1 - fatigue²)` | Quadratic | Mild fatigue barely hurts; high fatigue devastates gains |
| `-20 × (f-0.8)²` | Quadratic penalty | Proportional to overtraining severity, not binary |

---

## Hackathon Task Variants (3 graders)

| Task Name | Env Var | Grader Formula | Score Range |
|-----------|---------|---------------|-------------|
| `muscle_gain` | `HYPERTROPHY_TASK=muscle_gain` | `(final_muscle - 50) / 50` | [0.0, 1.0] |
| `fatigue_management` | `HYPERTROPHY_TASK=fatigue_management` | `(1 - avg_fatigue) × muscle_score` | [0.0, 1.0] |
| `periodization` | `HYPERTROPHY_TASK=periodization` | `muscle_score×0.6 + no_overtrain_ratio×0.4` | [0.0, 1.0] |

---

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `models.py` | Action: intensity/volume/recovery_strategy. Observation: day/muscle/strength/fatigue/status | ✅ Done |
| `server/hypertrophy_env_environment.py` | Full MDP simulator with fatigue dynamics and reward shaping | ✅ Done |
| `client.py` | Updated `_step_payload` and `_parse_result` for new schemas | ✅ Done |
| `inference.py` | Multi-task graders, JSON parsing with regex fallback, score normalization | ✅ Done |
| `__init__.py` | No changes needed (exports match) | ✅ Verified |
| `server/__init__.py` | No changes needed (class name unchanged) | ✅ Verified |
| `server/app.py` | No changes needed (uses same class names) | ✅ Verified |
| `pyproject.toml` | No dependency changes (pure Python math, no numpy) | ✅ Verified |
| `Dockerfile` | No changes needed | ✅ Verified |

---

## Remaining Steps (Build, Test, Deploy)

### Step 1: Regenerate Lock File
```bash
cd "d:\Projects\scaler school\hypertrophy_env"
uv lock
```

### Step 2: Docker Build
```bash
docker build -t hypertrophy_env:latest .
```

### Step 3: HTTP Smoke Test
```bash
# Start container
docker run -d -p 8000:8000 --name hyp_test hypertrophy_env:latest

# Test endpoints
curl http://localhost:8000/schema
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"intensity": 7, "volume": 6, "recovery_strategy": 8}}'

# Cleanup
docker stop hyp_test && docker rm hyp_test
```

### Step 4: Run Inference (each task)
```bash
export HF_TOKEN="your-key"
export IMAGE_NAME="hypertrophy_env:latest"

# Task 1: muscle_gain (default)
HYPERTROPHY_TASK=muscle_gain python inference.py

# Task 2: fatigue_management
HYPERTROPHY_TASK=fatigue_management python inference.py

# Task 3: periodization
HYPERTROPHY_TASK=periodization python inference.py
```

### Step 5: Deploy to HF Spaces
```bash
openenv push
```

### Step 6: Run Pre-Submission Validator
```bash
# Run the hackathon validator script
```

---

## Verification Checklist

### Core Functionality
- [ ] Reset → day=0, muscle=50, strength=50, fatigue=0
- [ ] 84 steps → `done=true` on final step
- [ ] Muscle clamped to [50, 100], fatigue to [0, 1]
- [ ] Overtraining penalty triggers when fatigue > 0.8
- [ ] Recovery (0.06/unit) can outpace moderate training fatigue (0.008×i×v)
- [ ] Reward positive for balanced training, negative for overtraining

### OpenEnv Compliance
- [ ] `openenv.yaml` valid
- [ ] Typed Pydantic models (Action with ge/le validators)
- [ ] `/step`, `/reset`, `/state`, `/schema` endpoints work
- [ ] WebSocket sessions supported
- [ ] Docker image builds cleanly

### Hackathon Submission
- [ ] 3+ task variants with graders returning score ∈ [0.0, 1.0]
- [ ] `[START]`, `[STEP]`, `[END]` stdout format exact match
- [ ] Inference completes in < 20 minutes
- [ ] Uses OpenAI Client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- [ ] No heavy dependencies (vcpu=2, 8GB memory target)
- [ ] HF Space `/health` returns 200
- [ ] Pre-submission validator passes

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 84 days (12 weeks) | Real hypertrophy progression timeline |
| Quadratic fatigue penalty `(1-f²)` | Mild fatigue barely hurts, high fatigue devastates — natural periodization pressure |
| Quadratic overtraining penalty `-20×(f-0.8)²` | Proportional severity, not binary cliff — agent learns gradual avoidance |
| No numpy dependency | Pure Python arithmetic is sufficient; saves 60MB Docker image and build time |
| Regex-based JSON parsing | LLMs wrap JSON in markdown code blocks; regex extraction handles this robustly |
| Task graders on final state | Score = f(final_obs), not sum(rewards) — hackathon grader expects [0,1] |
| `client.py` updated | Without this, WebSocket payload serialization silently fails |
| `int` actions with ge/le | Discrete action space gives LLM clear choices; Pydantic validates server-side |
