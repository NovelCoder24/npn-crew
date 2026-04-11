from fastapi import FastAPI, HTTPException
from threading import Lock
from typing import Any, Dict

import uvicorn


app = FastAPI()
_env = None
_env_lock = Lock()


def _get_env():
    """Lazy-initialize the environment to avoid import failures at module load time."""
    global _env
    if _env is None:
        from server.hypertrophy_env_environment import HypertrophyEnvironment
        _env = HypertrophyEnvironment()
    return _env


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _step_payload(obs) -> Dict[str, Any]:
    return {
        "observation": _model_to_dict(obs),
        "reward": float(getattr(obs, "reward", 0.0)),
        "done": bool(getattr(obs, "done", False)),
    }


@app.post("/reset")
def reset():
    try:
        env = _get_env()
        with _env_lock:
            obs = env.reset()
            return _step_payload(obs)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step(action: Dict[str, Any]):
    try:
        from models import HypertrophyAction
        parsed_action = HypertrophyAction(**action)
        env = _get_env()
        with _env_lock:
            obs = env.step(parsed_action)
            return _step_payload(obs)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
def state():
    try:
        env = _get_env()
        with _env_lock:
            st = env.state
            return {
                "episode_id": st.episode_id,
                "step_count": st.step_count,
            }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/schema")
def schema() -> Dict[str, Any]:
    from models import HypertrophyAction, HypertrophyObservation
    action_schema = (
        HypertrophyAction.model_json_schema()
        if hasattr(HypertrophyAction, "model_json_schema")
        else HypertrophyAction.schema()
    )
    observation_schema = (
        HypertrophyObservation.model_json_schema()
        if hasattr(HypertrophyObservation, "model_json_schema")
        else HypertrophyObservation.schema()
    )
    return {
        "action": action_schema,
        "observation": observation_schema,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == '__main__':
    main()
