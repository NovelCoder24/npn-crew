from fastapi import FastAPI
import uvicorn
from fastapi import HTTPException
from threading import Lock
from typing import Any, Dict

from models import HypertrophyAction, HypertrophyObservation
from server.hypertrophy_env_environment import HypertrophyEnvironment


app = FastAPI()
_env = HypertrophyEnvironment()
_env_lock = Lock()


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _step_payload(obs: HypertrophyObservation) -> Dict[str, Any]:
    return {
        "observation": _model_to_dict(obs),
        "reward": float(getattr(obs, "reward", 0.0)),
        "done": bool(getattr(obs, "done", False)),
    }


@app.post("/reset")
def reset():
    try:
        with _env_lock:
            obs = _env.reset()
            return _step_payload(obs)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step(action: HypertrophyAction):
    try:
        with _env_lock:
            obs = _env.step(action)
            return _step_payload(obs)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/state")
def state():
    try:
        with _env_lock:
            st = _env.state
            return {
                "episode_id": st.episode_id,
                "step_count": st.step_count,
            }
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/schema")
def schema() -> Dict[str, Any]:
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
