"""FastAPI application entrypoint for the Hypertrophy OpenEnv server."""

import os
import sys
from typing import Optional

# Ensure repo root is importable when this module is executed directly.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

APP_INIT_ERROR: Optional[Exception] = None

try:
    from openenv.core.env_server.http_server import create_app

    # Support both package-relative and absolute imports.
    try:
        from ..models import HypertrophyAction, HypertrophyObservation
        from .hypertrophy_env_environment import HypertrophyEnvironment
    except ImportError:
        from models import HypertrophyAction, HypertrophyObservation
        from server.hypertrophy_env_environment import HypertrophyEnvironment

    app = create_app(
        HypertrophyEnvironment,
        HypertrophyAction,
        HypertrophyObservation,
        env_name="hypertrophy_env",
        max_concurrent_envs=1,
    )
except Exception as e:  # pragma: no cover
    APP_INIT_ERROR = e
    from fastapi import FastAPI

    app = FastAPI(title="Hypertrophy Env Server")

    @app.get("/health")
    def health() -> dict:
        return {"status": "degraded", "error": str(APP_INIT_ERROR)}


def main() -> None:
    import uvicorn

    if APP_INIT_ERROR is not None:
        raise RuntimeError(
            "Failed to initialize OpenEnv application. "
            "Install dependencies with 'uv sync' and retry."
        ) from APP_INIT_ERROR

    # Keep main() a no-arg callable for brittle validator checks.
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()
