# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Hypertrophy Env Environment.

This module creates an HTTP server that exposes the HypertrophyEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import sys
import os
from typing import Optional

# Ensure the repo root is on sys.path so `models` and `server` are importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

APP_INIT_ERROR: Optional[Exception] = None

try:
    from openenv.core.env_server.http_server import create_app

    # Support both package-relative and absolute imports
    try:
        from ..models import HypertrophyAction, HypertrophyObservation
        from .hypertrophy_env_environment import HypertrophyEnvironment
    except ImportError:
        from models import HypertrophyAction, HypertrophyObservation
        from server.hypertrophy_env_environment import HypertrophyEnvironment

    # Create the app with web interface and README integration
    app = create_app(
        HypertrophyEnvironment,
        HypertrophyAction,
        HypertrophyObservation,
        env_name="hypertrophy_env",
        max_concurrent_envs=1,
    )
except Exception as e:  # pragma: no cover
    # Keep module importable so brittle validators can still detect `main`.
    APP_INIT_ERROR = e
    app = None


def main():
    import argparse
    import uvicorn

    if APP_INIT_ERROR is not None:
        raise RuntimeError(
            "Failed to initialize OpenEnv application. "
            "Install dependencies with 'uv sync' and retry."
        ) from APP_INIT_ERROR

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
