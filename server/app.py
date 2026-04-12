from openenv.core.env_server.http_server import create_app

try:
    from server.hypertrophy_env_environment import HypertrophyEnvironment
except ModuleNotFoundError:
    from .hypertrophy_env_environment import HypertrophyEnvironment

try:
    from models import HypertrophyAction, HypertrophyObservation
except ModuleNotFoundError:
    from ..models import HypertrophyAction, HypertrophyObservation

# =========================
# CREATE APP
# =========================
app = create_app(
    HypertrophyEnvironment,
    action_cls=HypertrophyAction,
    observation_cls=HypertrophyObservation,
)


# =========================
# RUN SERVER
# =========================
def main():
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
