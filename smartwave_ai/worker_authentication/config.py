import os

from smartwave_ai.visual_analysis.config import BASE_DIR


JWT_ISSUER = "mygov-smartwave-local"
JWT_SECRET = os.getenv("SMARTWAVE_JWT_SECRET", "smartwave-dev-secret")
GPS_RADIUS_CHECK_METERS = 50
DEFAULT_WORKER_ACTION_LOG_PATH = BASE_DIR / "runtime" / "worker_actions.json"

