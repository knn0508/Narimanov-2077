from pathlib import Path

from smartwave_ai.visual_analysis.config import BASE_DIR


PREDICTION_MODEL_VERSION = "prophet-ensemble-waste-v1.0"
BASELINE_MODEL_NAME = "prophet-compatible-baseline-v1"
ENSEMBLE_MODEL_NAME = "xgboost-zone-ensemble-v1"

RED_FULLNESS_THRESHOLD = 71
URGENT_DISPATCH_HOURS = 12
PROACTIVE_SCHEDULE_HOURS = 36
ROLLING_HISTORY_DAYS = 30
FORECAST_HORIZON_HOURS = 24 * 14

DEFAULT_FILL_HISTORY_PATH = BASE_DIR / "runtime" / "container_fill_history.json"
DEFAULT_COLLECTION_LOG_PATH = BASE_DIR / "runtime" / "collection_log.json"
DEFAULT_ZONE_DEMOGRAPHICS_PATH = BASE_DIR / "data" / "zone_demographics.json"
DEFAULT_WEATHER_DATA_PATH = BASE_DIR / "data" / "weather_data.json"
DEFAULT_DYNAMIC_HOLIDAYS_PATH = BASE_DIR / "data" / "az_dynamic_holidays.json"
