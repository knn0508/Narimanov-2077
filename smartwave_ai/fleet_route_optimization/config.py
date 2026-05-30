from smartwave_ai.visual_analysis.config import BASE_DIR


ROUTE_OPTIMIZATION_ALGORITHM = "clarke-wright-v2"
SYSTEM_TIMEZONE = "Asia/Baku"

DEFAULT_DEPOT_LOCATION = {"lat": 40.4099, "lon": 49.8667}
DEFAULT_VEHICLES_PATH = BASE_DIR / "data" / "fleet_vehicles.json"

CLUSTER_RADIUS_METERS = 500
AVERAGE_URBAN_SPEED_KMH = 22.0
SERVICE_MINUTES_PER_CONTAINER = 8
FUEL_LITERS_PER_KM = 0.28
CO2_KG_PER_LITER_DIESEL = 2.68

DEFAULT_CONTAINER_VOLUME_M3 = {
    "organic": 1.1,
    "mixed": 1.3,
    "recycling": 0.9,
}

