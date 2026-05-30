# smartwave_ai/fleet_route_optimization/osrm_client.py
import requests
from smartwave_ai.fleet_route_optimization.models import GeoPoint

OSRM_BASE_URL = "http://localhost:5000/route/v1/driving"

def get_street_distance_and_geometry(points: list[GeoPoint]) -> dict | None:
    """
    Queries OSRM server for the shortest street path connecting points sequentially.
    Returns travel distance (km), duration (minutes), and GeoJSON LineString coordinates.
    """
    coords = ";".join([f"{pt.lon},{pt.lat}" for pt in points])
    url = f"{OSRM_BASE_URL}/{coords}?overview=full&geometries=geojson"
    
    try:
        response = requests.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        if "routes" not in data or not data["routes"]:
            raise ValueError("No route found between points.")
            
        best_route = data["routes"][0]
        return {
            "distance_km": best_route["distance"] / 1000.0,
            "duration_minutes": best_route["duration"] / 60.0,
            "coordinates": best_route["geometry"]["coordinates"]
        }
    except Exception as e:
        # Fallback to straight-line approximations if OSRM service is down
        print(f"OSRM Error: {e}. Falling back to Haversine calculations.")
        return None
