from __future__ import annotations

import math

from smartwave_ai.fleet_route_optimization.models import GeoPoint


EARTH_RADIUS_KM = 6371.0


def haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    lat1 = math.radians(a.lat)
    lon1 = math.radians(a.lon)
    lat2 = math.radians(b.lat)
    lon2 = math.radians(b.lon)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    h = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(h))


def geojson_line(points: list[GeoPoint]) -> list[list[float]]:
    return [[point.lon, point.lat] for point in points]

