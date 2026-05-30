from __future__ import annotations

import json
from pathlib import Path

from smartwave_ai.fleet_route_optimization.models import VehicleInput


class VehicleRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._vehicles = self._load()

    def _load(self) -> list[VehicleInput]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as file:
            raw_vehicles = json.load(file)
        return [VehicleInput.model_validate(vehicle) for vehicle in raw_vehicles]

    def all(self) -> list[VehicleInput]:
        return list(self._vehicles)

