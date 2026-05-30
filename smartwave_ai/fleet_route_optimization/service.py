from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from uuid import uuid4

from smartwave_ai.fleet_route_optimization.config import (
    AVERAGE_URBAN_SPEED_KMH,
    CLUSTER_RADIUS_METERS,
    CO2_KG_PER_LITER_DIESEL,
    DEFAULT_CONTAINER_VOLUME_M3,
    DEFAULT_DEPOT_LOCATION,
    FUEL_LITERS_PER_KM,
    ROUTE_OPTIMIZATION_ALGORITHM,
    SERVICE_MINUTES_PER_CONTAINER,
    SYSTEM_TIMEZONE,
)
from smartwave_ai.fleet_route_optimization.geo import geojson_line, haversine_km
from smartwave_ai.fleet_route_optimization.models import (
    GeoJsonLineString,
    GeoPoint,
    MinimumVehicleRecommendation,
    OptimizationTrigger,
    RouteContainer,
    RouteManifest,
    RouteOptimizationRequest,
    ServiceContainerInput,
    VehicleInput,
    VehicleRoute,
)
from smartwave_ai.fleet_route_optimization.repository import VehicleRepository
from smartwave_ai.multi_report_validation.models import IncidentStatus, ReportType
from smartwave_ai.multi_report_validation.service import MultiReportValidationService
from smartwave_ai.predictive_analytics.service import PredictiveAnalyticsService
from smartwave_ai.visual_analysis.audit import AuditLogger, build_audit_entry, sha256_hex
from smartwave_ai.visual_analysis.models import StatusColor
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.visual_analysis.service import assign_status_color


class RouteOptimizationError(Exception):
    error_code = "ERR_ROUTE_OPTIMIZATION_FAILED"


class NoServiceContainersError(RouteOptimizationError):
    error_code = "ERR_NO_SERVICE_CONTAINERS"


class NoVehiclesAvailableError(RouteOptimizationError):
    error_code = "ERR_NO_VEHICLES_AVAILABLE"


class FleetRouteOptimizationService:
    def __init__(
        self,
        *,
        registry: ContainerRegistry,
        predictive_service: PredictiveAnalyticsService,
        validation_service: MultiReportValidationService,
        vehicle_repository: VehicleRepository,
        audit_logger: AuditLogger,
    ) -> None:
        self.registry = registry
        self.predictive_service = predictive_service
        self.validation_service = validation_service
        self.vehicle_repository = vehicle_repository
        self.audit_logger = audit_logger
        self.generated_manifests: list[RouteManifest] = []

    def optimize_routes(
        self,
        *,
        request: RouteOptimizationRequest,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
        operator_id: str | None = None,
    ) -> RouteManifest:
        generated_at = datetime.now(timezone.utc)
        depot = request.depot_location or GeoPoint(**DEFAULT_DEPOT_LOCATION)
        vehicles = request.available_vehicles or self.vehicle_repository.all()
        if not vehicles:
            raise NoVehiclesAvailableError("No available vehicles supplied or configured.")

        containers = self._prepare_service_containers(
            request.containers_needing_service,
            now=generated_at,
        )
        if not containers:
            raise NoServiceContainersError("No containers require service.")
        if request.trigger == OptimizationTrigger.EVENT_DRIVEN and not self.event_trigger_ready(
            containers
        ):
            raise NoServiceContainersError(
                "Event-driven route optimization requires at least 3 RED containers in one route zone."
            )

        ranked = sorted(
            containers,
            key=lambda item: (item.urgency_score or 0.0),
            reverse=True,
        )
        clusters = self._cluster_by_proximity(ranked)
        ordered_groups = [
            self._clarke_wright_order(cluster, depot) for cluster in clusters
        ]
        routes = self._assign_groups_to_vehicles(
            ordered_groups=ordered_groups,
            vehicles=vehicles,
            depot=depot,
            traffic_multiplier=request.traffic_multiplier,
        )
        recommendation = self._minimum_vehicle_recommendation(ranked, vehicles)
        audit_entry_id = str(uuid4())
        manifest = RouteManifest(
            route_id=str(uuid4()),
            generated_at_utc=generated_at,
            timezone=SYSTEM_TIMEZONE,
            vehicles_assigned=len(routes),
            routes=routes,
            optimization_algorithm=ROUTE_OPTIMIZATION_ALGORITHM,
            minimum_vehicle_recommendation=recommendation,
            trigger=request.trigger,
            audit_entry_id=audit_entry_id,
        )
        self._append_route_audit(
            manifest=manifest,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            operator_id=operator_id,
        )
        self.generated_manifests.append(manifest)
        return manifest

    def event_trigger_ready(
        self, containers: list[ServiceContainerInput] | None = None
    ) -> bool:
        prepared = self._prepare_service_containers(containers, now=datetime.now(timezone.utc))
        red_counts: dict[str, int] = {}
        for container in prepared:
            if container.status_color == StatusColor.RED.value:
                zone = container.route_zone or "UNKNOWN"
                red_counts[zone] = red_counts.get(zone, 0) + 1
        return any(count >= 3 for count in red_counts.values())

    def _prepare_service_containers(
        self,
        supplied: list[ServiceContainerInput] | None,
        now: datetime,
    ) -> list[ServiceContainerInput]:
        if supplied is not None:
            return [self._hydrate_container_input(item, now) for item in supplied]
        return self._derive_service_containers(now)

    def _derive_service_containers(self, now: datetime) -> list[ServiceContainerInput]:
        derived: list[ServiceContainerInput] = []
        for container in self.registry._records:
            latest = self.predictive_service.fill_history.latest_for_container(
                container.container_id
            )
            fullness = latest.fullness_score if latest else 0
            status = assign_status_color(fullness).value
            validation_count = self._validation_report_count(container.container_id)
            fire_hazard = self._has_fire_hazard(container.container_id)
            odor_risk = self._has_verified_odor(container.container_id)
            needs_service = (
                status == StatusColor.RED.value
                or validation_count > 0
                or fire_hazard
            )
            if not needs_service:
                continue
            derived.append(
                self._hydrate_container_input(
                    ServiceContainerInput(
                        container_id=container.container_id,
                        fullness_score=fullness,
                        status_color=status,
                        odor_risk_flag=odor_risk,
                        validation_report_count=validation_count,
                        fire_hazard_flag=fire_hazard,
                    ),
                    now,
                )
            )
        return derived

    def _hydrate_container_input(
        self, item: ServiceContainerInput, now: datetime
    ) -> ServiceContainerInput:
        record = self.registry.resolve(item.container_id)
        coords = item.coords
        route_zone = item.route_zone
        volume_m3 = item.volume_m3
        hours_since_last_collection = item.hours_since_last_collection
        if record is not None:
            coords = coords or GeoPoint(
                lat=record.geo_coordinates.lat,
                lon=record.geo_coordinates.lon,
            )
            route_zone = route_zone or record.assigned_route_id
            volume_m3 = volume_m3 or DEFAULT_CONTAINER_VOLUME_M3.get(
                str(record.container_type),
                1.0,
            )
            if hours_since_last_collection is None:
                last_collection = self.predictive_service.collection_log.latest_for_container(
                    item.container_id
                )
                last_emptied = (
                    last_collection.cleaned_at_utc
                    if last_collection
                    else record.last_emptied_timestamp
                )
                if last_emptied.tzinfo is None:
                    last_emptied = last_emptied.replace(tzinfo=timezone.utc)
                hours_since_last_collection = max(
                    0.0, (now - last_emptied).total_seconds() / 3600
                )
        if coords is None:
            raise RouteOptimizationError(
                f"Container {item.container_id} has no coordinates."
            )
        fullness = item.fullness_score if item.fullness_score is not None else 0
        status_color = item.status_color or assign_status_color(fullness).value
        validation_count = item.validation_report_count or self._validation_report_count(
            item.container_id
        )
        fire_hazard = item.fire_hazard_flag or self._has_fire_hazard(item.container_id)
        odor_risk = item.odor_risk_flag or self._has_verified_odor(item.container_id)
        urgency_score = (
            999.0
            if fire_hazard
            else item.urgency_score
            if item.urgency_score is not None
            else self.compute_urgency_score(
                fullness_score=fullness,
                odor_risk_flag=odor_risk,
                hours_since_last_collection=hours_since_last_collection or 0.0,
                validation_report_count=validation_count,
                calendar_factor_multiplier=item.calendar_factor_multiplier,
                fire_hazard_flag=fire_hazard,
            )
        )
        return ServiceContainerInput(
            container_id=item.container_id,
            coords=coords,
            status_color=status_color,
            urgency_score=round(urgency_score, 2),
            fullness_score=fullness,
            odor_risk_flag=odor_risk,
            hours_since_last_collection=hours_since_last_collection,
            validation_report_count=validation_count,
            calendar_factor_multiplier=item.calendar_factor_multiplier,
            fire_hazard_flag=fire_hazard,
            volume_m3=volume_m3 or 1.0,
            route_zone=route_zone or "UNASSIGNED",
        )

    def compute_urgency_score(
        self,
        *,
        fullness_score: int,
        odor_risk_flag: bool,
        hours_since_last_collection: float,
        validation_report_count: int,
        calendar_factor_multiplier: float,
        fire_hazard_flag: bool = False,
    ) -> float:
        if fire_hazard_flag:
            return 999.0
        return (
            (fullness_score * 0.5)
            + (25 if odor_risk_flag else 0)
            + (hours_since_last_collection * 0.8)
            + (validation_report_count * 5)
            + (calendar_factor_multiplier * 10)
        )

    def _validation_report_count(self, container_id: str) -> int:
        return sum(
            1
            for report in self.validation_service.incidents.all()
            if report.container_id == container_id
            and report.status == IncidentStatus.VERIFIED
        )

    def _has_fire_hazard(self, container_id: str) -> bool:
        return any(
            report.container_id == container_id
            and report.report_type == ReportType.FIRE_HAZARD
            and report.status == IncidentStatus.VERIFIED
            for report in self.validation_service.incidents.all()
        )

    def _has_verified_odor(self, container_id: str) -> bool:
        return any(
            report.container_id == container_id
            and report.report_type == ReportType.ODOR_COMPLAINT
            and report.status == IncidentStatus.VERIFIED
            for report in self.validation_service.incidents.all()
        )

    def _cluster_by_proximity(
        self, containers: list[ServiceContainerInput]
    ) -> list[list[ServiceContainerInput]]:
        clusters: list[list[ServiceContainerInput]] = []
        for container in containers:
            placed = False
            for cluster in clusters:
                if any(
                    haversine_km(container.coords, other.coords) * 1000
                    <= CLUSTER_RADIUS_METERS
                    for other in cluster
                ):
                    cluster.append(container)
                    placed = True
                    break
            if not placed:
                clusters.append([container])
        return clusters

    def _clarke_wright_order(
        self, containers: list[ServiceContainerInput], depot: GeoPoint
    ) -> list[ServiceContainerInput]:
        if len(containers) <= 2:
            return containers
        routes = [[container] for container in containers]
        savings: list[tuple[float, ServiceContainerInput, ServiceContainerInput]] = []
        for index, left in enumerate(containers):
            for right in containers[index + 1 :]:
                saving = (
                    haversine_km(depot, left.coords)
                    + haversine_km(depot, right.coords)
                    - haversine_km(left.coords, right.coords)
                )
                savings.append((saving, left, right))
        savings.sort(key=lambda item: item[0], reverse=True)

        for _, left, right in savings:
            left_route = self._find_route(routes, left)
            right_route = self._find_route(routes, right)
            if left_route is None or right_route is None or left_route is right_route:
                continue
            if left_route[-1] == left and right_route[0] == right:
                left_route.extend(right_route)
                routes.remove(right_route)
            elif right_route[-1] == right and left_route[0] == left:
                right_route.extend(left_route)
                routes.remove(left_route)
            elif left_route[0] == left and right_route[0] == right:
                left_route.reverse()
                left_route.extend(right_route)
                routes.remove(right_route)
            elif left_route[-1] == left and right_route[-1] == right:
                right_route.reverse()
                left_route.extend(right_route)
                routes.remove(right_route)

        ordered: list[ServiceContainerInput] = []
        for route in routes:
            ordered.extend(route)
        return ordered

    def _find_route(
        self,
        routes: list[list[ServiceContainerInput]],
        container: ServiceContainerInput,
    ) -> list[ServiceContainerInput] | None:
        for route in routes:
            if container in route:
                return route
        return None

    def _assign_groups_to_vehicles(
        self,
        *,
        ordered_groups: list[list[ServiceContainerInput]],
        vehicles: list[VehicleInput],
        depot: GeoPoint,
        traffic_multiplier: float,
    ) -> list[VehicleRoute]:
        groups = sorted(
            ordered_groups,
            key=lambda group: sum(item.urgency_score or 0 for item in group),
            reverse=True,
        )
        vehicles_by_capacity = sorted(vehicles, key=lambda item: item.capacity_m3, reverse=True)
        routes: list[VehicleRoute] = []
        used_vehicle_ids: set[str] = set()
        for group in groups:
            remaining = list(group)
            while remaining:
                vehicle = self._select_vehicle(remaining, vehicles_by_capacity, used_vehicle_ids)
                if vehicle is None:
                    break
                chunk, remaining = self._take_capacity_chunk(remaining, vehicle.capacity_m3)
                if not chunk:
                    chunk = [remaining.pop(0)]
                used_vehicle_ids.add(vehicle.vehicle_id)
                routes.append(
                    self._build_vehicle_route(
                        vehicle=vehicle,
                        containers=chunk,
                        depot=depot,
                        traffic_multiplier=traffic_multiplier,
                    )
                )
        return routes

    def _select_vehicle(
        self,
        containers: list[ServiceContainerInput],
        vehicles: list[VehicleInput],
        used_vehicle_ids: set[str],
    ) -> VehicleInput | None:
        available = [vehicle for vehicle in vehicles if vehicle.vehicle_id not in used_vehicle_ids]
        if not available:
            available = vehicles
        first = containers[0]
        return min(
            available,
            key=lambda vehicle: (
                haversine_km(vehicle.current_location, first.coords),
                -vehicle.capacity_m3,
            ),
        )

    def _take_capacity_chunk(
        self, containers: list[ServiceContainerInput], capacity_m3: float
    ) -> tuple[list[ServiceContainerInput], list[ServiceContainerInput]]:
        chunk: list[ServiceContainerInput] = []
        volume = 0.0
        index = 0
        for container in containers:
            next_volume = volume + (container.volume_m3 or 1.0)
            if chunk and next_volume > capacity_m3:
                break
            chunk.append(container)
            volume = next_volume
            index += 1
        return chunk, containers[index:]

    def _build_vehicle_route(
        self,
        *,
        vehicle: VehicleInput,
        containers: list[ServiceContainerInput],
        depot: GeoPoint,
        traffic_multiplier: float,
    ) -> VehicleRoute:
        points = [vehicle.current_location] + [item.coords for item in containers] + [depot]
        
        from smartwave_ai.fleet_route_optimization.osrm_client import get_street_distance_and_geometry
        osrm_data = get_street_distance_and_geometry(points)
        
        if osrm_data:
            distance_km = osrm_data["distance_km"] * traffic_multiplier
            duration_minutes = int(
                round(
                    (osrm_data["duration_minutes"] * traffic_multiplier)
                    + (len(containers) * SERVICE_MINUTES_PER_CONTAINER)
                )
            )
            waypoints_coords = osrm_data["coordinates"]
        else:
            distance_km = self._route_distance(points) * traffic_multiplier
            duration_minutes = int(
                round(
                    (distance_km / AVERAGE_URBAN_SPEED_KMH * 60)
                    + (len(containers) * SERVICE_MINUTES_PER_CONTAINER)
                )
            )
            waypoints_coords = geojson_line(points)
            
        fuel_liters = distance_km * FUEL_LITERS_PER_KM
        return VehicleRoute(
            vehicle_id=vehicle.vehicle_id,
            driver_id=vehicle.driver_id,
            waypoints=[
                GeoJsonLineString(
                    coordinates=waypoints_coords,
                )
            ],
            containers=[
                RouteContainer(
                    container_id=item.container_id,
                    urgency_score=round(item.urgency_score or 0.0, 2),
                    status_color=item.status_color or "UNKNOWN",
                    volume_m3=round(item.volume_m3 or 1.0, 2),
                    coords=item.coords,
                    route_zone=item.route_zone or "UNASSIGNED",
                )
                for item in containers
            ],
            estimated_duration_minutes=duration_minutes,
            estimated_distance_km=round(distance_km, 2),
            fuel_estimate_liters=round(fuel_liters, 2),
            co2_footprint_kg=round(fuel_liters * CO2_KG_PER_LITER_DIESEL, 2),
        )

    def _route_distance(self, points: list[GeoPoint]) -> float:
        return sum(haversine_km(left, right) for left, right in zip(points, points[1:]))

    def _minimum_vehicle_recommendation(
        self, containers: list[ServiceContainerInput], vehicles: list[VehicleInput]
    ) -> MinimumVehicleRecommendation:
        total_volume = sum(item.volume_m3 or 1.0 for item in containers)
        average_capacity = sum(vehicle.capacity_m3 for vehicle in vehicles) / len(vehicles)
        route_zones = {item.route_zone or "UNASSIGNED" for item in containers}
        recommended = max(1, ceil(total_volume / average_capacity))
        red_count = sum(
            1
            for item in containers
            if (item.status_color or "").upper() == StatusColor.RED.value
        )
        vehicle_word = "vehicle" if recommended == 1 else "vehicles"
        container_word = "container" if red_count == 1 else "containers"
        zone_word = "zone" if len(route_zones) == 1 else "zones"
        reasoning = (
            f"Recommending {recommended} {vehicle_word} to service {red_count} RED "
            f"{container_word} across {len(route_zones)} {zone_word} within the 4-hour SLA window."
        )
        return MinimumVehicleRecommendation(
            recommended_vehicles=recommended,
            total_volume_m3=round(total_volume, 2),
            average_vehicle_capacity_m3=round(average_capacity, 2),
            reasoning=reasoning,
        )

    def _append_route_audit(
        self,
        *,
        manifest: RouteManifest,
        session_id: str | None,
        ip_address_hash: str | None,
        operator_id: str | None,
    ) -> None:
        container_ids = [
            container.container_id
            for route in manifest.routes
            for container in route.containers
        ]
        audit_entry = build_audit_entry(
            audit_entry_id=manifest.audit_entry_id,
            module="ROUTING",
            action="ROUTE_MANIFEST_GENERATED",
            input_hash=sha256_hex("|".join(container_ids).encode("utf-8")),
            output_summary=(
                f"{manifest.vehicles_assigned} vehicles assigned for "
                f"{len(container_ids)} service containers"
            ),
            model_used=ROUTE_OPTIMIZATION_ALGORITHM,
            confidence_score=1.0,
            human_reviewable=False,
            operator_id=operator_id,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra={
                "event_type": "ROUTE_MANIFEST_GENERATED",
                "route_id": manifest.route_id,
                "trigger": manifest.trigger.value if hasattr(manifest.trigger, "value") else manifest.trigger,
                "vehicles_assigned": manifest.vehicles_assigned,
                "container_ids": container_ids,
                "minimum_vehicle_recommendation": manifest.minimum_vehicle_recommendation.model_dump(
                    mode="json"
                ),
            },
        )
        self.audit_logger.append(audit_entry)
