# SmartWave AI

FastAPI service for the SmartWave visual analysis module in Narimanov District,
Baku. The implemented endpoint accepts raw image bytes from the QR-linked
container report flow and returns the Module 1 JSON contract.

## Run

```powershell
uvicorn smartwave_ai.app:app --reload
```

Submit a report with a registered QR UUID:

```powershell
Invoke-WebRequest `
  -Method POST `
  -Uri http://127.0.0.1:8000/api/v1/containers/NRM-ORG-001/report `
  -Headers @{"X-QR-Code-UUID"="1b8bcf70-4a18-48f2-a6a6-b3f3f5c10a01"} `
  -ContentType "image/jpeg" `
  -InFile .\sample.jpg
```

Submit a non-visual citizen complaint:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/api/v1/containers/NRM-ORG-001/complaints `
  -ContentType "application/json" `
  -Body '{
    "report_type": "ODOR_COMPLAINT",
    "citizen_id": "citizen-demo-1",
    "device_fingerprint": "device-demo-1"
  }'
```

Confirm a structural-damage report as a worker:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/api/v1/incidents/{report_id}/worker-confirmation `
  -ContentType "application/json" `
  -Body '{
    "worker_id": "worker-7",
    "operator_id": "supervisor-1"
  }'
```

Request a proactive RED-status forecast:

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri http://127.0.0.1:8000/api/v1/containers/NRM-ORG-001/forecast
```

Generate a fleet route manifest:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/api/v1/routes/optimize `
  -ContentType "application/json" `
  -Body '{
    "trigger": "manual",
    "containers_needing_service": [
      {
        "container_id": "NRM-ORG-001",
        "status_color": "RED",
        "fullness_score": 82,
        "volume_m3": 1.2,
        "validation_report_count": 1
      }
    ]
  }'
```

Detect which panel to show after a QR scan:

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri http://127.0.0.1:8000/api/v1/containers/NRM-ORG-001/panel-route
```

Use a development JWT for worker endpoints:

```powershell
$token = python -c "from smartwave_ai.worker_authentication.jwt import create_dev_jwt; import time; print(create_dev_jwt({'sub':'worker-demo-1','worker_id':'worker-demo-1','role':'WORKER','exp':int(time.time())+3600}))"
```

Mark a container as cleaned:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri http://127.0.0.1:8000/api/v1/worker/containers/NRM-ORG-001/mark-cleaned `
  -Headers @{Authorization="Bearer $token"} `
  -ContentType "application/json" `
  -Body '{
    "vehicle_id": "NRM-FLEET-01",
    "gps_coordinates_at_action": {"lat": 40.409278, "lon": 49.867092}
  }'
```

## YOLOv8 Model

Set `SMARTWAVE_YOLO_MODEL_PATH` to a local Ultralytics YOLOv8 segmentation model
to use real inference. Without that environment variable the service uses a
local deterministic fallback so the API, audit trail, registry validation, and
decision rules can run during development.

## Module 2 Rules

The multi-report validation engine stores incident state in
`runtime/incident_reports.json` for local development and writes immutable
validation audit events to `runtime/audit_log.jsonl`.

Implemented verification paths:

- `ODOR_COMPLAINT`: 2+ reports, or Module 1 AI odor corroboration within 4 hours.
- `STRUCTURAL_DAMAGE`: 1 report plus worker confirmation.
- `PEST_INFESTATION`: 2+ reports or photo evidence.
- `FIRE_HAZARD`: immediate escalation.
- `ILLEGAL_DUMPING`: photo evidence or 3+ reports.
- `CONTAINER_MISSING`: 2+ reports within 6 hours.
- Anti-spam: 3 reports per citizen/container/24h and 5 reports per device/hour.

## Module 3 Forecasting

The predictive analytics engine stores local fill history in
`runtime/container_fill_history.json`, records visual-analysis fill observations
automatically, and writes prediction audit events to `runtime/audit_log.jsonl`.

Local development inputs:

- `container_fill_history`: JSON-backed rolling fill observations.
- `collection_log`: JSON-backed cleanup records used to ignore stale pre-cleaning history.
- `calendar_factors`: fixed Azerbaijan events plus optional dynamic holidays in `data/az_dynamic_holidays.json`.
- `weather_data`: seeded rain and humidity multipliers in `data/weather_data.json`.
- `zone_demographics`: density and commercial indices in `data/zone_demographics.json`.

Implemented forecast behavior:

- Uses the last 30 days of post-collection fill-rate data.
- Applies Azerbaijan holiday, weekend, market-day, weather, and zone multipliers.
- Emits an 80% confidence interval around `hours_to_red`.
- `<=36h` sets `SCHEDULE_NEXT_BATCH`; `<=12h` sets `URGENT_OVERRIDE`.
- Returns English and Azerbaijani citizen-facing forecast messages.
- Uses a deterministic Prophet-compatible baseline plus an XGBoost-style zone ensemble placeholder so production model adapters can replace the local MVP internals without changing the API.

## Module 4 Routing

The fleet optimizer uses seeded vehicles from `data/fleet_vehicles.json`, the
Narimanov depot default, latest fill history, verified incident counts, and
collection timestamps to generate route manifests. Route audit events are
written to `runtime/audit_log.jsonl`.

Implemented routing behavior:

- Supports `scheduled`, `event_driven`, and `manual` triggers.
- Event-driven optimization requires 3 or more RED containers in one route zone.
- Computes urgency with fullness, odor risk, hours since collection, validation reports, and calendar multiplier.
- Overrides fire hazard containers to urgency `999`.
- Clusters containers within 500m and orders each cluster with Clarke-Wright savings.
- Assigns work to vehicles by capacity and current location.
- Outputs GeoJSON `LineString` waypoints, duration, distance, fuel, and CO2 estimates.
- Includes the required minimum vehicle recommendation and SLA reasoning.

## Module 5 Worker Auth

The worker authentication module routes QR scans by role and protects worker
actions with local HS256 JWT validation. In production, the verifier is the seam
for myGov SSO/JWKS validation.

Implemented worker behavior:

- No token or `CITIZEN` token routes to `CITIZEN_FORM`.
- `WORKER` token routes to `WORKER_PANEL`.
- `SUPERVISOR` token routes to `SUPERVISOR_DASHBOARD`.
- Invalid or expired tokens return `401` and require myGov re-authentication.
- `MARK_CLEANED` resets fill history to `0`, records collection log, resolves open verified incidents, and emits `COLLECTION_COMPLETED`.
- Structural damage sets `MAINTENANCE_REQUIRED` and notifies infrastructure supervision.
- Critical safety hazards escalate to safety officer and emergency services.
- Worker task history returns the latest 30 worker actions.
- GPS actions outside 50m are held for supervisor review as `LOCATION_ANOMALY`.

Worker action records are stored in `runtime/worker_actions.json` and mirrored
to the immutable audit log.

## Test

```powershell
python -m pytest
```
