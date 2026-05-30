from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from smartwave_ai.fleet_route_optimization.api import build_default_fleet_service
from smartwave_ai.fleet_route_optimization.models import (
    RouteManifest,
    RouteOptimizationErrorResponse,
    RouteOptimizationRequest,
)
from smartwave_ai.fleet_route_optimization.service import (
    FleetRouteOptimizationService,
    NoServiceContainersError,
    NoVehiclesAvailableError,
    RouteOptimizationError,
)
from smartwave_ai.multi_report_validation.api import build_default_validation_service
from smartwave_ai.multi_report_validation.models import (
    CitizenComplaintRequest,
    ValidationDecision,
    ValidationErrorResponse,
    WorkerConfirmationRequest,
)
from smartwave_ai.multi_report_validation.service import (
    IncidentNotFoundError,
    InvalidWorkerConfirmationError,
    MultiReportValidationService,
    UnknownContainerError,
)
from smartwave_ai.predictive_analytics.api import build_default_predictive_service
from smartwave_ai.predictive_analytics.models import (
    ContainerForecastResponse,
    ForecastErrorResponse,
)
from smartwave_ai.predictive_analytics.service import (
    ForecastUnavailableError,
    PredictiveAnalyticsService,
    UnknownForecastContainerError,
)
from smartwave_ai.visual_analysis.audit import AuditLogger, hash_text
from smartwave_ai.visual_analysis.config import (
    AI_MODEL_VERSION,
    DEFAULT_AUDIT_LOG_PATH,
    DEFAULT_REGISTRY_PATH,
    INFERENCE_LATENCY_P95_TARGET_MS,
)
from smartwave_ai.visual_analysis.inference import create_vision_model
from smartwave_ai.visual_analysis.models import ApiError, VisualAnalysisResponse
from smartwave_ai.visual_analysis.registry import ContainerRegistry
from smartwave_ai.visual_analysis.service import (
    EmptyImageError,
    UnknownQrCodeError,
    VisionAnalysisError,
    VisualAnalysisService,
)
from smartwave_ai.worker_authentication.api import build_default_worker_auth_service
from smartwave_ai.worker_authentication.jwt import JwtAuthError, JwtExpiredError
from smartwave_ai.worker_authentication.models import (
    MarkCleanedRequest,
    SafetyHazardRequest,
    SessionRouteResponse,
    StructuralDamageRequest,
    WorkerActionResponse,
    WorkerAuthErrorResponse,
    WorkerTaskHistoryResponse,
)
from smartwave_ai.worker_authentication.service import (
    AuthRequiredError,
    ForbiddenRoleError,
    UnknownWorkerContainerError,
    WorkerAuthenticationService,
)


def build_default_visual_analysis_service(
    registry: ContainerRegistry | None = None,
    audit_logger: AuditLogger | None = None,
) -> VisualAnalysisService:
    return VisualAnalysisService(
        registry=registry or ContainerRegistry.from_json(DEFAULT_REGISTRY_PATH),
        vision_model=create_vision_model(),
        audit_logger=audit_logger or AuditLogger(DEFAULT_AUDIT_LOG_PATH),
    )


def create_app(
    service: VisualAnalysisService | None = None,
    validation_service: MultiReportValidationService | None = None,
    predictive_service: PredictiveAnalyticsService | None = None,
    fleet_service: FleetRouteOptimizationService | None = None,
    worker_auth_service: WorkerAuthenticationService | None = None,
) -> FastAPI:
    from smartwave_ai.database import initialize_db

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Initialize tables and seed registry
        initialize_db()

        # 2. Preload YOLOv8
        model_path = os.getenv("SMARTWAVE_YOLO_MODEL_PATH")
        if model_path:
            try:
                from ultralytics import YOLO
                app.state.yolo_model = YOLO(model_path)
                device = os.getenv("SMARTWAVE_DEVICE", "cuda")
                app.state.yolo_model.to(device)
                print(f"YOLOv8 Model loaded successfully on device: {device}")
                # Re-create visual analysis service using preloaded instance
                visual_service = app.state.visual_analysis_service
                from smartwave_ai.visual_analysis.inference import create_vision_model
                visual_service.vision_model = create_vision_model(app.state.yolo_model)
            except Exception as e:
                print(f"Failed to preload YOLOv8 model: {e}")

        # 3. Preload Gemini for Comment AI
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                available_models = [
                    m.name for m in genai.list_models() 
                    if 'generateContent' in m.supported_generation_methods
                ]
                target_model = next((m for m in available_models if "flash" in m), None)
                if not target_model:
                    target_model = next((m for m in available_models if "pro" in m), "gemini-pro")
                
                app.state.gemini_model = genai.GenerativeModel(target_model)
                print(f"Google Gemini Model loaded successfully: {target_model}")
            except Exception as e:
                print(f"Failed to preload Gemini model: {e}")

        yield
        if hasattr(app.state, "yolo_model"):
            del app.state.yolo_model
        if hasattr(app.state, "gemini_model"):
            del app.state.gemini_model

    app = FastAPI(
        title="SmartWave AI Engine",
        version="1.0.0",
        lifespan=lifespan,
    )

    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    shared_registry = ContainerRegistry.from_json(DEFAULT_REGISTRY_PATH)
    shared_audit_logger = AuditLogger(DEFAULT_AUDIT_LOG_PATH)
    app.state.visual_analysis_service = service or build_default_visual_analysis_service(
        registry=shared_registry,
        audit_logger=shared_audit_logger,
    )
    app.state.validation_service = validation_service or build_default_validation_service(
        registry=shared_registry,
        audit_logger=shared_audit_logger,
    )
    app.state.predictive_service = predictive_service or build_default_predictive_service(
        registry=shared_registry,
        audit_logger=shared_audit_logger,
    )
    app.state.fleet_service = fleet_service or build_default_fleet_service(
        registry=shared_registry,
        predictive_service=app.state.predictive_service,
        validation_service=app.state.validation_service,
        audit_logger=shared_audit_logger,
    )
    app.state.worker_auth_service = worker_auth_service or build_default_worker_auth_service(
        registry=shared_registry,
        validation_service=app.state.validation_service,
        predictive_service=app.state.predictive_service,
        audit_logger=shared_audit_logger,
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        visual_service: VisualAnalysisService = app.state.visual_analysis_service
        validation_engine: MultiReportValidationService = app.state.validation_service
        predictive_engine: PredictiveAnalyticsService = app.state.predictive_service
        fleet_engine: FleetRouteOptimizationService = app.state.fleet_service
        worker_engine: WorkerAuthenticationService = app.state.worker_auth_service
        return {
            "status": "ok",
            "modules": [
                "VISUAL_ANALYSIS",
                "VALIDATION",
                "PREDICTION",
                "ROUTING",
                "WORKER_ACTION",
            ],
            "ai_model_version": AI_MODEL_VERSION,
            "metrics": {
                "inference_latency_p95_ms": visual_service.inference_latency_p95_ms,
                "inference_latency_p95_target_ms": INFERENCE_LATENCY_P95_TARGET_MS,
                "incident_reports_tracked": len(validation_engine.incidents.all()),
                "operations_events_buffered": len(validation_engine.operations_events),
                "fill_history_points_tracked": len(predictive_engine.fill_history.all()),
                "predictive_dispatch_events_buffered": len(
                    predictive_engine.dispatch_events
                ),
                "route_manifests_generated": len(fleet_engine.generated_manifests),
                "worker_actions_logged": len(worker_engine.worker_actions.all()),
            },
        }

    def auth_error_response(exc: Exception) -> JSONResponse:
        if isinstance(exc, JwtExpiredError):
            error_code = exc.error_code
            message = "Authentication token expired. Re-authenticate via myGov SSO."
        elif isinstance(exc, (JwtAuthError, AuthRequiredError)):
            error_code = getattr(exc, "error_code", "ERR_AUTH_INVALID")
            message = "Invalid authentication token. Re-authenticate via myGov SSO."
        else:
            error_code = "ERR_AUTH_INVALID"
            message = "Invalid authentication token. Re-authenticate via myGov SSO."
        error = WorkerAuthErrorResponse(error_code=error_code, message=message)
        return JSONResponse(status_code=401, content=error.model_dump(mode="json"))

    def forbidden_response(exc: Exception) -> JSONResponse:
        error = WorkerAuthErrorResponse(
            error_code=getattr(exc, "error_code", "ERR_ROLE_FORBIDDEN"),
            message="Worker or supervisor role required.",
        )
        return JSONResponse(status_code=403, content=error.model_dump(mode="json"))

    @app.post(
        "/api/v1/containers/{container_id}/report",
        response_model=VisualAnalysisResponse,
    )
    async def submit_container_report(
        container_id: str,
        request: Request,
        comment: str | None = None,
        x_qr_code_uuid: Annotated[
            str | None, Header(alias="X-QR-Code-UUID")
        ] = None,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        image_bytes = await request.body()
        client_host = request.client.host if request.client else None
        ip_address_hash = hash_text(client_host)
        visual_service: VisualAnalysisService = app.state.visual_analysis_service
        validation_engine: MultiReportValidationService = app.state.validation_service
        predictive_engine: PredictiveAnalyticsService = app.state.predictive_service

        try:
            gemini_model = getattr(request.app.state, "gemini_model", None)
            response = visual_service.process_report(
                container_id=container_id,
                qr_code_uuid=x_qr_code_uuid,
                image_bytes=image_bytes,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
                user_comment=comment,
                gemini_model=gemini_model,
            )
            predictive_engine.record_visual_fill(
                container_id=response.container_id,
                timestamp_utc=response.timestamp_utc,
                fullness_score=response.fullness_score,
                confidence=response.confidence,
                audit_entry_id=response.audit_entry_id,
            )
            validation_engine.apply_ai_odor_corroboration(
                container_id=container_id,
                odor_risk_flag=response.odor_risk_flag,
                visual_audit_entry_id=response.audit_entry_id,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
            )
        except UnknownQrCodeError as exc:
            error = ApiError(
                error_code=exc.error_code,
                message="Unknown QR code. Container report rejected.",
                audit_entry_id=exc.audit_entry_id,
                details={
                    "container_id": exc.container_id,
                    "qr_code_uuid": exc.qr_code_uuid,
                },
            )
            return JSONResponse(
                status_code=404,
                content=error.model_dump(mode="json"),
            )
        except EmptyImageError as exc:
            error = ApiError(
                error_code=exc.error_code,
                message="Image payload is required.",
            )
            return JSONResponse(
                status_code=400,
                content=error.model_dump(mode="json"),
            )
        except VisionAnalysisError as exc:
            error = ApiError(
                error_code=exc.error_code,
                message="Image could not be analyzed.",
                details={"reason": str(exc)},
            )
            return JSONResponse(
                status_code=422,
                content=error.model_dump(mode="json"),
            )

        return response.model_dump(mode="json")

    @app.get(
        "/api/v1/containers/{container_id}/panel-route",
        response_model=SessionRouteResponse,
    )
    async def detect_panel_route(
        container_id: str,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> JSONResponse | dict[str, object]:
        worker_engine: WorkerAuthenticationService = app.state.worker_auth_service
        try:
            route = worker_engine.session_route(
                container_id=container_id,
                authorization_header=authorization,
            )
        except UnknownWorkerContainerError as exc:
            error = WorkerAuthErrorResponse(
                error_code=exc.error_code,
                message="Unknown container.",
                details={"container_id": exc.container_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))
        except (JwtAuthError, JwtExpiredError, AuthRequiredError) as exc:
            return auth_error_response(exc)
        return route.model_dump(mode="json")

    @app.post(
        "/api/v1/worker/containers/{container_id}/mark-cleaned",
        response_model=WorkerActionResponse,
    )
    async def mark_container_cleaned(
        container_id: str,
        request: Request,
        payload: MarkCleanedRequest,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        worker_engine: WorkerAuthenticationService = app.state.worker_auth_service
        client_host = request.client.host if request.client else None
        try:
            claims = worker_engine.parse_authorization(authorization)
            response = worker_engine.mark_cleaned(
                container_id=container_id,
                request=payload,
                claims=claims,
                session_id=x_session_id,
                ip_address_hash=hash_text(client_host),
            )
        except (JwtAuthError, JwtExpiredError, AuthRequiredError) as exc:
            return auth_error_response(exc)
        except ForbiddenRoleError as exc:
            return forbidden_response(exc)
        except UnknownWorkerContainerError as exc:
            error = WorkerAuthErrorResponse(
                error_code=exc.error_code,
                message="Unknown container.",
                details={"container_id": exc.container_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))
        return response.model_dump(mode="json")

    @app.post(
        "/api/v1/worker/containers/{container_id}/structural-damage",
        response_model=WorkerActionResponse,
    )
    async def report_worker_structural_damage(
        container_id: str,
        request: Request,
        payload: StructuralDamageRequest,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        worker_engine: WorkerAuthenticationService = app.state.worker_auth_service
        client_host = request.client.host if request.client else None
        try:
            claims = worker_engine.parse_authorization(authorization)
            response = worker_engine.report_structural_damage(
                container_id=container_id,
                request=payload,
                claims=claims,
                session_id=x_session_id,
                ip_address_hash=hash_text(client_host),
            )
        except (JwtAuthError, JwtExpiredError, AuthRequiredError) as exc:
            return auth_error_response(exc)
        except ForbiddenRoleError as exc:
            return forbidden_response(exc)
        except UnknownWorkerContainerError as exc:
            error = WorkerAuthErrorResponse(
                error_code=exc.error_code,
                message="Unknown container.",
                details={"container_id": exc.container_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))
        return response.model_dump(mode="json")

    @app.post(
        "/api/v1/worker/containers/{container_id}/safety-hazard",
        response_model=WorkerActionResponse,
    )
    async def report_worker_safety_hazard(
        container_id: str,
        request: Request,
        payload: SafetyHazardRequest,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        worker_engine: WorkerAuthenticationService = app.state.worker_auth_service
        client_host = request.client.host if request.client else None
        try:
            claims = worker_engine.parse_authorization(authorization)
            response = worker_engine.report_safety_hazard(
                container_id=container_id,
                request=payload,
                claims=claims,
                session_id=x_session_id,
                ip_address_hash=hash_text(client_host),
            )
        except (JwtAuthError, JwtExpiredError, AuthRequiredError) as exc:
            return auth_error_response(exc)
        except ForbiddenRoleError as exc:
            return forbidden_response(exc)
        except UnknownWorkerContainerError as exc:
            error = WorkerAuthErrorResponse(
                error_code=exc.error_code,
                message="Unknown container.",
                details={"container_id": exc.container_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))
        return response.model_dump(mode="json")

    @app.get(
        "/api/v1/worker/task-history",
        response_model=WorkerTaskHistoryResponse,
    )
    async def worker_task_history(
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
        limit: int = 30,
    ) -> JSONResponse | dict[str, object]:
        worker_engine: WorkerAuthenticationService = app.state.worker_auth_service
        try:
            claims = worker_engine.parse_authorization(authorization)
            history = worker_engine.task_history(claims=claims, limit=limit)
        except (JwtAuthError, JwtExpiredError, AuthRequiredError) as exc:
            return auth_error_response(exc)
        except ForbiddenRoleError as exc:
            return forbidden_response(exc)
        return history.model_dump(mode="json")

    @app.get(
        "/api/v1/containers/{container_id}/forecast",
        response_model=ContainerForecastResponse,
    )
    async def forecast_container_red_status(
        container_id: str,
        request: Request,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        client_host = request.client.host if request.client else None
        ip_address_hash = hash_text(client_host)
        predictive_engine: PredictiveAnalyticsService = app.state.predictive_service

        try:
            forecast = predictive_engine.forecast_container(
                container_id=container_id,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
            )
        except UnknownForecastContainerError as exc:
            error = ForecastErrorResponse(
                error_code=exc.error_code,
                message="Unknown container. Forecast rejected.",
                details={"container_id": exc.container_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))
        except ForecastUnavailableError as exc:
            error = ForecastErrorResponse(
                error_code=exc.error_code,
                message=str(exc),
                details={"container_id": container_id},
            )
            return JSONResponse(status_code=422, content=error.model_dump(mode="json"))

        return forecast.model_dump(mode="json")

    @app.post(
        "/api/v1/routes/optimize",
        response_model=RouteManifest,
    )
    async def optimize_fleet_routes(
        request: Request,
        optimization_request: RouteOptimizationRequest,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
        x_operator_id: Annotated[str | None, Header(alias="X-Operator-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        client_host = request.client.host if request.client else None
        ip_address_hash = hash_text(client_host)
        fleet_engine: FleetRouteOptimizationService = app.state.fleet_service

        try:
            manifest = fleet_engine.optimize_routes(
                request=optimization_request,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
                operator_id=x_operator_id,
            )
        except NoServiceContainersError as exc:
            error = RouteOptimizationErrorResponse(
                error_code=exc.error_code,
                message=str(exc),
            )
            return JSONResponse(status_code=422, content=error.model_dump(mode="json"))
        except NoVehiclesAvailableError as exc:
            error = RouteOptimizationErrorResponse(
                error_code=exc.error_code,
                message=str(exc),
            )
            return JSONResponse(status_code=422, content=error.model_dump(mode="json"))
        except RouteOptimizationError as exc:
            error = RouteOptimizationErrorResponse(
                error_code=exc.error_code,
                message=str(exc),
            )
            return JSONResponse(status_code=422, content=error.model_dump(mode="json"))

        return manifest.model_dump(mode="json")

    @app.post(
        "/api/v1/containers/{container_id}/complaints",
        response_model=ValidationDecision,
    )
    async def submit_citizen_complaint(
        container_id: str,
        request: Request,
        complaint: CitizenComplaintRequest,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        client_host = request.client.host if request.client else None
        ip_address_hash = hash_text(client_host)
        validation_engine: MultiReportValidationService = app.state.validation_service

        try:
            decision = validation_engine.submit_complaint(
                container_id=container_id,
                request=complaint,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
            )
        except UnknownContainerError as exc:
            error = ValidationErrorResponse(
                error_code=exc.error_code,
                message="Unknown container. Complaint rejected.",
                details={"container_id": exc.container_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

        return decision.model_dump(mode="json")

    @app.post(
        "/api/v1/incidents/{report_id}/worker-confirmation",
        response_model=ValidationDecision,
    )
    async def confirm_incident_by_worker(
        report_id: str,
        request: Request,
        confirmation: WorkerConfirmationRequest,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        client_host = request.client.host if request.client else None
        ip_address_hash = hash_text(client_host)
        validation_engine: MultiReportValidationService = app.state.validation_service

        try:
            decision = validation_engine.confirm_worker_report(
                report_id=report_id,
                request=confirmation,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
            )
        except IncidentNotFoundError as exc:
            error = ValidationErrorResponse(
                error_code=exc.error_code,
                message="Incident report not found.",
                details={"report_id": exc.report_id},
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))
        except InvalidWorkerConfirmationError as exc:
            error = ValidationErrorResponse(
                error_code=exc.error_code,
                message=str(exc),
            )
            return JSONResponse(status_code=409, content=error.model_dump(mode="json"))

        return decision.model_dump(mode="json")

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        static_file_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
        if not static_file_path.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Dashboard index.html not found")
        with open(static_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    @app.get("/api/v1/dev/token")
    async def get_dev_token(role: str = "WORKER", subject: str = "worker-1"):
        import time
        from smartwave_ai.worker_authentication.jwt import create_dev_jwt
        payload = {
            "sub": subject,
            "worker_id": subject,
            "role": role.upper(),
            "exp": int(time.time()) + 3600
        }
        token = create_dev_jwt(payload)
        return {"token": token}

    @app.get("/api/v1/dev/ledger")
    async def get_dev_ledger():
        from smartwave_ai.database import SessionLocal, DbAuditLedger
        db = SessionLocal()
        try:
            records = db.query(DbAuditLedger).order_by(DbAuditLedger.timestamp_utc.desc()).all()
            result = []
            for r in records:
                result.append({
                    "audit_entry_id": r.audit_entry_id,
                    "module": r.module,
                    "action": r.action,
                    "input_hash": r.input_hash,
                    "output_summary": r.output_summary,
                    "model_used": r.model_used,
                    "confidence_score": r.confidence_score,
                    "human_reviewable": r.human_reviewable,
                    "operator_id": r.operator_id,
                    "session_id": r.session_id,
                    "ip_address_hash": r.ip_address_hash,
                    "timestamp_utc": r.timestamp_utc.isoformat() if r.timestamp_utc else None,
                    "extra": r.extra,
                    "previous_row_hash": r.previous_row_hash,
                    "row_hash": r.row_hash
                })
            return {"ledger": result}
        finally:
            db.close()

    return app
