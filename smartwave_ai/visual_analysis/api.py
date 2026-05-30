from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

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
    UnknownContainerError,
    VisionAnalysisError,
    VisualAnalysisService,
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
) -> FastAPI:
    from smartwave_ai.database import initialize_db

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. Initialize tables and seed registry
        initialize_db()

        # 2. Preload per-category YOLOv8 models
        from smartwave_ai.visual_analysis.inference import UltralyticsYoloV8Model
        device = os.getenv("SMARTWAVE_DEVICE", "cpu")

        mixed_path = os.getenv("SMARTWAVE_YOLO_MODEL_MIXED")
        if mixed_path and os.path.exists(mixed_path):
            try:
                app.state.yolo_model_mixed = UltralyticsYoloV8Model(
                    mixed_path, model_label="yolov8m-seg-mixed"
                )
                print(f"[mixed] YOLOv8 model loaded from {mixed_path}")
            except Exception as exc:
                print(f"[mixed] Failed to load model: {exc}")

        plastic_path = os.getenv("SMARTWAVE_YOLO_MODEL_PLASTIC")
        if plastic_path and os.path.exists(plastic_path):
            try:
                app.state.yolo_model_plastic = UltralyticsYoloV8Model(
                    plastic_path, model_label="yolov8n-seg-plastic"
                )
                print(f"[plastic] YOLOv8 model loaded from {plastic_path}")
            except Exception as exc:
                print(f"[plastic] Failed to load model: {exc}")

        # 3. Preload Gemini for Comment AI
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                available_models = [
                    m.name for m in genai.list_models()
                    if "generateContent" in m.supported_generation_methods
                ]
                target_model = next((m for m in available_models if "flash" in m), None)
                if not target_model:
                    target_model = next((m for m in available_models if "pro" in m), "gemini-pro")
                app.state.gemini_model = genai.GenerativeModel(target_model)
                print(f"Google Gemini Model loaded successfully: {target_model}")
            except Exception as exc:
                print(f"Failed to preload Gemini model: {exc}")

        yield
        for attr in ("yolo_model_mixed", "yolo_model_plastic", "gemini_model"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)

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

    @app.get("/health")
    def health() -> dict[str, object]:
        visual_service: VisualAnalysisService = app.state.visual_analysis_service
        return {
            "status": "ok",
            "modules": [
                "VISUAL_ANALYSIS",
            ],
            "ai_model_version": AI_MODEL_VERSION,
            "metrics": {
                "inference_latency_p95_ms": visual_service.inference_latency_p95_ms,
                "inference_latency_p95_target_ms": INFERENCE_LATENCY_P95_TARGET_MS,
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

    @app.post(
        "/api/v1/containers/{container_id}/report",
        response_model=VisualAnalysisResponse,
    )
    async def submit_container_report(
        container_id: str,
        request: Request,
        comment: str | None = None,
        x_session_id: Annotated[str | None, Header(alias="X-Session-ID")] = None,
    ) -> JSONResponse | dict[str, object]:
        image_bytes = await request.body()
        client_host = request.client.host if request.client else None
        ip_address_hash = hash_text(client_host)
        visual_service: VisualAnalysisService = app.state.visual_analysis_service

        # Select the per-category YOLO model based on the container type from the registry
        from smartwave_ai.visual_analysis.registry import ContainerRegistry
        from smartwave_ai.visual_analysis.inference import create_vision_model
        registry: ContainerRegistry = app.state.visual_analysis_service.registry
        try:
            container_record = registry.resolve(container_id)
            ctype = container_record.container_type.lower() if container_record else "mixed"
        except Exception:
            ctype = "mixed"

        if ctype == "plastic" and hasattr(app.state, "yolo_model_plastic"):
            visual_service.vision_model = app.state.yolo_model_plastic
        elif ctype != "plastic" and hasattr(app.state, "yolo_model_mixed"):
            visual_service.vision_model = app.state.yolo_model_mixed
        else:
            # Fallback: create on-the-fly (uses HeuristicVisionModel if no weights)
            visual_service.vision_model = create_vision_model(container_type=ctype)

        try:
            gemini_model = getattr(request.app.state, "gemini_model", None)
            response = visual_service.process_report(
                container_id=container_id,
                image_bytes=image_bytes,
                session_id=x_session_id,
                ip_address_hash=ip_address_hash,
                user_comment=comment,
                gemini_model=gemini_model,
            )
        except UnknownContainerError as exc:
            error = ApiError(
                error_code=exc.error_code,
                message="Unknown container. Container report rejected.",
                audit_entry_id=exc.audit_entry_id,
                details={
                    "container_id": exc.container_id,
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

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        static_file_path = Path(__file__).resolve().parent.parent / "static" / "index.html"
        if not static_file_path.exists():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Dashboard index.html not found")
        with open(static_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

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
