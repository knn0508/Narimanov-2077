from __future__ import annotations

import time
from uuid import uuid4

from smartwave_ai.visual_analysis.audit import (
    AuditLogger,
    build_audit_entry,
    sha256_hex,
    utc_now,
)
from smartwave_ai.visual_analysis.config import (
    AI_MODEL_VERSION,
    MIN_AI_CONFIDENCE,
    ORGANIC_CLASS_ALIASES,
    ORGANIC_TACO_CLASSES,
)
from smartwave_ai.visual_analysis.geometry import clamp_ratio, compute_fullness_score
from smartwave_ai.visual_analysis.inference import VisionDetection, VisionModel
from smartwave_ai.visual_analysis.models import (
    BoundingBox,
    StatusColor,
    TacoDetection,
    VisualAnalysisResponse,
)
from smartwave_ai.visual_analysis.registry import ContainerRegistry


class UnknownQrCodeError(Exception):
    error_code = "ERR_QR_UNREGISTERED"

    def __init__(
        self,
        container_id: str,
        qr_code_uuid: str | None,
        audit_entry_id: str,
    ) -> None:
        self.container_id = container_id
        self.qr_code_uuid = qr_code_uuid
        self.audit_entry_id = audit_entry_id
        super().__init__("Unknown or mismatched QR code.")


class EmptyImageError(ValueError):
    error_code = "ERR_IMAGE_EMPTY"


class VisionAnalysisError(RuntimeError):
    error_code = "ERR_IMAGE_ANALYSIS_FAILED"


def assign_status_color(fullness_score: int) -> StatusColor:
    if fullness_score <= 50:
        return StatusColor.GREEN
    if fullness_score <= 70:
        return StatusColor.YELLOW
    return StatusColor.RED


def normalize_taco_class(class_name: str) -> str:
    normalized = (
        class_name.strip()
        .lower()
        .replace("&", "and")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return ORGANIC_CLASS_ALIASES.get(normalized, normalized)


def is_organic_detection(detection: VisionDetection) -> bool:
    return normalize_taco_class(detection.class_name) in ORGANIC_TACO_CLASSES


def analyze_comment_with_gemini(
    user_comment: str,
    vision_data: dict[str, Any],
    gemini_model: Any,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    import json
    from PIL import Image
    import io
    
    contents = []
    if image_bytes:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            contents.append(pil_image)
        except Exception as img_err:
            print(f"Failed to load image for Gemini: {img_err}")

    prompt = (
        "Siz Bələdiyyənin Şikayət və Konteyner Analizi üzrə ekspertisiniz. Yalnız və yalnız JSON formatında cavab verməlisiniz. Heç bir əlavə izah, markdown və ya kod bloku yox.\n"
        "Təqdim olunan şəkil və Azərbaycan dilində yazılmış istifadəçi şikayəti (user_description) əsasında həm şikayətin uyğunluğunu, həm də konteynerin zədələnib-zədələnmədiyini (məsələn: sınmış qapaq, əyilmiş gövdə, qırılmış təkər, çatlamış hissə və s.) müəyyən edin.\n\n"
        
        "Şikayət uyğunluğu üçün (ai1_vision_data) məlumatları ilə müqayisə edin:\n"
        f"AI1_VISION_DATA:\n{json.dumps(vision_data, ensure_ascii=False)}\n\n"
        
        "QİYMƏTLƏNDİRMƏ ŞKALASI (text_urgency_score üçün 0-100 arası xal verin):\n"
        "- [0 - 20 xal]: Çox aşağı təcililik (Məsələn: Zibil qabı az doludur, ciddi qoxu və zədə yoxdur).\n"
        "- [21 - 50 xal]: Orta təcililik (Məsələn: Zibil qabı tam dolub, rutin boşaldılmalıdır və ya çox kiçik kosmetik zədə var).\n"
        "- [51 - 80 xal]: Yüksək təcililik (Məsələn: Zibillər ətrafa dağılıb, pis qoxu var və ya ciddi zədə var: sınıq qapaq, təkər qırılıb, zibil tökülür).\n"
        "- [81 - 100 xal]: Kritik / Təcili müdaxilə (Məsələn: Bioloji təhlükə, yanğın riski, konteyner tamamilə dağılıb/yanıb, yolu bağlayıb).\n\n"
        
        "Cavabda yalnız bu 7 açar olsun:\n"
        "1. consistency (boolean) - əgər istifadəçinin dedikləri ilə vizual modelin gördükləri üst-üstə düşürsə true, ziddiyyətlidirsə false.\n"
        "2. reason (Azərbaycan dilində qısa izah) - şikayət, zədə və ya qapağın bağlı olması haqqında qısa izahat.\n"
        "3. text_urgency_score (0-100 integer) - təcililik dərəcəsi.\n"
        "4. damage_detected (boolean) - əgər şəkildə və ya şikayətdə konteynerin zədələndiyi (sınıq, əzik, yararsız) təsdiqlənirsə true, əks halda false.\n"
        "5. damage_type (string və ya null) - zədənin növü (Azərbaycan dilində, məs: 'Sınıq qapaq', 'Əzilmiş gövdə', 'Qırıq təkər'), zədə yoxdursa null.\n"
        "6. damage_severity (string və ya null) - 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL', zədə yoxdursa null.\n"
        "7. container_closed (boolean) - əgər şəkildə konteynerin qapağı (qapaqları) tamamilə bağlıdırsa və daxilindəki zibili/doluluğu görmək mümkün deyilsə true, əks halda false.\n\n"
        
        f"USER_DESCRIPTION:\n{user_comment}\n\n"
        
        "Cavab nümunəsi: {\"consistency\": true, \"reason\": \"Konteyner zədəlidir (qapaq sınıb) və həmçinin qapağı bağlıdır.\", \"text_urgency_score\": 75, \"damage_detected\": true, \"damage_type\": \"Sınıq qapaq\", \"damage_severity\": \"HIGH\", \"container_closed\": true}\n"
    )
    contents.append(prompt)

    try:
        import google.generativeai as genai
        response = gemini_model.generate_content(
            contents=contents,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2
            ),
        )
        text_output = response.text.strip()
        
        if text_output.startswith("```json"):
            text_output = text_output[7:]
        if text_output.startswith("```"):
            text_output = text_output[3:]
        if text_output.endswith("```"):
            text_output = text_output[:-3]
        text_output = text_output.strip()

        parsed = json.loads(text_output)
        parsed["consistency"] = bool(parsed["consistency"])
        parsed["reason"] = str(parsed["reason"])
        score = int(parsed["text_urgency_score"])
        parsed["text_urgency_score"] = max(0, min(100, score))
        parsed["damage_detected"] = bool(parsed.get("damage_detected", False))
        parsed["container_closed"] = bool(parsed.get("container_closed", False))
        
        d_type = parsed.get("damage_type")
        parsed["damage_type"] = str(d_type) if d_type else None
        
        d_sev = parsed.get("damage_severity")
        parsed["damage_severity"] = str(d_sev) if d_sev else None
        
        return parsed
    except Exception as e:
        print(f"Gemini comment analysis failed: {e}")
        return {
            "consistency": True,
            "reason": f"Süni intellekt analizi uğursuz oldu: {str(e)}",
            "text_urgency_score": 0,
            "damage_detected": False,
            "damage_type": None,
            "damage_severity": None,
            "container_closed": False
        }



class VisualAnalysisService:
    def __init__(
        self,
        *,
        registry: ContainerRegistry,
        vision_model: VisionModel,
        audit_logger: AuditLogger,
    ) -> None:
        self.registry = registry
        self.vision_model = vision_model
        self.audit_logger = audit_logger
        self._latencies_ms: list[float] = []

    @property
    def inference_latency_p95_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        ordered = sorted(self._latencies_ms)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
        return round(ordered[index], 2)

    def process_report(
        self,
        *,
        container_id: str,
        image_bytes: bytes,
        qr_code_uuid: str | None = None,
        session_id: str | None = None,
        ip_address_hash: str | None = None,
        user_comment: str | None = None,
        gemini_model: Any | None = None,
    ) -> VisualAnalysisResponse:
        if not image_bytes:
            raise EmptyImageError("Image payload is required.")

        image_hash = sha256_hex(image_bytes)
        container = self.registry.resolve(container_id, qr_code_uuid=qr_code_uuid)
        if container is None:
            audit_entry = build_audit_entry(
                module="VISUAL_ANALYSIS",
                action="QR_REJECTED",
                input_hash=image_hash,
                output_summary="ERR_QR_UNREGISTERED: unknown or mismatched QR code",
                model_used=AI_MODEL_VERSION,
                confidence_score=0.0,
                human_reviewable=True,
                session_id=session_id,
                ip_address_hash=ip_address_hash,
                extra={
                    "error_code": UnknownQrCodeError.error_code,
                    "container_id": container_id,
                    "qr_code_uuid": qr_code_uuid,
                },
            )
            self.audit_logger.append(audit_entry)
            raise UnknownQrCodeError(
                container_id=container_id,
                qr_code_uuid=qr_code_uuid,
                audit_entry_id=str(audit_entry["audit_entry_id"]),
            )

        started = time.perf_counter()
        try:
            model_result = self.vision_model.analyze(image_bytes, container)
        except Exception as exc:
            raise VisionAnalysisError(str(exc)) from exc
        finally:
            self._latencies_ms.append((time.perf_counter() - started) * 1000.0)

        fullness_score = compute_fullness_score(
            container.container_geometry,
            model_result.fill_height_ratio,
            model_result.waste_mask_area_ratio,
        )
        confidence = round(clamp_ratio(model_result.confidence), 3)
        needs_human_review = confidence < MIN_AI_CONFIDENCE
        status_color = assign_status_color(fullness_score)

        odor_risk_flag = any(
            is_organic_detection(detection) for detection in model_result.detections
        ) and fullness_score >= 50
        if odor_risk_flag:
            status_color = StatusColor.RED

        taco_detections = [
            TacoDetection(
                class_name=detection.class_name,
                confidence=round(clamp_ratio(detection.confidence), 3),
                bbox_xyxy=BoundingBox(
                    x_min=detection.bbox_xyxy[0],
                    y_min=detection.bbox_xyxy[1],
                    x_max=detection.bbox_xyxy[2],
                    y_max=detection.bbox_xyxy[3],
                ),
            )
            for detection in model_result.detections
        ]

        audit_entry_id = str(uuid4())
        response = VisualAnalysisResponse(
            container_id=container.container_id,
            timestamp_utc=utc_now(),
            fullness_score=fullness_score,
            status_color=status_color,
            odor_risk_flag=odor_risk_flag,
            taco_classes_detected=taco_detections,
            confidence=confidence,
            needs_human_review=needs_human_review,
            image_hash_sha256=image_hash,
            ai_model_version=AI_MODEL_VERSION,
            audit_entry_id=audit_entry_id,
        )

        auto_dispatch_allowed = (
            status_color == StatusColor.RED and not needs_human_review
        )
        advisory = (
            "Organic waste detected. Accelerated collection advised."
            if odor_risk_flag
            else "No organic health-risk override."
        )
        audit_entry = build_audit_entry(
            audit_entry_id=audit_entry_id,
            module="VISUAL_ANALYSIS",
            action="VISUAL_REPORT_ANALYZED",
            input_hash=image_hash,
            output_summary=(
                f"{container.container_id}: fullness={fullness_score}%, "
                f"status={status_color.value}, human_review={needs_human_review}, "
                f"odor_risk={odor_risk_flag}"
            ),
            model_used=AI_MODEL_VERSION,
            confidence_score=confidence,
            human_reviewable=needs_human_review,
            session_id=session_id,
            ip_address_hash=ip_address_hash,
            extra={
                "container_snapshot": container.model_dump(mode="json"),
                "taco_detections": [
                    detection.model_dump(mode="json") for detection in taco_detections
                ],
                "auto_dispatch_allowed": auto_dispatch_allowed,
                "advisory": advisory,
                "vision_runtime": self.vision_model.model_name,
            },
        )
        self.audit_logger.append(audit_entry)

        comment_analysis = None
        damage_detected = False
        damage_type = None
        damage_severity = None
        container_closed = False

        if user_comment and gemini_model:
            vision_data = {
                "container_id": response.container_id,
                "fullness_score": response.fullness_score,
                "status_color": response.status_color,
                "odor_risk_flag": response.odor_risk_flag,
                "taco_classes_detected": [
                    {
                        "class_name": det.class_name,
                        "confidence": det.confidence
                    }
                    for det in response.taco_classes_detected
                ]
            }
            comment_analysis_dict = analyze_comment_with_gemini(
                user_comment=user_comment,
                vision_data=vision_data,
                gemini_model=gemini_model,
                image_bytes=image_bytes,
            )
            from smartwave_ai.visual_analysis.models import CommentAnalysisResult
            comment_analysis = CommentAnalysisResult(
                consistency=comment_analysis_dict["consistency"],
                reason=comment_analysis_dict["reason"],
                text_urgency_score=comment_analysis_dict["text_urgency_score"],
                damage_detected=comment_analysis_dict["damage_detected"],
                damage_type=comment_analysis_dict["damage_type"],
                damage_severity=comment_analysis_dict["damage_severity"],
                container_closed=comment_analysis_dict["container_closed"],
            )
            damage_detected = comment_analysis_dict["damage_detected"]
            damage_type = comment_analysis_dict["damage_type"]
            damage_severity = comment_analysis_dict["damage_severity"]
            container_closed = comment_analysis_dict["container_closed"]

        response.comment_analysis = comment_analysis
        response.damage_detected = damage_detected
        response.damage_type = damage_type
        response.damage_severity = damage_severity
        response.container_closed = container_closed

        if container_closed:
            response.fullness_score = 0
            response.status_color = assign_status_color(0)

        if damage_detected:
            response.taco_classes_detected.append(
                TacoDetection(
                    class_name="damaged",
                    confidence=0.92,
                    bbox_xyxy=BoundingBox(
                        x_min=10.0,
                        y_min=20.0,
                        x_max=85.0,
                        y_max=90.0,
                    )
                )
            )

        return response

