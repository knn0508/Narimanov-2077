from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from smartwave_ai.visual_analysis.models import ContainerRecord


@dataclass(frozen=True)
class VisionDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class VisionModelResult:
    fill_height_ratio: float | None
    waste_mask_area_ratio: float | None
    confidence: float
    detections: tuple[VisionDetection, ...]


class VisionModel(Protocol):
    model_name: str

    def analyze(
        self, image_bytes: bytes, container: ContainerRecord
    ) -> VisionModelResult:
        ...


class HeuristicVisionModel:
    """Development fallback used when YOLOv8 weights are not available locally."""

    model_name = "heuristic-local-fallback"

    def analyze(
        self, image_bytes: bytes, container: ContainerRecord
    ) -> VisionModelResult:
        with Image.open(io.BytesIO(image_bytes)) as image:
            gray = image.convert("L").resize((96, 96))

        pixels = np.asarray(gray, dtype=np.float32)
        mean = float(pixels.mean())
        std = float(pixels.std())
        threshold = min(215.0, mean + (std * 0.15))
        likely_waste = pixels < threshold

        row_density = likely_waste.mean(axis=1)
        active_rows = np.where(row_density > 0.28)[0]
        if active_rows.size:
            fill_height_ratio = (pixels.shape[0] - int(active_rows[0])) / pixels.shape[0]
        else:
            fill_height_ratio = 0.0

        waste_mask_area_ratio = float(likely_waste.mean())
        contrast_component = min(std / 85.0, 1.0) * 0.35
        confidence = max(0.35, min(0.88, 0.45 + contrast_component))

        detections: list[VisionDetection] = []
        container_type = container.container_type.lower() if container.container_type else ""
        if container_type == "plastic" or "pla" in container.container_id.lower():
            detections = [
                VisionDetection(
                    class_name="plastic_bottle",
                    confidence=0.88,
                    bbox_xyxy=(10.0, 20.0, 80.0, 90.0),
                ),
                VisionDetection(
                    class_name="plastic_bag",
                    confidence=0.82,
                    bbox_xyxy=(15.0, 25.0, 75.0, 85.0),
                ),
            ]
        else:
            # mixed container fallback
            detections = [
                VisionDetection(
                    class_name="trash",
                    confidence=0.90,
                    bbox_xyxy=(10.0, 20.0, 85.0, 90.0),
                ),
                VisionDetection(
                    class_name="food_waste",
                    confidence=0.75,
                    bbox_xyxy=(20.0, 30.0, 70.0, 80.0),
                ),
            ]

        return VisionModelResult(
            fill_height_ratio=fill_height_ratio,
            waste_mask_area_ratio=waste_mask_area_ratio,
            confidence=confidence,
            detections=tuple(detections),
        )


class UltralyticsYoloV8Model:
    """Wraps a single Ultralytics YOLOv8 segmentation model for one container category."""

    def __init__(self, model_or_path: str | Any, model_label: str = "yolov8-seg") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required when a YOLO model path is set"
            ) from exc

        if isinstance(model_or_path, str):
            self._model = YOLO(model_or_path)
        else:
            self._model = model_or_path

        self.model_name = f"ultralytics-{model_label}"
        self._secondary_model = None
        device = os.getenv("SMARTWAVE_DEVICE", "cpu")
        try:
            self._model.to(device)
            print(f"[{self.model_name}] loaded on device: {device}")
        except Exception as exc:
            print(f"[{self.model_name}] could not move to {device}: {exc}")

    def analyze(
        self, image_bytes: bytes, container: ContainerRecord
    ) -> VisionModelResult:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")

        # Run primary model with a lower confidence threshold for better recall on complex trash
        result1 = self._model.predict(rgb_image, conf=0.15, verbose=False)[0]

        # Run secondary model if loaded
        result2 = None
        if self._secondary_model is not None:
            try:
                result2 = self._secondary_model.predict(rgb_image, conf=0.15, verbose=False)[0]
            except Exception as e:
                print(f"Secondary model inference failed: {e}")

        detections: list[VisionDetection] = []
        confidence_values: list[float] = []
        filtered_masks = []

        from smartwave_ai.visual_analysis.config import TRASH_CLASSES
        from smartwave_ai.visual_analysis.service import normalize_taco_class

        def process_result(result):
            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                masks = None
                if result.masks is not None and result.masks.data is not None:
                    masks = result.masks.data.cpu().numpy().astype(bool)

                for idx, (bbox, class_id, confidence) in enumerate(zip(boxes_xyxy, class_ids, confidences)):
                    class_name = str(result.names.get(int(class_id), class_id))
                    confidence_float = float(confidence)
                    confidence_values.append(confidence_float)

                    normalized_name = normalize_taco_class(class_name)
                    detections.append(
                        VisionDetection(
                            class_name=class_name,
                            confidence=confidence_float,
                            bbox_xyxy=tuple(float(value) for value in bbox),
                        )
                    )

                    if masks is not None and normalized_name in TRASH_CLASSES:
                        # Filter out tiny noise masks (<0.5% of total image area) for better accuracy
                        if masks[idx].mean() >= 0.005:
                            filtered_masks.append(masks[idx])

        # Process both model results
        process_result(result1)
        if result2 is not None:
            process_result(result2)

        fill_height_ratio: float | None = None
        waste_mask_area_ratio: float | None = None

        if filtered_masks:
            first_shape = filtered_masks[0].shape
            valid_masks = []
            for mask in filtered_masks:
                if mask.shape == first_shape:
                    valid_masks.append(mask)
                else:
                    from PIL import Image as PILImage
                    pil_mask = PILImage.fromarray((mask * 255).astype(np.uint8))
                    resized_mask = np.array(pil_mask.resize((first_shape[1], first_shape[0]), PILImage.NEAREST)) > 0
                    valid_masks.append(resized_mask)

            if valid_masks:
                combined_mask = np.any(valid_masks, axis=0)
                waste_mask_area_ratio = float(combined_mask.mean())
                row_density = combined_mask.mean(axis=1)
                # Use a slightly higher threshold (8%) to avoid noise at the container top
                active_rows = np.where(row_density > 0.08)[0]
                if active_rows.size:
                    fill_height_ratio = (
                        combined_mask.shape[0] - int(active_rows[0])
                    ) / combined_mask.shape[0]

        confidence = (
            float(np.mean(confidence_values)) if confidence_values else 0.0
        )
        return VisionModelResult(
            fill_height_ratio=fill_height_ratio,
            waste_mask_area_ratio=waste_mask_area_ratio,
            confidence=confidence,
            detections=tuple(detections),
        )


def create_vision_model(
    model_or_path: Any | None = None,
    container_type: str | None = None,
) -> VisionModel:
    """Return the correct VisionModel for the given container category.

    - mixed   → SMARTWAVE_YOLO_MODEL_MIXED  (yolov8m-seg.pt)
    - plastic → SMARTWAVE_YOLO_MODEL_PLASTIC (yolov8n-seg.pt)
    Falls back to HeuristicVisionModel when no weight file is available.
    """
    if model_or_path is not None:
        # Caller already resolved a model object / path (e.g. preloaded in lifespan)
        return UltralyticsYoloV8Model(model_or_path)

    ctype = (container_type or "").lower()

    if ctype == "plastic":
        path = os.getenv("SMARTWAVE_YOLO_MODEL_PLASTIC")
        label = "yolov8n-seg-plastic"
    else:
        # default: mixed
        path = os.getenv("SMARTWAVE_YOLO_MODEL_MIXED")
        label = "yolov8m-seg-mixed"

    if path and os.path.exists(path):
        return UltralyticsYoloV8Model(path, model_label=label)

    print(
        f"[create_vision_model] No weight file found for type='{ctype}'. "
        "Using HeuristicVisionModel fallback."
    )
    return HeuristicVisionModel()

