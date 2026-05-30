from __future__ import annotations

import math

from smartwave_ai.visual_analysis.models import ContainerGeometry


def clamp_ratio(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def cylindrical_segment_volume_ratio(fill_height_ratio: float) -> float:
    height = clamp_ratio(fill_height_ratio)
    if height <= 0.0:
        return 0.0
    if height >= 1.0:
        return 1.0

    normalized_radius_offset = 1.0 - (2.0 * height)
    segment_area = math.acos(normalized_radius_offset) - (
        normalized_radius_offset
        * math.sqrt(max(0.0, 1.0 - normalized_radius_offset**2))
    )
    return segment_area / math.pi


def compute_fullness_score(
    geometry: ContainerGeometry | str,
    fill_height_ratio: float | None,
    waste_mask_area_ratio: float | None = None,
) -> int:
    geometry_value = geometry.value if isinstance(geometry, ContainerGeometry) else str(geometry)
    if fill_height_ratio is None:
        volume_ratio = clamp_ratio(waste_mask_area_ratio)
    elif geometry_value == ContainerGeometry.CYLINDRICAL.value:
        volume_ratio = cylindrical_segment_volume_ratio(fill_height_ratio)
    else:
        volume_ratio = clamp_ratio(fill_height_ratio)

    return int(round(clamp_ratio(volume_ratio) * 100))
