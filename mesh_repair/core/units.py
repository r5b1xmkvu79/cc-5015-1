"""Model unit handling.

Coordinates are stored exactly as authored in the file.  All repair and
detection thresholds are expressed in the model unit, and converted to
millimetres for display and for the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass

# millimetres per unit
_UNITS: dict[str, tuple[str, float]] = {
    "mm": ("毫米", 1.0),
    "cm": ("厘米", 10.0),
    "m": ("米", 1000.0),
    "in": ("英寸", 25.4),
    "ft": ("英尺", 304.8),
}

UNIT_NAMES = list(_UNITS)


def unit_label(unit: str) -> str:
    return _UNITS[unit][0]


def mm_per_unit(unit: str) -> float:
    return _UNITS[unit][1]


def to_mm(value: float, unit: str) -> float:
    return float(value) * mm_per_unit(unit)


def from_mm(value_mm: float, unit: str) -> float:
    return float(value_mm) / mm_per_unit(unit)


def format_length(value: float, unit: str, digits: int = 4) -> str:
    return f"{value:.{digits}g} {unit}"


def format_mm(value: float, unit: str, digits: int = 4) -> str:
    if unit == "mm":
        return f"{value:.{digits}g} mm"
    return f"{value:.{digits}g} {unit} ({to_mm(value, unit):.{digits}g} mm)"


@dataclass(frozen=True)
class UnitContext:
    """A threshold expressed in the model unit with its physical equivalent."""

    unit: str
    value: float

    @property
    def mm(self) -> float:
        return to_mm(self.value, self.unit)

    def describe(self) -> str:
        return format_mm(self.value, self.unit)


def scale_guard(value: float, unit: str, bbox_diagonal: float) -> tuple[float, list[str]]:
    """Clamp an absolute threshold to a sane fraction of the model size.

    Returns the clamped value and human readable warnings.  This keeps a
    user-entered threshold from collapsing a whole model when unit metadata
    was wrong (for example treating a metre model as millimetres).
    """
    warnings: list[str] = []
    if value < 0:
        value = 0.0
        warnings.append("阈值不能为负，已按 0 处理")
    from . import units as _u  # local to avoid a cycle at import time

    if bbox_diagonal > 0:
        limit = bbox_diagonal * 0.08
        if value > limit:
            warnings.append(
                f"阈值 {_u.format_mm(value, unit)} 超过模型对角线的 8%，已限制为 "
                f"{_u.format_mm(limit, unit)}"
            )
            value = limit
    return value, warnings
