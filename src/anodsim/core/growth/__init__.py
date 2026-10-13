"""산화막 성장 모델 (치밀형 / 다공성형)."""

from .base import GrowthModel, GrowthRates
from .compact import CompactGrowth
from .high_field import HighFieldLaw
from .porous import PorousGrowth


def make_growth_model(metal, props) -> GrowthModel:
    """전해질 프리셋의 성장 모드(regime)에 맞는 모델을 만든다."""
    if props.regime == "compact":
        return CompactGrowth(metal, props)
    if props.regime == "porous":
        return PorousGrowth(metal, props)
    raise ValueError(f"unknown growth regime: {props.regime!r}")


__all__ = ["CompactGrowth", "GrowthModel", "GrowthRates", "HighFieldLaw", "PorousGrowth", "make_growth_model"]
