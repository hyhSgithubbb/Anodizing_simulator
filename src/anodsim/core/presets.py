"""JSON 프리셋(금속·전해질·음극재) 로더.

JSON 은 사람이 읽기 쉬운 단위(g/cm³, nm/V, μm/h, °C …)로 쓰고, 여기서 SI 로 바꾼다.
기본 프리셋은 ``anodsim/data/*.json`` 에 있다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from . import units as u
from .cathode import CathodeMaterial
from .electrolyte import DissolutionParams, ElectrolytePreset, Range, SteadyStateParams
from .materials import Metal, Oxide


class PresetError(ValueError):
    pass


def _get(d: dict, key: str, ctx: str):
    try:
        return d[key]
    except KeyError:
        raise PresetError(f"{ctx}: '{key}' 항목이 없습니다.") from None


def _range(values, ctx: str, scale: float = 1.0, offset: float = 0.0) -> Range:
    if not isinstance(values, (list, tuple)) or len(values) != 2 or values[0] > values[1]:
        raise PresetError(f"{ctx}: 범위는 [하한, 상한] 형식이어야 합니다.")
    return Range(values[0] * scale + offset, values[1] * scale + offset)


def parse_metal(d: dict) -> Metal:
    ctx = f"metal {d.get('symbol', '?')}"
    o = _get(d, "oxide", ctx)
    oxide = Oxide(
        formula=_get(o, "formula", ctx),
        molar_mass=_get(o, "molar_mass_g_mol", ctx) * u.G_PER_MOL,
        density=_get(o, "density_g_cm3", ctx) * u.G_PER_CM3,
        electrons=int(_get(o, "electrons_per_formula", ctx)),
        metal_atoms=int(_get(o, "metal_atoms_per_formula", ctx)),
        refractive_index=float(o.get("refractive_index", 1.0)),
    )
    return Metal(
        symbol=_get(d, "symbol", ctx),
        name=_get(d, "name", ctx),
        molar_mass=_get(d, "molar_mass_g_mol", ctx) * u.G_PER_MOL,
        density=_get(d, "density_g_cm3", ctx) * u.G_PER_CM3,
        valence=int(_get(d, "valence", ctx)),
        native_oxide_thickness=_get(d, "native_oxide_nm", ctx) * u.NM,
        oxide=oxide,
    )


def parse_electrolyte(d: dict) -> ElectrolytePreset:
    ctx = f"electrolyte {d.get('id', '?')}"
    regime = _get(d, "regime", ctx)
    if regime not in ("compact", "porous"):
        raise PresetError(f"{ctx}: regime 은 'compact' 또는 'porous' 여야 합니다.")
    conc = _get(d, "concentration", ctx)
    temp = _get(d, "temperature_C", ctx)
    kappa = _get(d, "conductivity", ctx)
    hf = _get(d, "high_field", ctx)
    limits = _get(d, "limits", ctx)
    k_conc, k_vals = tuple(_get(kappa, "concentration", ctx)), tuple(_get(kappa, "S_per_m", ctx))
    if len(k_conc) != len(k_vals) or len(k_conc) == 0 or list(k_conc) != sorted(k_conc):
        raise PresetError(f"{ctx}: conductivity 표는 오름차순 농도와 같은 길이의 값이어야 합니다.")

    steady = dissolution = None
    if regime == "porous":
        ss = _get(d, "steady_state", ctx)
        steady = SteadyStateParams(
            j_ref=_get(ss, "j_ref_A_m2", ctx),
            U_ref=_get(ss, "U_ref_V", ctx),
            U_s=_get(ss, "U_s_V", ctx),
            activation_energy=_get(ss, "activation_kJ_mol", ctx) * u.KJ_PER_MOL,
        )
        ds = _get(d, "dissolution", ctx)
        dissolution = DissolutionParams(
            surface_rate=_get(ds, "surface_um_per_h", ctx) * u.UM_PER_HOUR,
            pore_wall_rate=_get(ds, "pore_wall_per_h", ctx) * u.PER_HOUR,
            barrier_thinning=_get(ds, "barrier_thinning_nm_per_s", ctx) * u.NM_PER_SECOND,
            concentration_order=_get(ds, "concentration_order", ctx),
            activation_energy=_get(ds, "activation_kJ_mol", ctx) * u.KJ_PER_MOL,
        )

    porosity = float(d.get("porosity", 0.0))
    voltage_range = _range(_get(d, "voltage_range_V", ctx), ctx)
    eta = float(_get(d, "current_efficiency", ctx))
    if not 0.0 < eta <= 1.0:
        raise PresetError(f"{ctx}: current_efficiency 는 (0, 1] 범위여야 합니다.")
    if not 0.0 <= porosity < 1.0:
        raise PresetError(f"{ctx}: porosity 는 [0, 1) 범위여야 합니다.")

    return ElectrolytePreset(
        id=_get(d, "id", ctx),
        name=_get(d, "name", ctx),
        metal=_get(d, "metal", ctx),
        regime=regime,
        description=d.get("description", ""),
        tags=tuple(d.get("tags", ())),
        recommended_cathodes=tuple(d.get("recommended_cathodes", ())),
        species=_get(conc, "species", ctx),
        concentration_unit=_get(conc, "unit", ctx),
        concentration_ref=_get(conc, "ref", ctx),
        concentration_range=_range(_get(conc, "range", ctx), ctx),
        temperature_ref=u.celsius_to_kelvin(_get(temp, "ref", ctx)),
        temperature_range=_range(_get(temp, "range", ctx), ctx, offset=u.celsius_to_kelvin(0.0)),
        voltage_range=voltage_range,
        typical_voltage=float(d.get("typical_voltage_V", 0.5 * (voltage_range.low + voltage_range.high))),
        conductivity_concentrations=k_conc,
        conductivity_values=k_vals,
        conductivity_temp_coeff=_get(kappa, "temp_coeff_per_K", ctx),
        current_efficiency=eta,
        porosity=porosity,
        field_coefficient=_get(hf, "B_nm_per_V", ctx) * u.NM,
        anodizing_ratio=_get(hf, "anodizing_ratio_nm_per_V", ctx) * u.NM,
        anodizing_ratio_j_ref=_get(hf, "j_ref_A_m2", ctx),
        steady_state=steady,
        dissolution=dissolution,
        breakdown_voltage=_get(limits, "breakdown_V", ctx),
        burning_current_density=_get(limits, "burning_A_dm2", ctx) * u.A_PER_DM2,
        calibrated=bool(d.get("calibrated", False)),
    )


def parse_cathode(d: dict) -> CathodeMaterial:
    ctx = f"cathode {d.get('id', '?')}"
    return CathodeMaterial(
        id=_get(d, "id", ctx),
        name=_get(d, "name", ctx),
        exchange_current_density=10.0 ** _get(d, "log10_j0_A_cm2", ctx) * u.A_PER_CM2,
        tafel_slope=_get(d, "tafel_slope_V_per_decade", ctx),
        incompatible_tags=tuple(d.get("incompatible_tags", ())),
        note=d.get("note", ""),
    )


@dataclass(frozen=True)
class PresetDB:
    metals: dict[str, Metal]
    electrolytes: dict[str, ElectrolytePreset]
    cathodes: dict[str, CathodeMaterial]

    @classmethod
    def load(cls, directory: str | Path | None = None) -> PresetDB:
        def read(name: str) -> dict:
            if directory is None:
                text = resources.files("anodsim.data").joinpath(name).read_text(encoding="utf-8")
            else:
                text = Path(directory, name).read_text(encoding="utf-8")
            return json.loads(text)

        metals = {m.symbol: m for m in map(parse_metal, read("metals.json")["metals"])}
        electrolytes = {e.id: e for e in map(parse_electrolyte, read("electrolytes.json")["electrolytes"])}
        cathodes = {c.id: c for c in map(parse_cathode, read("cathodes.json")["cathodes"])}
        for e in electrolytes.values():
            if e.metal not in metals:
                raise PresetError(f"electrolyte {e.id}: 알 수 없는 금속 '{e.metal}'")
            unknown = [c for c in e.recommended_cathodes if c not in cathodes]
            if unknown:
                raise PresetError(f"electrolyte {e.id}: 알 수 없는 음극재 {unknown}")
        return cls(metals, electrolytes, cathodes)

    @classmethod
    @lru_cache(maxsize=1)
    def default(cls) -> PresetDB:
        return cls.load()

    def metal(self, symbol: str) -> Metal:
        try:
            return self.metals[symbol]
        except KeyError:
            raise PresetError(f"알 수 없는 금속: {symbol}") from None

    def electrolyte(self, preset_id: str) -> ElectrolytePreset:
        try:
            return self.electrolytes[preset_id]
        except KeyError:
            raise PresetError(f"알 수 없는 전해질 프리셋: {preset_id}") from None

    def cathode(self, cathode_id: str) -> CathodeMaterial:
        try:
            return self.cathodes[cathode_id]
        except KeyError:
            raise PresetError(f"알 수 없는 음극재: {cathode_id}") from None

    def electrolytes_for(self, metal: str) -> list[ElectrolytePreset]:
        return [e for e in self.electrolytes.values() if e.metal == metal]
