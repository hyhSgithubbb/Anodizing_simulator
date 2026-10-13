import json
from importlib import resources

import pytest

from anodsim.core import units as u
from anodsim.core.presets import PresetDB, PresetError


def test_all_presets_load(presets):
    assert set(presets.metals) == {"Al", "Ti"}
    assert len(presets.electrolytes) == 7
    assert {"pb", "al", "sus316", "ti", "graphite", "pt"} <= set(presets.cathodes)
    for e in presets.electrolytes.values():
        assert e.metal in presets.metals
        assert e.recommended_cathodes and all(c in presets.cathodes for c in e.recommended_cathodes)
        assert (e.steady_state is not None) == (e.regime == "porous")
        assert (e.dissolution is not None) == (e.regime == "porous")
        assert not e.calibrated


def test_electrolytes_for_metal(presets):
    assert {e.id for e in presets.electrolytes_for("Ti")} == {"ti_color_acid", "ti_nanotube_eg"}
    assert all(e.metal == "Al" for e in presets.electrolytes_for("Al"))


def test_properties_at_reference(presets):
    for e in presets.electrolytes.values():
        p = e.properties(e.temperature_ref)
        assert p.conductivity > 0
        # 양극산화비는 기준 전류밀도에서 프리셋 값과 같아야 한다
        assert p.high_field.anodizing_ratio(e.anodizing_ratio_j_ref) == pytest.approx(e.anodizing_ratio, rel=1e-9)
        if e.regime == "porous":
            assert p.j_ss_ref == pytest.approx(e.steady_state.j_ref)


def test_conductivity_interpolation_and_temperature(presets):
    e = presets.electrolyte("al_sulfuric_typeII")
    assert e.conductivity(e.temperature_ref, 12.5) == pytest.approx(47.5)
    warmer = e.conductivity(e.temperature_ref + 10.0)
    assert warmer == pytest.approx(55.0 * (1 + 0.015 * 10.0))


def test_arrhenius_temperature_dependence(presets):
    e = presets.electrolyte("al_sulfuric_typeII")
    cold, warm = e.properties(u.celsius_to_kelvin(10.0)), e.properties(u.celsius_to_kelvin(30.0))
    assert warm.j_ss_ref > e.steady_state.j_ref > cold.j_ss_ref
    assert warm.surface_dissolution > cold.surface_dissolution


def _copy_presets(directory):
    for name in ("metals.json", "cathodes.json", "electrolytes.json"):
        text = resources.files("anodsim.data").joinpath(name).read_text(encoding="utf-8")
        (directory / name).write_text(text, encoding="utf-8")


def test_load_from_directory(tmp_path):
    _copy_presets(tmp_path)
    assert len(PresetDB.load(tmp_path).electrolytes) == 7


def test_missing_key_reports_context(tmp_path):
    _copy_presets(tmp_path)
    path = tmp_path / "electrolytes.json"
    broken = json.loads(path.read_text(encoding="utf-8"))
    del broken["electrolytes"][0]["high_field"]
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(PresetError, match="al_sulfuric_typeII.*high_field"):
        PresetDB.load(tmp_path)


def test_unknown_lookup_raises(presets):
    with pytest.raises(PresetError):
        presets.electrolyte("nope")
    with pytest.raises(PresetError):
        presets.cathode("nope")
    with pytest.raises(PresetError):
        presets.metal("Mg")
