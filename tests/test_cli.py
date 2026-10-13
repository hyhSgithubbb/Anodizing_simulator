import csv

from anodsim import cli


def test_list(capsys):
    assert cli.main(["--list"]) == 0
    assert "al_sulfuric_typeII" in capsys.readouterr().out


def test_run_and_csv(tmp_path, capsys):
    out = tmp_path / "run.csv"
    assert cli.main(["--metal", "Ti", "--mode", "CV", "--voltage", "30", "--minutes", "2", "--csv", str(out)]) == 0
    assert "TiO2" in capsys.readouterr().out or out.exists()
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) > 10 and float(rows[-1]["time_min"]) == 2.0


def test_invalid_config_returns_error(capsys):
    assert cli.main(["--metal", "Ti", "--electrolyte", "al_oxalic", "--mode", "CV", "--voltage", "40"]) == 2
    assert "설정 오류" in capsys.readouterr().err


def test_l1_profile_csv(tmp_path, capsys):
    out = tmp_path / "profile.csv"
    assert cli.main(["--metal", "Ti", "--electrolyte", "ti_nanotube_eg", "--mode", "CV", "--voltage", "60",
                     "--minutes", "2", "--one-cathode", "--profile-csv", str(out)]) == 0
    text = capsys.readouterr().out
    assert "해석 L1" in text and "편차" in text
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) > 100
    assert {r["face"] for r in rows} == {"−x 면", "바닥", "+x 면", "윗면"}
