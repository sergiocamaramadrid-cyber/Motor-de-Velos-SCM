"""
tests/test_SCM_Motor_de_Velos_Nebulosas.py

Tests unitarios para scripts/SCM_Motor_de_Velos_Nebulosas.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Asegura que el directorio raíz del repo esté en sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from SCM_Motor_de_Velos_Nebulosas import (  # noqa: E402
    N_PERMUTACIONES,
    N_PUNTOS_M16,
    RANDOM_SEED,
    SPEARMAN_RHO_ESPERADO,
    UMBRAL_PUNTAS,
    crear_animacion_burbuja,
    generar_figura_estatica,
    guardar_csv_burbuja,
    guardar_csv_pilares,
    main,
    simular_burbuja,
    simular_pilares,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture
def res_burbuja(rng):
    return simular_burbuja(rng=rng)


@pytest.fixture
def res_pilares(rng):
    return simular_pilares(rng=rng)


# ---------------------------------------------------------------------------
# simular_burbuja
# ---------------------------------------------------------------------------


class TestSimularBurbuja:
    def test_returns_dict_with_expected_keys(self, res_burbuja):
        assert set(res_burbuja.keys()) == {"theta", "densidad", "F3", "rho", "p_val"}

    def test_theta_shape(self, res_burbuja):
        assert res_burbuja["theta"].shape == (200,)

    def test_densidad_shape(self, res_burbuja):
        assert res_burbuja["densidad"].shape == (200,)

    def test_F3_shape(self, res_burbuja):
        assert res_burbuja["F3"].shape == (200,)

    def test_theta_range(self, res_burbuja):
        theta = res_burbuja["theta"]
        assert theta[0] == pytest.approx(0.0)
        assert theta[-1] == pytest.approx(2 * np.pi)

    def test_rho_negative(self, res_burbuja):
        """F3 debe ser negativamente correlacionada con la densidad."""
        assert res_burbuja["rho"] < 0

    def test_rho_stronger_than_expected(self, res_burbuja):
        """ρ debe ser al menos tan negativo como el valor de referencia -0.45."""
        assert res_burbuja["rho"] <= SPEARMAN_RHO_ESPERADO

    def test_p_val_float(self, res_burbuja):
        assert isinstance(res_burbuja["p_val"], float)

    def test_p_val_valid_range(self, res_burbuja):
        assert 0.0 <= res_burbuja["p_val"] <= 1.0

    def test_rho_float(self, res_burbuja):
        assert isinstance(res_burbuja["rho"], float)

    def test_reproducible_with_same_seed(self):
        r1 = simular_burbuja(rng=np.random.default_rng(0))
        r2 = simular_burbuja(rng=np.random.default_rng(0))
        np.testing.assert_array_equal(r1["F3"], r2["F3"])

    def test_different_seeds_give_different_F3(self):
        r1 = simular_burbuja(rng=np.random.default_rng(1))
        r2 = simular_burbuja(rng=np.random.default_rng(99))
        assert not np.allclose(r1["F3"], r2["F3"])

    def test_densidad_values_positive(self, res_burbuja):
        """La densidad debe ser positiva (1 + 0.8*cos ∈ [0.2, 1.8])."""
        assert np.all(res_burbuja["densidad"] > 0)

    def test_default_rng_uses_fixed_seed(self):
        """Sin rng, el resultado debe ser reproducible (usa RANDOM_SEED)."""
        r1 = simular_burbuja()
        r2 = simular_burbuja()
        np.testing.assert_array_equal(r1["theta"], r2["theta"])

    def test_densidad_max_value(self, res_burbuja):
        assert res_burbuja["densidad"].max() <= 1.8 + 1e-10


# ---------------------------------------------------------------------------
# simular_pilares
# ---------------------------------------------------------------------------


class TestSimularPilares:
    def test_returns_dict_with_expected_keys(self, res_pilares):
        assert set(res_pilares.keys()) == {
            "x", "densidad", "acumulacion", "F3", "delta_AIC", "p_perm", "slope"
        }

    def test_x_shape(self, res_pilares):
        assert res_pilares["x"].shape == (N_PUNTOS_M16,)

    def test_densidad_shape(self, res_pilares):
        assert res_pilares["densidad"].shape == (N_PUNTOS_M16,)

    def test_acumulacion_shape(self, res_pilares):
        assert res_pilares["acumulacion"].shape == (N_PUNTOS_M16,)

    def test_F3_shape(self, res_pilares):
        assert res_pilares["F3"].shape == (N_PUNTOS_M16,)

    def test_x_range(self, res_pilares):
        assert res_pilares["x"][0] == pytest.approx(0.0)
        assert res_pilares["x"][-1] == pytest.approx(10.0)

    def test_delta_AIC_positive(self, res_pilares):
        """ΔAIC > 0 significa que el modelo con acumulación es mejor."""
        assert res_pilares["delta_AIC"] > 0

    def test_p_perm_in_range(self, res_pilares):
        assert 0.0 <= res_pilares["p_perm"] <= 1.0

    def test_slope_negative(self, res_pilares):
        """La pendiente F3 vs acumulación debe ser negativa."""
        assert res_pilares["slope"] < 0

    def test_slope_float(self, res_pilares):
        assert isinstance(res_pilares["slope"], float)

    def test_acumulacion_non_negative(self, res_pilares):
        assert np.all(res_pilares["acumulacion"] >= 0)

    def test_reproducible_with_same_seed(self):
        r1 = simular_pilares(rng=np.random.default_rng(5))
        r2 = simular_pilares(rng=np.random.default_rng(5))
        np.testing.assert_array_equal(r1["F3"], r2["F3"])

    def test_default_rng_uses_fixed_seed(self):
        r1 = simular_pilares()
        r2 = simular_pilares()
        np.testing.assert_array_equal(r1["x"], r2["x"])

    def test_delta_AIC_float(self, res_pilares):
        assert isinstance(res_pilares["delta_AIC"], float)

    def test_p_perm_float(self, res_pilares):
        assert isinstance(res_pilares["p_perm"], float)


# ---------------------------------------------------------------------------
# guardar_csv_burbuja
# ---------------------------------------------------------------------------


class TestGuardarCsvBurbuja:
    def test_creates_file(self, res_burbuja, tmp_path):
        path = guardar_csv_burbuja(res_burbuja, tmp_path)
        assert path.exists()

    def test_file_extension(self, res_burbuja, tmp_path):
        path = guardar_csv_burbuja(res_burbuja, tmp_path)
        assert path.suffix == ".csv"

    def test_filename(self, res_burbuja, tmp_path):
        path = guardar_csv_burbuja(res_burbuja, tmp_path)
        assert path.name == "resultados_burbuja_NGC7635.csv"

    def test_csv_has_header(self, res_burbuja, tmp_path):
        path = guardar_csv_burbuja(res_burbuja, tmp_path)
        import pandas as pd
        df = pd.read_csv(path, comment="#")
        assert "theta_rad" in df.columns
        assert "densidad_entorno" in df.columns
        assert "F3_pendiente_externa" in df.columns

    def test_csv_row_count(self, res_burbuja, tmp_path):
        path = guardar_csv_burbuja(res_burbuja, tmp_path)
        import pandas as pd
        df = pd.read_csv(path, comment="#")
        assert len(df) == 200

    def test_csv_metadata_comment(self, res_burbuja, tmp_path):
        path = guardar_csv_burbuja(res_burbuja, tmp_path)
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        assert "rho_Spearman" in first_line

    def test_creates_output_dir(self, res_burbuja, tmp_path):
        new_dir = tmp_path / "subdir" / "nested"
        guardar_csv_burbuja(res_burbuja, new_dir)
        assert new_dir.exists()

    def test_returns_path(self, res_burbuja, tmp_path):
        result = guardar_csv_burbuja(res_burbuja, tmp_path)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# guardar_csv_pilares
# ---------------------------------------------------------------------------


class TestGuardarCsvPilares:
    def test_creates_file(self, res_pilares, tmp_path):
        path = guardar_csv_pilares(res_pilares, tmp_path)
        assert path.exists()

    def test_filename(self, res_pilares, tmp_path):
        path = guardar_csv_pilares(res_pilares, tmp_path)
        assert path.name == "resultados_pilares_M16.csv"

    def test_csv_has_header(self, res_pilares, tmp_path):
        path = guardar_csv_pilares(res_pilares, tmp_path)
        import pandas as pd
        df = pd.read_csv(path, comment="#")
        assert "posicion" in df.columns
        assert "densidad_barionica" in df.columns
        assert "acumulacion_energia" in df.columns
        assert "F3_pendiente_externa" in df.columns

    def test_csv_row_count(self, res_pilares, tmp_path):
        path = guardar_csv_pilares(res_pilares, tmp_path)
        import pandas as pd
        df = pd.read_csv(path, comment="#")
        assert len(df) == N_PUNTOS_M16

    def test_csv_metadata_comment(self, res_pilares, tmp_path):
        path = guardar_csv_pilares(res_pilares, tmp_path)
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        assert "delta_AIC" in first_line

    def test_returns_path(self, res_pilares, tmp_path):
        result = guardar_csv_pilares(res_pilares, tmp_path)
        assert isinstance(result, Path)

    def test_creates_output_dir(self, res_pilares, tmp_path):
        new_dir = tmp_path / "outdir"
        guardar_csv_pilares(res_pilares, new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# generar_figura_estatica
# ---------------------------------------------------------------------------


class TestGenerarFiguraEstatica:
    def test_creates_file(self, res_burbuja, res_pilares, tmp_path):
        path = generar_figura_estatica(res_burbuja, res_pilares, tmp_path)
        assert path.exists()

    def test_filename(self, res_burbuja, res_pilares, tmp_path):
        path = generar_figura_estatica(res_burbuja, res_pilares, tmp_path)
        assert path.name == "simulacion_nebulosas_SCM.png"

    def test_returns_path(self, res_burbuja, res_pilares, tmp_path):
        result = generar_figura_estatica(res_burbuja, res_pilares, tmp_path)
        assert isinstance(result, Path)

    def test_file_size_nonzero(self, res_burbuja, res_pilares, tmp_path):
        path = generar_figura_estatica(res_burbuja, res_pilares, tmp_path)
        assert path.stat().st_size > 1000

    def test_creates_output_dir(self, res_burbuja, res_pilares, tmp_path):
        new_dir = tmp_path / "fig_out"
        generar_figura_estatica(res_burbuja, res_pilares, new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# crear_animacion_burbuja (requiere pillow)
# ---------------------------------------------------------------------------


class TestCrearAnimacionBurbuja:
    pillow_available = pytest.mark.skipif(
        not _pillow_ok(), reason="pillow no instalado"
    ) if False else None  # se evalúa abajo

    @staticmethod
    def _pillow_ok():
        try:
            import PIL  # noqa: F401
            return True
        except ImportError:
            return False

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("PIL"),
        reason="pillow no instalado",
    )
    def test_creates_gif(self, res_burbuja, tmp_path):
        path = crear_animacion_burbuja(res_burbuja, tmp_path)
        assert path.exists()
        assert path.suffix == ".gif"

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("PIL"),
        reason="pillow no instalado",
    )
    def test_returns_path(self, res_burbuja, tmp_path):
        result = crear_animacion_burbuja(res_burbuja, tmp_path)
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_dict(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation", "--seed", "7"])
        assert isinstance(result, dict)

    def test_dict_keys(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation"])
        assert "burbuja" in result
        assert "pilares" in result
        assert "csv_burbuja" in result
        assert "csv_pilares" in result
        assert "figura" in result
        assert "animacion" in result

    def test_animacion_none_when_disabled(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation"])
        assert result["animacion"] is None

    def test_csv_files_created(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation"])
        assert result["csv_burbuja"].exists()
        assert result["csv_pilares"].exists()

    def test_figure_created(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation"])
        assert result["figura"].exists()

    def test_burbuja_rho_negative(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation"])
        assert result["burbuja"]["rho"] < 0

    def test_pilares_delta_AIC_positive(self, tmp_path):
        result = main(["--out-dir", str(tmp_path), "--no-animation"])
        assert result["pilares"]["delta_AIC"] > 0

    def test_custom_seed(self, tmp_path):
        r1 = main(["--out-dir", str(tmp_path / "a"), "--no-animation", "--seed", "42"])
        r2 = main(["--out-dir", str(tmp_path / "b"), "--no-animation", "--seed", "42"])
        assert r1["burbuja"]["rho"] == pytest.approx(r2["burbuja"]["rho"])

    def test_output_dir_is_created(self, tmp_path):
        new_dir = tmp_path / "nested" / "output"
        main(["--out-dir", str(new_dir), "--no-animation"])
        assert new_dir.exists()
