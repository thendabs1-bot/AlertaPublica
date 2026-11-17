"""Tests mínimos de detector.py con un DataFrame sintético (sin SQLite, sin red)."""
import numpy as np
import pandas as pd
import pytest

import detector


def _fila(i, entidad="E1", ruc="R1", ref=100_000.0, adj=100_000.0, dias=20.0):
    dif = (adj - ref) / ref * 100 if ref else np.nan
    return {
        "codigoentidad": entidad,
        "entidad": f"Entidad {entidad}",
        "ruc_proveedor": ruc,
        "proveedor": f"Proveedor {ruc}",
        "monto_referencial": ref,
        "monto_adjudicado": adj,
        "dias_proceso": dias,
        "diferencia_pct": dif,
        "id": i,
    }


def _df(filas):
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- detector 2
def test_proveedor_dominante():
    filas = []
    # E1: R1 gana 4 de 5 (80 %) -> dominante
    filas += [_fila(i, "E1", "R1") for i in range(4)]
    filas += [_fila(4, "E1", "R9")]
    # E2: R2 gana 2 de 3 (66 %) pero < 3 contratos -> no dominante
    filas += [_fila(5, "E2", "R2"), _fila(6, "E2", "R2"), _fila(7, "E2", "R8")]
    # E3: R3 gana 3 de 6 (50 %) -> no dominante
    filas += [_fila(8 + i, "E3", "R3") for i in range(3)]
    filas += [_fila(11 + i, "E3", f"X{i}") for i in range(3)]

    dom = detector.detectar_proveedor_dominante(_df(filas))

    assert list(zip(dom["codigoentidad"], dom["ruc_proveedor"])) == [("E1", "R1")]
    fila = dom.iloc[0]
    assert fila["contratos_proveedor"] == 4
    assert fila["total"] == 5
    assert fila["pct_contratos"] == pytest.approx(80.0)


# ---------------------------------------------------------------- detector 3
def test_proceso_rapido():
    filas = [
        _fila(0, adj=60_000, dias=3),      # sí
        _fila(1, adj=200_000, dias=0),     # sí (límite inferior)
        _fila(2, adj=90_000, dias=5),      # sí (límite superior)
        _fila(3, adj=40_000, dias=3),      # monto bajo
        _fila(4, adj=60_000, dias=6),      # demasiado lento
        _fila(5, adj=60_000, dias=-1),     # fechas incoherentes
        _fila(6, adj=60_000, dias=np.nan), # sin fechas
    ]
    rap = detector.detectar_proceso_rapido(_df(filas))

    # Ordenados por monto descendente
    assert list(rap["id"]) == [1, 2, 0]


# ---------------------------------------------------------------- detector 4
def test_diferencia_monto():
    filas = [
        _fila(0, ref=100_000, adj=200_000),  # +100 %, sí
        _fila(1, ref=250_000, adj=150_000),  # -40 %, sí
        _fila(2, ref=30_000, adj=50_000),    # +66 % pero monto bajo
        _fila(3, ref=500_000, adj=550_000),  # +10 %
        _fila(4, ref=0, adj=500_000),        # diferencia NaN
    ]
    dif = detector.detectar_diferencia_monto(_df(filas))

    assert list(dif["id"]) == [0, 1]


# ---------------------------------------------------------------- detector 1
def test_anomalias_monto_isolation_forest():
    rng = np.random.default_rng(0)
    filas = []
    for i in range(60):
        ref = float(rng.normal(100_000, 5_000))
        adj = ref * float(rng.normal(1.0, 0.02))
        filas.append(_fila(i, ref=ref, adj=adj))
    # Atípico evidente
    filas.append(_fila(60, ref=100_000, adj=50_000_000))
    # Filas que deben descartarse antes del modelo
    filas.append(_fila(61, ref=100_000, adj=0))
    filas.append(_fila(62, ref=np.nan, adj=100_000))
    infinito = _fila(63, ref=100_000, adj=100_000)
    infinito["diferencia_pct"] = np.inf
    filas.append(infinito)
    df = _df(filas)

    df_valid, anomalias = detector.detectar_anomalias_monto(df)

    assert set(df_valid["id"]) == set(range(61)) | {63}
    assert np.isfinite(df_valid["diferencia_pct"]).all()
    assert set(df_valid["anomalia_monto"].unique()) <= {-1, 1}
    assert 60 in set(anomalias["id"])
    # contamination=0.05 -> ~5 % de las filas válidas
    assert 1 <= len(anomalias) <= int(np.ceil(0.05 * len(df_valid))) + 1
    # No modifica el DataFrame original
    assert "anomalia_monto" not in df.columns


# ---------------------------------------------------------------- score
def test_calcular_score_riesgo():
    filas = [_fila(i, "E1", "R1") for i in range(3)] + [
        _fila(3, "E2", "R2"),
        _fila(4, "E2", "R3"),
    ]
    df = _df(filas)

    anomalias_if = df.loc[[0]]
    dominantes = pd.DataFrame({"codigoentidad": ["E1"], "ruc_proveedor": ["R1"]})
    rapidos = df.loc[[0, 3]]
    diferencias = df.loc[[0, 4]]

    scored, alertas = detector.calcular_score_riesgo(
        df.copy(), anomalias_if, dominantes, rapidos, diferencias
    )

    assert list(scored["score_riesgo"]) == [100, 30, 30, 20, 10]
    assert list(alertas["id"]) == [0]


def test_calcular_score_riesgo_sin_hallazgos():
    df = _df([_fila(i) for i in range(3)])
    vacio = df.iloc[0:0]
    scored, alertas = detector.calcular_score_riesgo(
        df.copy(), vacio, vacio, vacio, vacio
    )
    assert (scored["score_riesgo"] == 0).all()
    assert alertas.empty


def test_pipeline_completo_sintetico():
    """Encadena los detectores como en __main__ (sin SQLite)."""
    rng = np.random.default_rng(1)
    filas = []
    for i in range(40):
        ref = float(rng.normal(80_000, 4_000))
        filas.append(_fila(i, f"E{i % 4}", f"R{i % 7}", ref=ref, adj=ref * 0.98))
    # Contrato que dispara todo: atípico, rápido, gran diferencia, proveedor dominante
    filas += [_fila(100 + i, "EX", "RX", ref=100_000, adj=100_000) for i in range(3)]
    filas.append(_fila(200, "EX", "RX", ref=100_000, adj=40_000_000, dias=1))
    df = _df(filas)

    df_scored, anomalias_if = detector.detectar_anomalias_monto(df)
    dominantes = detector.detectar_proveedor_dominante(df)
    rapidos = detector.detectar_proceso_rapido(df)
    diferencias = detector.detectar_diferencia_monto(df)
    df_final, alertas = detector.calcular_score_riesgo(
        df_scored, anomalias_if, dominantes, rapidos, diferencias
    )

    fila = df_final.set_index("id").loc[200]
    assert fila["score_riesgo"] == 100
    assert 200 in set(alertas["id"])
    assert df_final["score_riesgo"].between(0, 100).all()
