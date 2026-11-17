"""Tests de ingest.procesar con datos sintéticos en texto, como llegan del Excel."""
import numpy as np
import pandas as pd
import pytest

import ingest


def test_procesar():
    df = pd.DataFrame(
        {
            "monto_referencial_item_soles": ["100000", "200000", "0", "abc"],
            "monto_adjudicado_item_soles": ["150000", "180000", "5000", "1000"],
            "fecha_convocatoria": ["01/02/2024", "10/03/2025", "xx", "01/01/2026"],
            "fecha_buenapro": ["06/02/2024", "09/04/2025", "01/01/2025", None],
        },
        dtype=object,
    )

    out = ingest.procesar(df)

    assert out["monto_referencial"].tolist()[:3] == [100000.0, 200000.0, 0.0]
    assert np.isnan(out["monto_referencial"].iloc[3])
    assert out["monto_adjudicado"].tolist() == [150000.0, 180000.0, 5000.0, 1000.0]

    # dayfirst=True: 01/02 es 1 de febrero, no 2 de enero
    assert out["fecha_convocatoria"].iloc[0] == pd.Timestamp("2024-02-01")
    assert out["dias_proceso"].iloc[0] == 5
    assert out["dias_proceso"].iloc[1] == 30
    assert pd.isna(out["dias_proceso"].iloc[2])
    assert pd.isna(out["dias_proceso"].iloc[3])

    assert out["diferencia_pct"].iloc[0] == pytest.approx(50.0)
    assert out["diferencia_pct"].iloc[1] == pytest.approx(-10.0)
    # Referencial 0 -> infinito reemplazado por 0
    assert out["diferencia_pct"].iloc[2] == 0
    # Referencial no numérico -> NaN
    assert np.isnan(out["diferencia_pct"].iloc[3])
