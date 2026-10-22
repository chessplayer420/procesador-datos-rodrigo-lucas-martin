import pytest
import pandas as pd
import numpy as np
from preprocesado import detectar_valores_inexistentes, tratar_valores_inexistentes


@pytest.fixture
def datasets_ejemplo():
    df_train = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0]})
    df_test = pd.DataFrame({"num": [np.nan, 5.0]})
    return df_train, df_test


def test_detectar_valores_inexistentes(datasets_ejemplo):
    train, test = datasets_ejemplo
    reporte = detectar_valores_inexistentes(train, test, ["num"])
    assert reporte["entrenamiento"]["num"] == 1
    assert reporte["test"]["num"] == 1


def test_tratar_media_sin_fuga_de_datos(datasets_ejemplo):
    train, test = datasets_ejemplo
    # Media en train es (1+2+4)/3 = 2.333333
    train_c, test_c, _ = tratar_valores_inexistentes(train, test, ["num"], "media")
    assert np.isclose(train_c.loc[2, "num"], 2.333333)
    # Debe aplicar la media de TRAIN al conjunto TEST
    assert np.isclose(test_c.loc[0, "num"], 2.333333)


def test_error_constante_no_numerica(datasets_ejemplo):
    train, test = datasets_ejemplo
    with pytest.raises(ValueError, match="número válido"):
        tratar_valores_inexistentes(train, test, ["num"], "constante", valor_constante="texto_invalido")