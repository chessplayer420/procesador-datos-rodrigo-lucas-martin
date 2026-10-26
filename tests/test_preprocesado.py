import pytest
import pandas as pd
import numpy as np
from preprocesado import detectar_valores_inexistentes, tratar_valores_inexistentes, validar_seleccion_columnas


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


@pytest.fixture
def df_prueba():
    return pd.DataFrame({
        "edad": [25, 30, 35],             # Numérico (int)
        "precio": [100.5, 200.0, 150.2],   # Numérico (float)
        "categoria": ["A", "B", "C"],       # NO numérico (string)
        "habitaciones": [2, 3, 4]          # Numérico (int)
    })

def test_seleccion_valida(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad", "habitaciones"], target="precio")
    assert exito is True
    assert "Selección confirmada correctamente" in msg

def test_error_sin_features(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=[], target="precio")
    assert exito is False
    assert "al menos una columna de entrada" in msg

def test_error_sin_target(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad"], target="")
    assert exito is False
    assert "columna de salida" in msg

def test_error_target_mismas_features(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad", "precio"], target="precio")
    assert exito is False
    assert "no puede ser seleccionada también como columna de entrada" in msg

def test_error_columna_no_numerica(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad", "categoria"], target="precio")
    assert exito is False
    assert "categoria" in msg
    assert "no son numéricas" in msg


if __name__ == "__main__":
    app = QApplication(sys.argv)
    df_tr = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    df_te = pd.DataFrame({"x": [np.nan, 3.0]})
    win = VentanaPreprocesado(df_tr, df_te, ["x"])
    win.show()
    sys.exit(app.exec())