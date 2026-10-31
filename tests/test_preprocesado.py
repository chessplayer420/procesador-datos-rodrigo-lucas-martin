import pytest
import pandas as pd
import numpy as np
from preprocesado import (
    detectar_valores_inexistentes,
    tratar_valores_inexistentes,
    validar_seleccion_columnas,
    separar_train_test,
    parsear_parametros_separacion,
    parsear_valor_constante,
    hay_valores_inexistentes,
    ConjuntoVacioError,
)


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


# --- Casos añadidos ---

def test_seleccion_valida_una_sola_feature(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad"], target="precio")
    assert exito is True
    assert "edad" in msg and "precio" in msg

def test_seleccion_valida_muestra_entrada_y_salida(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad", "habitaciones"], target="precio")
    assert exito is True
    assert "edad" in msg and "habitaciones" in msg
    assert "Columna de salida: precio" in msg

def test_error_sin_features_ni_target_indica_ambas(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=[], target="")
    assert exito is False
    assert "columna de entrada" in msg
    assert "columna de salida" in msg

def test_error_columna_inexistente(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["no_existe"], target="precio")
    assert exito is False
    assert "no_existe" in msg
    assert "no existen" in msg

def test_error_target_no_numerico(df_prueba):
    exito, msg = validar_seleccion_columnas(df_prueba, features=["edad"], target="categoria")
    assert exito is False
    assert "categoria" in msg
    assert "no son numéricas" in msg

def test_error_varias_no_numericas_se_listan_todas(df_prueba):
    df = df_prueba.assign(ciudad=["Lugo", "Vigo", "Madrid"])
    exito, msg = validar_seleccion_columnas(df, features=["categoria", "ciudad", "edad"], target="precio")
    assert exito is False
    assert "categoria" in msg and "ciudad" in msg
    assert "edad" not in msg and "precio" not in msg

def test_error_columna_booleana(df_prueba):
    df = df_prueba.assign(activo=[True, False, True])
    exito, msg = validar_seleccion_columnas(df, features=["activo"], target="precio")
    assert exito is False
    assert "activo" in msg

def test_nombres_de_columna_numericos_como_texto():
    # La GUI pasa los nombres como texto, aunque el DataFrame tenga columnas 0, 1, 2
    df = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    exito, _ = validar_seleccion_columnas(df, features=["0", "1"], target="2")
    assert exito is True


# ---------------------------------------------------------------------------
# Separación en entrenamiento y test
# ---------------------------------------------------------------------------

@pytest.fixture
def df_100():
    # Datos creados en la propia prueba; "id" identifica cada fila de forma única
    return pd.DataFrame({
        "id": range(100),
        "x": np.arange(100, dtype=float),
        "y": np.arange(100, dtype=float) * 2,
    })


def test_separacion_por_defecto_80_20(df_100):
    train, test = separar_train_test(df_100)
    assert len(train) == 80
    assert len(test) == 20


@pytest.mark.parametrize("pct_test, n_test", [(1, 1), (10, 10), (25, 25), (33, 33), (50, 50), (99, 99)])
def test_tamanos_corresponden_al_porcentaje(df_100, pct_test, n_test):
    train, test = separar_train_test(df_100, pct_test, 7)
    assert len(test) == n_test
    assert len(train) == 100 - n_test


def test_tamanos_con_dataset_no_multiplo_de_100():
    df = pd.DataFrame({"x": range(37)})
    train, test = separar_train_test(df, 20, 1)
    assert len(test) == 7          # 37 * 0.20 = 7.4 -> 7
    assert len(train) == 30
    assert len(train) + len(test) == 37


def test_ninguna_fila_en_los_dos_conjuntos(df_100):
    train, test = separar_train_test(df_100, 30, 42)
    assert set(train["id"]).isdisjoint(set(test["id"]))
    # Y entre los dos están todas las filas del dataset
    assert set(train["id"]) | set(test["id"]) == set(df_100["id"])


def test_filas_duplicadas_no_se_mezclan_entre_conjuntos():
    # Filas con valores idénticos siguen siendo filas distintas (se comprueba por índice)
    df = pd.DataFrame({"x": [1.0] * 10, "y": [2.0] * 10})
    train, test = separar_train_test(df, 30, 3)
    assert set(train.index).isdisjoint(set(test.index))
    assert len(train) + len(test) == 10


def test_misma_semilla_misma_separacion(df_100):
    train1, test1 = separar_train_test(df_100, 25, 123)
    train2, test2 = separar_train_test(df_100, 25, 123)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_distinta_semilla_distinta_separacion(df_100):
    _, test1 = separar_train_test(df_100, 25, 1)
    _, test2 = separar_train_test(df_100, 25, 2)
    assert list(test1["id"]) != list(test2["id"])


def test_la_separacion_es_aleatoria_y_no_los_ultimos_n_registros(df_100):
    _, test = separar_train_test(df_100, 20, 42)
    assert list(test["id"]) != list(range(80, 100))


def test_no_modifica_el_dataset_original(df_100):
    copia = df_100.copy()
    train, test = separar_train_test(df_100, 20, 42)
    pd.testing.assert_frame_equal(df_100, copia)
    train.iloc[0, 1] = -999.0   # modificar un conjunto no debe alterar el original
    pd.testing.assert_frame_equal(df_100, copia)


def test_conserva_todas_las_columnas_y_el_indice_original(df_100):
    train, test = separar_train_test(df_100, 20, 42)
    assert list(train.columns) == list(df_100.columns)
    assert list(test.columns) == list(df_100.columns)
    assert (train["id"] == train.index).all()   # el índice original se conserva


def test_error_dataset_con_menos_de_5_filas():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="al menos 5 filas"):
        separar_train_test(df, 20, 42)


def test_dataset_con_exactamente_5_filas_es_valido():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    train, test = separar_train_test(df, 20, 42)
    assert len(train) == 4 and len(test) == 1


def test_error_test_quedaria_vacio():
    df = pd.DataFrame({"x": range(5)})
    with pytest.raises(ValueError, match="test"):
        separar_train_test(df, 1, 42)       # 5 * 1 % = 0.05 -> 0 filas de test


def test_error_entrenamiento_quedaria_vacio():
    df = pd.DataFrame({"x": range(5)})
    with pytest.raises(ValueError, match="entrenamiento"):
        separar_train_test(df, 99, 42)      # 5 * 99 % = 4.95 -> 5 filas de test


@pytest.mark.parametrize("pct", [0, 100, -5, 150, float("nan"), float("inf"), "20", None, True])
def test_error_porcentaje_fuera_de_rango_en_la_funcion(df_100, pct):
    with pytest.raises(ValueError, match="entre 1 y 99"):
        separar_train_test(df_100, pct, 42)


@pytest.mark.parametrize("semilla", [1.5, "42", None, True, -1])
def test_error_semilla_no_valida_en_la_funcion(df_100, semilla):
    with pytest.raises(ValueError, match="semilla"):
        separar_train_test(df_100, 20, semilla)


# --- Conversión de los textos de la interfaz ---

def test_parsear_valores_por_defecto():
    assert parsear_parametros_separacion("20", "42") == (20.0, 42)


def test_parsear_acepta_decimales_con_coma_o_punto_y_espacios():
    assert parsear_parametros_separacion(" 12,5 ", " 7 ") == (12.5, 7)
    assert parsear_parametros_separacion("12.5", "7") == (12.5, 7)


@pytest.mark.parametrize("texto", ["", "abc", "0", "100", "-3", "0.5", "99.5", "nan", "inf", "20%"])
def test_parsear_error_porcentaje_no_valido(texto):
    with pytest.raises(ValueError, match="entre 1 y 99"):
        parsear_parametros_separacion(texto, "42")


@pytest.mark.parametrize("texto", ["", "abc", "1.5", "4,2", "1e3"])
def test_parsear_error_semilla_no_entera(texto):
    with pytest.raises(ValueError, match="número entero"):
        parsear_parametros_separacion("20", texto)


def test_parsear_error_semilla_negativa():
    with pytest.raises(ValueError, match="no negativo"):
        parsear_parametros_separacion("20", "-5")


# ---------------------------------------------------------------------------
# Detección y tratamiento de valores inexistentes
# ---------------------------------------------------------------------------

@pytest.fixture
def conjuntos():
    # Tres columnas seleccionadas (x1, x2, y) y una no seleccionada (otra) con NaN
    train = pd.DataFrame({
        "x1":   [1.0, 2.0, np.nan, 4.0, 5.0],
        "x2":   [10.0, np.nan, 30.0, 40.0, np.nan],
        "y":    [1.0, 2.0, 3.0, np.nan, 5.0],
        "otra": [np.nan, 1.0, 2.0, 3.0, 4.0],
    })
    test = pd.DataFrame({
        "x1":   [100.0, np.nan, 300.0],
        "x2":   [np.nan, 200.0, 300.0],
        "y":    [7.0, 8.0, np.nan],
        "otra": [1.0, 2.0, np.nan],
    })
    return train, test

COLS = ["x1", "x2", "y"]


# --- Detección ---

def test_detectar_por_columna_y_por_conjunto(conjuntos):
    train, test = conjuntos
    r = detectar_valores_inexistentes(train, test, COLS)
    assert r["entrenamiento"] == {"x1": 1, "x2": 2, "y": 1}
    assert r["test"] == {"x1": 1, "x2": 1, "y": 1}


def test_detectar_ignora_columnas_no_seleccionadas(conjuntos):
    train, test = conjuntos
    r = detectar_valores_inexistentes(train, test, COLS)
    assert "otra" not in r["entrenamiento"] and "otra" not in r["test"]


def test_detectar_sin_valores_inexistentes():
    train = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    test = pd.DataFrame({"x": [5.0], "y": [6.0]})
    r = detectar_valores_inexistentes(train, test, ["x", "y"])
    assert sum(r["entrenamiento"].values()) == 0
    assert sum(r["test"].values()) == 0
    assert hay_valores_inexistentes(train, test, ["x", "y"]) is False


def test_hay_valores_inexistentes_solo_en_test(conjuntos):
    train, test = conjuntos
    train_ok = train.fillna(0)
    assert hay_valores_inexistentes(train_ok, test, COLS) is True
    assert hay_valores_inexistentes(train_ok, test.fillna(0), COLS) is False


def test_hay_valores_inexistentes_ignora_columnas_no_seleccionadas(conjuntos):
    train, test = conjuntos
    assert hay_valores_inexistentes(train.fillna({"x1": 0, "x2": 0, "y": 0}),
                                    test.fillna({"x1": 0, "x2": 0, "y": 0}), COLS) is False


# --- Estrategia: eliminar filas ---

def test_eliminar_filas_en_ambos_conjuntos():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, np.nan, 4.0]})
    test = pd.DataFrame({"x": [np.nan, 6.0, 7.0], "y": [5.0, 6.0, 7.0]})
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, ["x", "y"], "eliminar")
    assert list(train_c.index) == [0, 3]
    assert list(test_c.index) == [1, 2]
    assert resumen["filas_eliminadas_train"] == 2
    assert resumen["filas_eliminadas_test"] == 1
    assert not hay_valores_inexistentes(train_c, test_c, ["x", "y"])


def test_eliminar_solo_mira_las_columnas_seleccionadas():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "otra": [np.nan, 1.0, 2.0]})
    test = pd.DataFrame({"x": [4.0, 5.0], "otra": [np.nan, np.nan]})
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, ["x"], "eliminar")
    assert len(train_c) == 3 and len(test_c) == 2     # no se elimina ninguna fila
    assert resumen["filas_eliminadas_train"] == 0 and resumen["filas_eliminadas_test"] == 0


def test_eliminar_error_si_queda_vacio_train():
    train = pd.DataFrame({"x": [np.nan, np.nan]})
    test = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ConjuntoVacioError, match="vacío"):
        tratar_valores_inexistentes(train, test, ["x"], "eliminar")


def test_eliminar_error_si_queda_vacio_test():
    train = pd.DataFrame({"x": [1.0, 2.0]})
    test = pd.DataFrame({"x": [np.nan, np.nan]})
    with pytest.raises(ConjuntoVacioError, match="vacío"):
        tratar_valores_inexistentes(train, test, ["x"], "eliminar")


def test_conjunto_vacio_error_es_un_value_error():
    assert issubclass(ConjuntoVacioError, ValueError)


# --- Estrategia: media ---

def test_media_calculada_solo_con_train_y_aplicada_a_ambos(conjuntos):
    train, test = conjuntos
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, COLS, "media")
    # Medias de TRAIN: x1 = (1+2+4+5)/4 = 3, x2 = (10+30+40)/3 = 26.666..., y = (1+2+3+5)/4 = 2.75
    assert train_c.loc[2, "x1"] == pytest.approx(3.0)
    assert train_c.loc[1, "x2"] == pytest.approx(80 / 3)
    assert train_c.loc[3, "y"] == pytest.approx(2.75)
    # El test se rellena con las medias de TRAIN, no con las suyas (x1 en test: 200)
    assert test_c.loc[1, "x1"] == pytest.approx(3.0)
    assert test_c.loc[0, "x2"] == pytest.approx(80 / 3)
    assert test_c.loc[2, "y"] == pytest.approx(2.75)


def test_media_no_depende_de_los_valores_del_test():
    train = pd.DataFrame({"x": [1.0, 3.0, np.nan]})
    test_a = pd.DataFrame({"x": [np.nan, 1000.0]})
    test_b = pd.DataFrame({"x": [np.nan, -1000.0]})
    _, test_a_c, _ = tratar_valores_inexistentes(train, test_a, ["x"], "media")
    _, test_b_c, _ = tratar_valores_inexistentes(train, test_b, ["x"], "media")
    assert test_a_c.loc[0, "x"] == test_b_c.loc[0, "x"] == pytest.approx(2.0)


def test_media_resumen_y_sin_valores_pendientes(conjuntos):
    train, test = conjuntos
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, COLS, "media")
    assert resumen["valores_rellenados_train"] == 4     # 1 + 2 + 1
    assert resumen["valores_rellenados_test"] == 3      # 1 + 1 + 1
    assert resumen["filas_eliminadas_train"] == 0 and resumen["filas_eliminadas_test"] == 0
    assert not hay_valores_inexistentes(train_c, test_c, COLS)
    assert len(train_c) == len(train) and len(test_c) == len(test)


def test_media_no_toca_columnas_no_seleccionadas(conjuntos):
    train, test = conjuntos
    train_c, test_c, _ = tratar_valores_inexistentes(train, test, COLS, "media")
    assert train_c["otra"].isna().sum() == 1
    assert test_c["otra"].isna().sum() == 1


# --- Estrategia: mediana ---

def test_mediana_calculada_solo_con_train_y_aplicada_a_ambos(conjuntos):
    train, test = conjuntos
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, COLS, "mediana")
    # Medianas de TRAIN: x1 = mediana(1,2,4,5) = 3, x2 = mediana(10,30,40) = 30, y = mediana(1,2,3,5) = 2.5
    assert train_c.loc[2, "x1"] == pytest.approx(3.0)
    assert train_c.loc[1, "x2"] == pytest.approx(30.0)
    assert train_c.loc[3, "y"] == pytest.approx(2.5)
    assert test_c.loc[1, "x1"] == pytest.approx(3.0)
    assert test_c.loc[0, "x2"] == pytest.approx(30.0)
    assert test_c.loc[2, "y"] == pytest.approx(2.5)
    assert resumen["valores_rellenados_train"] == 4
    assert resumen["valores_rellenados_test"] == 3
    assert not hay_valores_inexistentes(train_c, test_c, COLS)


def test_mediana_difiere_de_la_media_cuando_hay_valores_extremos():
    train = pd.DataFrame({"x": [1.0, 2.0, 100.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan]})
    _, test_media, _ = tratar_valores_inexistentes(train, test, ["x"], "media")
    _, test_mediana, _ = tratar_valores_inexistentes(train, test, ["x"], "mediana")
    assert test_media.loc[0, "x"] == pytest.approx(103 / 3)
    assert test_mediana.loc[0, "x"] == pytest.approx(2.0)


@pytest.mark.parametrize("estrategia", ["media", "mediana"])
def test_error_columna_no_numerica_en_media_o_mediana(estrategia):
    train = pd.DataFrame({"cat": ["a", "b", None]})
    test = pd.DataFrame({"cat": ["a"]})
    with pytest.raises(ValueError, match="no es numérica"):
        tratar_valores_inexistentes(train, test, ["cat"], estrategia)


@pytest.mark.parametrize("estrategia", ["media", "mediana"])
def test_error_columna_sin_valores_en_train_para_media_o_mediana(estrategia):
    train = pd.DataFrame({"x": [np.nan, np.nan]})
    test = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ValueError, match="entrenamiento"):
        tratar_valores_inexistentes(train, test, ["x"], estrategia)


# --- Estrategia: valor constante ---

def test_constante_rellena_ambos_conjuntos(conjuntos):
    train, test = conjuntos
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, COLS, "constante", 0)
    assert train_c.loc[2, "x1"] == 0 and train_c.loc[1, "x2"] == 0 and train_c.loc[3, "y"] == 0
    assert test_c.loc[1, "x1"] == 0 and test_c.loc[0, "x2"] == 0 and test_c.loc[2, "y"] == 0
    assert resumen["valores_rellenados_train"] == 4
    assert resumen["valores_rellenados_test"] == 3
    assert not hay_valores_inexistentes(train_c, test_c, COLS)


def test_constante_no_toca_columnas_no_seleccionadas(conjuntos):
    train, test = conjuntos
    train_c, test_c, _ = tratar_valores_inexistentes(train, test, COLS, "constante", -1)
    assert train_c["otra"].isna().sum() == 1
    assert test_c["otra"].isna().sum() == 1


@pytest.mark.parametrize("valor, esperado", [("7", 7.0), ("-2.5", -2.5), ("1,5", 1.5), (" 3 ", 3.0), (4, 4.0)])
def test_constante_acepta_numeros_en_texto_o_numericos(valor, esperado):
    train = pd.DataFrame({"x": [1.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan]})
    train_c, test_c, _ = tratar_valores_inexistentes(train, test, ["x"], "constante", valor)
    assert train_c.loc[1, "x"] == pytest.approx(esperado)
    assert test_c.loc[0, "x"] == pytest.approx(esperado)


@pytest.mark.parametrize("valor", ["abc", "", "  ", "1,2,3", "nan", "inf", "-inf", True])
def test_constante_error_si_no_es_numerica(valor):
    train = pd.DataFrame({"x": [1.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan]})
    with pytest.raises(ValueError, match="número válido"):
        tratar_valores_inexistentes(train, test, ["x"], "constante", valor)


def test_constante_error_si_no_se_proporciona():
    train = pd.DataFrame({"x": [1.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan]})
    with pytest.raises(ValueError, match="valor constante"):
        tratar_valores_inexistentes(train, test, ["x"], "constante", None)


def test_parsear_valor_constante():
    assert parsear_valor_constante("0") == 0.0
    assert parsear_valor_constante("2,75") == 2.75
    with pytest.raises(ValueError, match="número válido"):
        parsear_valor_constante("texto")


# --- Comunes a todas las estrategias ---

def test_estrategia_desconocida(conjuntos):
    train, test = conjuntos
    with pytest.raises(ValueError, match="no reconocida"):
        tratar_valores_inexistentes(train, test, COLS, "inventada")


@pytest.mark.parametrize("estrategia, constante", [
    ("eliminar", None), ("media", None), ("mediana", None), ("constante", 0),
])
def test_no_modifica_los_conjuntos_originales(estrategia, constante):
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, np.nan, 4.0]})
    test = pd.DataFrame({"x": [np.nan, 6.0, 7.0], "y": [5.0, 6.0, 7.0]})
    train_antes, test_antes = train.copy(), test.copy()
    tratar_valores_inexistentes(train, test, ["x", "y"], estrategia, constante)
    pd.testing.assert_frame_equal(train, train_antes)
    pd.testing.assert_frame_equal(test, test_antes)


@pytest.mark.parametrize("estrategia, constante", [
    ("eliminar", None), ("media", None), ("mediana", None), ("constante", 0),
])
def test_sin_valores_inexistentes_no_cambia_nada(estrategia, constante):
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    test = pd.DataFrame({"x": [7.0, 8.0], "y": [9.0, 10.0]})
    train_c, test_c, resumen = tratar_valores_inexistentes(train, test, ["x", "y"], estrategia, constante)
    pd.testing.assert_frame_equal(train_c, train)
    pd.testing.assert_frame_equal(test_c, test)
    assert all(v == 0 for v in resumen.values())