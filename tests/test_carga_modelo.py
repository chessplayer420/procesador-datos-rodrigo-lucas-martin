"""
Pruebas unitarias de la Historia #115: cargar un modelo guardado previamente.

Los modelos se entrenan con datos creados en este mismo archivo y todos los
ficheros se escriben en carpetas temporales (tmp_path), sin añadir nada al repositorio.
"""
import json

import numpy as np
import pandas as pd
import pytest

try:
    from src.modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
        cargar_modelo,
        predecir_salida,
        ModeloInvalidoError,
    )
except ImportError:
    from modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
        cargar_modelo,
        predecir_salida,
        ModeloInvalidoError,
    )


FEATURES = ["superficie", "habitaciones"]
TARGET = "precio"


@pytest.fixture
def datos():
    """Datos sintéticos (train/test) con una relación lineal más algo de ruido."""
    rng = np.random.default_rng(42)
    n = 60
    X = pd.DataFrame({
        "superficie": rng.uniform(40, 200, n),
        "habitaciones": rng.integers(1, 6, n).astype(float),
    })
    y = pd.Series(
        1500 * X["superficie"] + 8000 * X["habitaciones"] + 20000 + rng.normal(0, 500, n),
        name=TARGET,
    )
    return X.iloc[:45], y.iloc[:45], X.iloc[45:], y.iloc[45:]


@pytest.fixture
def modelo_entrenado(datos):
    X_train, y_train, X_test, y_test = datos
    return entrenar_modelo_regresion(X_train, y_train, X_test, y_test)


# -------------------------------------------------------------
# 1. Guardar y cargar conserva toda la información
# -------------------------------------------------------------
@pytest.mark.parametrize(
    "descripcion",
    ["", "Modelo de precios: ajuste con áéíóú y la letra ñ."],
    ids=["descripcion_vacia", "descripcion_con_texto"],
)
def test_guardar_y_cargar_conserva_toda_la_informacion(tmp_path, datos, modelo_entrenado, descripcion):
    _, _, X_test, _ = datos
    modelo_entrenado["descripcion"] = descripcion

    ruta = guardar_modelo(modelo_entrenado, tmp_path / "modelo.json")
    cargado = cargar_modelo(ruta)

    assert cargado["features"] == modelo_entrenado["features"]
    assert cargado["target"] == modelo_entrenado["target"]
    assert cargado["formula"] == modelo_entrenado["formula"]
    assert cargado["metricas"] == modelo_entrenado["metricas"]
    assert cargado["descripcion"] == descripcion
    np.testing.assert_array_equal(cargado["modelo"].coef_, modelo_entrenado["modelo"].coef_)
    assert cargado["modelo"].intercept_ == modelo_entrenado["modelo"].intercept_

    # Mismas predicciones que el modelo original
    np.testing.assert_allclose(
        cargado["modelo"].predict(X_test),
        modelo_entrenado["modelo"].predict(X_test),
    )
    valores = {"superficie": "85,5", "habitaciones": "3"}
    assert predecir_salida(cargado["modelo"], cargado["features"], valores) == predecir_salida(
        modelo_entrenado["modelo"], modelo_entrenado["features"], valores
    )


def test_carga_desde_carpeta_con_espacios_tildes_y_enie(tmp_path, modelo_entrenado):
    carpeta = tmp_path / "Mis modelos del año ñandú áéí"
    ruta = guardar_modelo(modelo_entrenado, carpeta / "modelo regresión ñ.json")

    cargado = cargar_modelo(ruta)

    assert cargado["formula"] == modelo_entrenado["formula"]
    assert cargado["features"] == FEATURES


# -------------------------------------------------------------
# 2. Modificar la descripción de un modelo cargado y volver a guardar
# -------------------------------------------------------------
def test_modificar_descripcion_y_guardar_conserva_el_resto(tmp_path, modelo_entrenado):
    modelo_entrenado["descripcion"] = "Descripción original"
    ruta_original = guardar_modelo(modelo_entrenado, tmp_path / "original.json")

    cargado = cargar_modelo(ruta_original)
    actualizar_descripcion_modelo(cargado, "Descripción modificada tras la carga")

    # Se guarda sobre el mismo archivo del que se cargó
    ruta_guardada = guardar_modelo(cargado, ruta_original)
    recargado = cargar_modelo(ruta_guardada)

    assert recargado["descripcion"] == "Descripción modificada tras la carga"
    assert recargado["features"] == modelo_entrenado["features"]
    assert recargado["target"] == modelo_entrenado["target"]
    assert recargado["formula"] == modelo_entrenado["formula"]
    assert recargado["metricas"] == modelo_entrenado["metricas"]
    np.testing.assert_array_equal(recargado["modelo"].coef_, modelo_entrenado["modelo"].coef_)
    assert recargado["modelo"].intercept_ == modelo_entrenado["modelo"].intercept_


def test_modificar_descripcion_y_guardar_en_otro_archivo(tmp_path, modelo_entrenado):
    ruta_original = guardar_modelo(modelo_entrenado, tmp_path / "original.json")

    cargado = cargar_modelo(ruta_original)
    actualizar_descripcion_modelo(cargado, "Solo cambia la descripción")
    ruta_copia = guardar_modelo(cargado, tmp_path / "copia.json")

    # El archivo original no se ha tocado
    assert cargar_modelo(ruta_original)["descripcion"] == ""
    copia = cargar_modelo(ruta_copia)
    assert copia["descripcion"] == "Solo cambia la descripción"
    assert copia["metricas"] == modelo_entrenado["metricas"]


# -------------------------------------------------------------
# 3. Archivos no válidos producen un error controlado
# -------------------------------------------------------------
@pytest.fixture
def json_valido(tmp_path, modelo_entrenado):
    """Contenido (dict) de un archivo de modelo correcto, para estropearlo en cada caso."""
    ruta = guardar_modelo(modelo_entrenado, tmp_path / "valido.json")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _escribir_json(ruta, contenido):
    ruta.write_text(json.dumps(contenido), encoding="utf-8")
    return ruta


@pytest.mark.parametrize(
    "contenido_texto",
    ["", "esto no es json {", "[1, 2, 3]", '"solo un texto"', "null"],
    ids=["vacio", "no_es_json", "lista", "cadena", "null"],
)
def test_archivo_con_formato_incorrecto_lanza_error_controlado(tmp_path, contenido_texto):
    ruta = tmp_path / "malo.json"
    ruta.write_text(contenido_texto, encoding="utf-8")

    with pytest.raises(ModeloInvalidoError):
        cargar_modelo(ruta)


def test_archivo_binario_lanza_error_controlado(tmp_path):
    ruta = tmp_path / "binario.json"
    ruta.write_bytes(b"\xff\xfe\x00\x80\x81\x82")

    with pytest.raises(ModeloInvalidoError):
        cargar_modelo(ruta)


@pytest.mark.parametrize(
    "campo_a_quitar",
    ["features", "target", "coeficientes", "intercepto", "formula", "metricas"],
)
def test_archivo_al_que_le_falta_informacion_lanza_error_controlado(tmp_path, json_valido, campo_a_quitar):
    del json_valido[campo_a_quitar]
    ruta = _escribir_json(tmp_path / "incompleto.json", json_valido)

    with pytest.raises(ModeloInvalidoError):
        cargar_modelo(ruta)


def test_coeficientes_que_no_coinciden_con_las_columnas_lanza_error_controlado(tmp_path, json_valido):
    json_valido["coeficientes"] = json_valido["coeficientes"][:-1]
    ruta = _escribir_json(tmp_path / "coef.json", json_valido)

    with pytest.raises(ModeloInvalidoError):
        cargar_modelo(ruta)


def test_metricas_incompletas_lanza_error_controlado(tmp_path, json_valido):
    del json_valido["metricas"]["r2_test"]
    ruta = _escribir_json(tmp_path / "metricas.json", json_valido)

    with pytest.raises(ModeloInvalidoError):
        cargar_modelo(ruta)


def test_valores_con_tipo_incorrecto_lanza_error_controlado(tmp_path, json_valido):
    json_valido["intercepto"] = "no es un número"
    ruta = _escribir_json(tmp_path / "tipo.json", json_valido)

    with pytest.raises(ModeloInvalidoError):
        cargar_modelo(ruta)


def test_archivo_inexistente_lanza_error_de_archivo(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_modelo(tmp_path / "no_existe.json")


def test_modelo_sin_descripcion_en_el_archivo_se_carga_con_descripcion_vacia(tmp_path, json_valido):
    del json_valido["descripcion"]
    ruta = _escribir_json(tmp_path / "sin_descripcion.json", json_valido)

    assert cargar_modelo(ruta)["descripcion"] == ""