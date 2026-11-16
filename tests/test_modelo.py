import pytest
import pandas as pd
import numpy as np
from modelo import entrenar_modelo_regresion, generar_formula_texto
from modelo import entrenar_modelo_regresion, actualizar_descripcion_modelo
import json





def test_entrenar_modelo_regresion_simple():
    X_tr = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]})
    y_tr = pd.Series([2.0, 4.0, 6.0, 8.0], name="Y")
    X_te = pd.DataFrame({"X": [5.0, 6.0]})
    y_te = pd.Series([10.0, 12.0], name="Y")

    res = entrenar_modelo_regresion(X_tr, y_tr, X_te, y_te)

    assert "modelo" in res
    assert res["metricas"]["r2_train"] == pytest.approx(1.0)
    assert res["metricas"]["ecm_train"] == pytest.approx(0.0)


def test_formula_coeficiente_negativo():
    # Comprobar que no genera expresiones del tipo + -
    formula = generar_formula_texto(
        nombre_salida="Precio",
        nombres_entrada=["Edad", "Km"],
        coeficientes=np.array([-150.5, 0.02]),
        termino_independiente=-500.25
    )

    assert "+ -" not in formula
    assert "Precio = -150.5 * Edad + 0.02 * Km - 500.25" == formula


def test_error_si_quedan_nulos():
    X_tr = pd.DataFrame({"X": [1.0, np.nan]})
    y_tr = pd.Series([2.0, 4.0], name="Y")
    X_te = pd.DataFrame({"X": [5.0, 6.0]})
    y_te = pd.Series([10.0, 12.0], name="Y")

    with pytest.raises(ValueError, match="valores inexistentes"):
        entrenar_modelo_regresion(X_tr, y_tr, X_te, y_te)


def test_modelo_nuevo_se_crea_con_descripcion_vacia():
    X_tr = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_tr = pd.Series([2.0, 4.0, 6.0, 8.0], name="y")
    X_te = pd.DataFrame({"x": [5.0, 6.0]})
    y_te = pd.Series([10.0, 12.0], name="y")

    res = entrenar_modelo_regresion(X_tr, y_tr, X_te, y_te)
    assert "descripcion" in res
    assert res["descripcion"] == ""

def test_descripcion_se_almacena_en_estructura_datos_modelo():
    X_tr = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    y_tr = pd.Series([2.0, 4.0, 6.0, 8.0], name="y")
    X_te = pd.DataFrame({"x": [5.0, 6.0]})
    y_te = pd.Series([10.0, 12.0], name="y")

    res = entrenar_modelo_regresion(X_tr, y_tr, X_te, y_te)
    actualizar_descripcion_modelo(res, "Modelo de prueba")
    assert res["descripcion"] == "Modelo de prueba"


try:
    from src.modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
    )
except ImportError:
    from modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
    )


def _entrenar_modelo_ejemplo():
    X_train = pd.DataFrame({
        "superficie": [50.0, 70.0, 90.0, 120.0],
        "habitaciones": [1.0, 2.0, 3.0, 4.0],
    })
    y_train = pd.Series([150.0, 210.0, 270.0, 350.0], name="precio")
    X_test = pd.DataFrame({
        "superficie": [100.0, 140.0],
        "habitaciones": [3.0, 4.0],
    })
    y_test = pd.Series([290.0, 390.0], name="precio")
    return entrenar_modelo_regresion(X_train, y_train, X_test, y_test)


def test_guardar_modelo_contiene_toda_la_informacion_y_no_contiene_dataset(tmp_path):
    estructura = _entrenar_modelo_ejemplo()
    actualizar_descripcion_modelo(estructura, "Modelo de prueba con descripción")

    ruta_destino = tmp_path / "modelo_guardado.json"
    ruta_final = guardar_modelo(estructura, ruta_destino)

    with open(ruta_final, "r", encoding="utf-8") as f:
        datos = json.load(f)

    # 1. Comprobar que contiene toda la información exigida
    assert "coeficientes" in datos and len(datos["coeficientes"]) == 2
    assert "intercepto" in datos and isinstance(datos["intercepto"], float)
    assert "formula" in datos and "precio =" in datos["formula"]
    assert datos["features"] == ["superficie", "habitaciones"]
    assert datos["target"] == "precio"
    assert datos["descripcion"] == "Modelo de prueba con descripción"
    assert set(datos["metricas"].keys()) == {"r2_train", "r2_test", "ecm_train", "ecm_test"}

    # 2. Comprobar que NO contiene datos del dataset original ni predicciones ni gráficos
    claves_prohibidas = {"df_cargado", "df_train", "df_test", "X_train", "y_train", "X_test", "y_test", "y_pred_train", "y_pred_test", "modelo", "grafico"}
    assert claves_prohibidas.isdisjoint(set(datos.keys()))


def test_guardar_modelo_con_descripcion_vacia_y_extension_automatica(tmp_path):
    estructura = _entrenar_modelo_ejemplo()
    # Sin extensión .json y en una carpeta con espacios, tildes y letra ñ
    carpeta_especial = tmp_path / "Carpeta de Diseño y Análisis 2026"
    carpeta_especial.mkdir(parents=True, exist_ok=True)
    ruta_sin_extension = carpeta_especial / "mi_modelo_español"

    ruta_final = guardar_modelo(estructura, ruta_sin_extension)

    assert ruta_final.endswith(".json")
    with open(ruta_final, "r", encoding="utf-8") as f:
        datos = json.load(f)

    assert datos["descripcion"] == ""