import pytest
import pandas as pd
import numpy as np
from modelo import entrenar_modelo_regresion, generar_formula_texto
from modelo import entrenar_modelo_regresion, actualizar_descripcion_modelo



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