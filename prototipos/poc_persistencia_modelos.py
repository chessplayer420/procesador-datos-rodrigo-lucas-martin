import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def crear_datos_y_modelo():
    """Genera datos de ejemplo y entrena un modelo de regresión lineal."""
    X_train = pd.DataFrame({
        "superficie": [50.0, 70.0, 90.0, 120.0, 150.0],
        "habitaciones": [1, 2, 3, 3, 4],
    })
    y_train = pd.Series(
        [150000.0, 200000.0, 250000.0, 320000.0, 400000.0],
        name="precio",
    )

    X_test = pd.DataFrame({
        "superficie": [60.0, 100.0],
        "habitaciones": [2, 3],
    })
    y_test = pd.Series([175000.0, 275000.0], name="precio")

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    metricas = {
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "ecm_train": float(mean_squared_error(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "ecm_test": float(mean_squared_error(y_test, y_pred_test)),
    }

    partes_formula = [f"{y_train.name} ="]
    for i, (col, coef) in enumerate(zip(X_train.columns, modelo.coef_)):
        c = round(float(coef), 4)
        if i == 0:
            partes_formula.append(f"{c} * {col}")
        elif c >= 0:
            partes_formula.append(f"+ {c} * {col}")
        else:
            partes_formula.append(f"- {abs(c)} * {col}")
    b = round(float(modelo.intercept_), 4)
    partes_formula.append(f"+ {b}" if b >= 0 else f"- {abs(b)}")
    formula = " ".join(partes_formula)

    metadatos = {
        "features": list(X_train.columns),
        "target": str(y_train.name),
        "formula": formula,
        "descripcion": "Modelo de prueba para predicción de precios de viviendas",
        "metricas": metricas,
    }

    return modelo, metadatos, X_test, y_test


def prueba_pickle(modelo, metadatos, X_test, ruta: str | Path):
    print("--- PRUEBA CON PICKLE ---")
    contenido_completo = {
        "modelo": modelo,
        "metadatos": metadatos,
    }

    with open(ruta, "wb") as f:
        pickle.dump(contenido_completo, f)

    tamano = os.path.getsize(ruta)
    print(f"Archivo guardado correctamente: {ruta} ({tamano} bytes)")

    with open(ruta, "rb") as f:
        datos_cargados = pickle.load(f)

    modelo_recuperado = datos_cargados["modelo"]
    print(f"Descripción recuperada: {datos_cargados['metadatos']['descripcion']}")
    print(f"Fórmula recuperada: {datos_cargados['metadatos']['formula']}")
    print(f"Métricas recuperadas: {datos_cargados['metadatos']['metricas']}")
    print(f"Coeficientes recuperados: {modelo_recuperado.coef_}")
    assert np.allclose(modelo.predict(X_test), modelo_recuperado.predict(X_test))
    print("Predicciones coincidentes con el modelo original.\n")


def prueba_json(modelo, metadatos, X_test, ruta: str | Path):
    print("--- PRUEBA CON JSON ---")
    datos_json = {
        "coeficientes": modelo.coef_.tolist(),
        "intercepto": float(modelo.intercept_),
        "features": metadatos["features"],
        "target": metadatos["target"],
        "formula": metadatos["formula"],
        "descripcion": metadatos["descripcion"],
        "metricas": metadatos["metricas"],
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos_json, f, indent=4, ensure_ascii=False)

    tamano = os.path.getsize(ruta)
    print(f"Archivo guardado correctamente: {ruta} ({tamano} bytes)")

    with open(ruta, "r", encoding="utf-8") as f:
        datos_cargados = json.load(f)

    modelo_reconstruido = LinearRegression()
    modelo_reconstruido.coef_ = np.array(datos_cargados["coeficientes"], dtype=float)
    modelo_reconstruido.intercept_ = float(datos_cargados["intercepto"])

    print(f"Descripción recuperada: {datos_cargados['descripcion']}")
    print(f"Fórmula recuperada: {datos_cargados['formula']}")
    print(f"Métricas recuperadas: {datos_cargados['metricas']}")
    print(f"Coeficientes reconstruidos: {modelo_reconstruido.coef_}")
    assert np.allclose(modelo.predict(X_test), modelo_reconstruido.predict(X_test))
    print("Predicciones coincidentes con el modelo original.\n")


class ObjetoConReduceInofensivo:
    """Al hacer pickle.load(), __reduce__ ejecuta este callable sin llamarlo a mano."""

    def __reduce__(self):
        return (print, ("[AVISO SEGURIDAD PICKLE] Ejecución automática al deserializar!",))


def prueba_seguridad_pickle(ruta: str | Path):
    print("--- PRUEBA DE SEGURIDAD CON PICKLE ---")
    with open(ruta, "wb") as f:
        pickle.dump(ObjetoConReduceInofensivo(), f)

    with open(ruta, "rb") as f:
        _ = pickle.load(f)


if __name__ == "__main__":
    modelo, metadatos, X_test, _ = crear_datos_y_modelo()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        prueba_pickle(modelo, metadatos, X_test, tmp_path / "modelo_prueba.pkl")
        prueba_json(modelo, metadatos, X_test, tmp_path / "modelo_prueba.json")
        prueba_seguridad_pickle(tmp_path / "payload_seguridad.pkl")