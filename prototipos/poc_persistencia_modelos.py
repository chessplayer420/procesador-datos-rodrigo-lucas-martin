import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def crear_datos_y_modelo():
    """Genera datos de ejemplo y entrena un modelo de regresión lineal."""
    X_train = pd.DataFrame({
        "superficie": [50.0, 70.0, 90.0, 120.0, 150.0],
        "habitaciones": [1, 2, 3, 3, 4]
    })
    y_train = pd.Series([150000.0, 200000.0, 250000.0, 320000.0, 400000.0], name="precio")

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_train)
    metricas = {
        "r2_train": float(r2_score(y_train, y_pred)),
        "ecm_train": float(mean_squared_error(y_train, y_pred))
    }

    metadatos = {
        "features": list(X_train.columns),
        "target": y_train.name,
        "descripcion": "Modelo de prueba para predicción de precios de viviendas",
        "metricas": metricas
    }

    return modelo, metadatos


def prueba_pickle(modelo, metadatos, ruta="prototipos/modelo_prueba.pkl"):
    print("--- PRUEBA CON PICKLE ---")
    contenido_completo = {
        "modelo": modelo,
        "metadatos": metadatos
    }

    with open(ruta, "wb") as f:
        pickle.dump(contenido_completo, f)
    
    tamano = os.path.getsize(ruta)
    print(f"Archivo guardado correctamente: {ruta} ({tamano} bytes)")

    with open(ruta, "rb") as f:
        datos_cargados = pickle.load(f)

    modelo_recuperado = datos_cargados["modelo"]
    print(f"Descripción recuperada: {datos_cargados['metadatos']['descripcion']}")
    print(f"Coeficientes recuperados: {modelo_recuperado.coef_}\n")


def prueba_json(modelo, metadatos, ruta="prototipos/modelo_prueba.json"):
    print("--- PRUEBA CON JSON ---")
    datos_json = {
        "coeficientes": modelo.coef_.tolist(),
        "intercepto": float(modelo.intercept_),
        "features": metadatos["features"],
        "target": metadatos["target"],
        "descripcion": metadatos["descripcion"],
        "metricas": metadatos["metricas"]
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos_json, f, indent=4, ensure_ascii=False)

    tamano = os.path.getsize(ruta)
    print(f"Archivo guardado correctamente: {ruta} ({tamano} bytes)")

    with open(ruta, "r", encoding="utf-8") as f:
        datos_cargados = json.load(f)

    modelo_reconstruido = LinearRegression()
    modelo_reconstruido.coef_ = np.array(datos_cargados["coeficientes"])
    modelo_reconstruido.intercept_ = datos_cargados["intercepto"]

    print(f"Descripción recuperada: {datos_cargados['descripcion']}")
    print(f"Coeficientes reconstruidos: {modelo_reconstruido.coef_}\n")


if __name__ == "__main__":
    modelo, metadatos = crear_datos_y_modelo()
    os.makedirs("prototipos", exist_ok=True)
    
    prueba_pickle(modelo, metadatos)
    prueba_json(modelo, metadatos)