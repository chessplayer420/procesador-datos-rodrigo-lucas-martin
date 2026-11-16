import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any

EXTENSION_MODELO = ".json"


def entrenar_modelo_regresion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Entrena la regresión lineal únicamente con el conjunto de entrenamiento.
    Calcula métricas R2 y ECM para train y test, y devuelve la fórmula formateada.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("El conjunto de entrenamiento está vacío.")

    if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
        raise ValueError("Quedan valores inexistentes en el conjunto de entrenamiento.")

    if X_test.isna().sum().sum() > 0 or y_test.isna().sum() > 0:
        raise ValueError("Quedan valores inexistentes en el conjunto de test.")

    # 1. Ajuste del modelo solo en TRAIN
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 2. Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # 3. Métricas
    r2_train = float(r2_score(y_train, y_pred_train))
    r2_test = float(r2_score(y_test, y_pred_test))
    ecm_train = float(mean_squared_error(y_train, y_pred_train))
    ecm_test = float(mean_squared_error(y_test, y_pred_test))

    # 4. Texto de la fórmula y metadatos de columnas
    features = [str(c) for c in X_train.columns.tolist()]
    target = str(y_train.name) if y_train.name else "y"
    coeficientes = [float(c) for c in np.ravel(modelo.coef_)]
    intercepto = float(modelo.intercept_)

    formula = generar_formula_texto(target, features, modelo.coef_, intercepto)

    return {
        "modelo": modelo,
        "coeficientes": coeficientes,
        "intercepto": intercepto,
        "features": features,
        "columnas_entrada": features,
        "target": target,
        "columna_salida": target,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "formula": formula,
        "descripcion": "",
        "metricas": {
            "r2_train": r2_train,
            "r2_test": r2_test,
            "ecm_train": ecm_train,
            "ecm_test": ecm_test
        }
    }


def generar_formula_texto(
    nombre_salida: str,
    nombres_entrada: list,
    coeficientes: np.ndarray,
    termino_independiente: float,
    decimales: int = 4
) -> str:
    """Genera la fórmula en texto omitiendo expresiones del tipo + -."""
    partes = [f"{nombre_salida} ="]

    for i, (col, coef) in enumerate(zip(nombres_entrada, coeficientes)):
        coef_red = round(float(coef), decimales)
        if i == 0:
            partes.append(f"{coef_red} * {col}")
        else:
            if coef_red >= 0:
                partes.append(f"+ {coef_red} * {col}")
            else:
                partes.append(f"- {abs(coef_red)} * {col}")

    intercepto_red = round(float(termino_independiente), decimales)
    if intercepto_red >= 0:
        partes.append(f"+ {intercepto_red}")
    else:
        partes.append(f"- {abs(intercepto_red)}")

    return " ".join(partes)


def actualizar_descripcion_modelo(estructura_modelo: dict, nueva_descripcion: str) -> dict:
    """Almacena o actualiza la descripción en la estructura de datos del modelo."""
    if not isinstance(estructura_modelo, dict):
        raise ValueError("No hay una estructura de modelo válida.")
    estructura_modelo["descripcion"] = "" if nueva_descripcion is None else str(nueva_descripcion)
    return estructura_modelo


def normalizar_ruta_guardado(ruta_archivo: str | Path) -> Path:
    """Añade automáticamente la extensión .json si el usuario no la escribe."""
    if not ruta_archivo or not str(ruta_archivo).strip():
        raise ValueError("La ruta del archivo no puede estar vacía.")
    ruta = Path(str(ruta_archivo).strip())
    if ruta.suffix.lower() != EXTENSION_MODELO:
        ruta = ruta.with_suffix(EXTENSION_MODELO)
    return ruta


def guardar_modelo(estructura_modelo: dict, ruta_archivo: str | Path) -> str:
    """
    Guarda en formato JSON la información del modelo necesaria para mostrarlo y hacer
    predicciones, sin incluir datos del dataset original ni gráficos.
    Devuelve la ruta final del archivo guardado como cadena.
    """
    if not isinstance(estructura_modelo, dict) or not estructura_modelo:
        raise ValueError("No hay ningún modelo válido para guardar.")

    ruta_final = normalizar_ruta_guardado(ruta_archivo)

    # Extraer coeficientes e intercepto
    if "coeficientes" in estructura_modelo and estructura_modelo["coeficientes"] is not None:
        coeficientes = [float(c) for c in estructura_modelo["coeficientes"]]
    elif "modelo" in estructura_modelo and hasattr(estructura_modelo["modelo"], "coef_"):
        coeficientes = [float(c) for c in np.ravel(estructura_modelo["modelo"].coef_)]
    else:
        raise ValueError("El modelo no contiene coeficientes válidos.")

    if "intercepto" in estructura_modelo and estructura_modelo["intercepto"] is not None:
        intercepto = float(estructura_modelo["intercepto"])
    elif "modelo" in estructura_modelo and hasattr(estructura_modelo["modelo"], "intercept_"):
        intercepto = float(estructura_modelo["modelo"].intercept_)
    else:
        raise ValueError("El modelo no contiene término independiente válido.")

    features = list(
        estructura_modelo.get("features")
        or estructura_modelo.get("columnas_entrada")
        or []
    )
    target = str(
        estructura_modelo.get("target")
        or estructura_modelo.get("columna_salida")
        or ""
    )
    if not features or not target:
        raise ValueError("El modelo debe incluir los nombres de las columnas de entrada y de salida.")

    formula = str(estructura_modelo.get("formula") or "")
    if not formula:
        formula = generar_formula_texto(target, features, np.array(coeficientes), intercepto)

    metricas_raw = estructura_modelo.get("metricas") or {}
    metricas = {
        "r2_train": float(metricas_raw["r2_train"]),
        "r2_test": float(metricas_raw["r2_test"]),
        "ecm_train": float(metricas_raw["ecm_train"]),
        "ecm_test": float(metricas_raw["ecm_test"]),
    }

    descripcion = str(estructura_modelo.get("descripcion") or "")

    # Estructura documentada en la historia #20 (sin dataset ni gráficos)
    datos_json = {
        "coeficientes": coeficientes,
        "intercepto": intercepto,
        "formula": formula,
        "features": features,
        "target": target,
        "descripcion": descripcion,
        "metricas": metricas,
    }

    with open(ruta_final, "w", encoding="utf-8") as archivo:
        json.dump(datos_json, archivo, indent=4, ensure_ascii=False)

    return str(ruta_final)