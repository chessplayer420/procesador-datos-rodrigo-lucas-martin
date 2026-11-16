import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def entrenar_modelo_regresion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Entrena el modelo de regresión lineal únicamente con el conjunto de entrenamiento.
    Calcula métricas R² y ECM (MSE) para train y test, y devuelve la fórmula formateada.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("El conjunto de entrenamiento está vacío.")

    if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
        raise ValueError("Quedan valores inexistentes en el conjunto de entrenamiento.")

    if X_test.isna().sum().sum() > 0 or y_test.isna().sum() > 0:
        raise ValueError("Quedan valores inexistentes en el conjunto de test.")

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    r2_train = float(r2_score(y_train, y_pred_train))
    r2_test = float(r2_score(y_test, y_pred_test))
    ecm_train = float(mean_squared_error(y_train, y_pred_train))
    ecm_test = float(mean_squared_error(y_test, y_pred_test))

    formula = generar_formula_texto(
        y_train.name, X_train.columns.tolist(), modelo.coef_, modelo.intercept_
    )

    return {
        "modelo": modelo,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "formula": formula,
        "descripcion": "",
        "features": X_train.columns.tolist(),
        "target": str(y_train.name),
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
    """Genera el texto de la fórmula redondeado sin expresiones tipo + -."""
    partes = [f"{nombre_salida} ="]

    for i, (col, coef) in enumerate(zip(nombres_entrada, coeficientes)):
        coef_red = round(coef, decimales)
        if i == 0:
            partes.append(f"{coef_red} * {col}")
        else:
            if coef_red >= 0:
                partes.append(f"+ {coef_red} * {col}")
            else:
                partes.append(f"- {abs(coef_red)} * {col}")

    intercepto_red = round(termino_independiente, decimales)
    if intercepto_red >= 0:
        partes.append(f"+ {intercepto_red}")
    else:
        partes.append(f"- {abs(intercepto_red)}")

    return " ".join(partes)


def actualizar_descripcion_modelo(estructura_modelo: Dict[str, Any], descripcion: str) -> None:
    """Actualiza la descripción en la estructura del modelo."""
    if isinstance(estructura_modelo, dict):
        estructura_modelo["descripcion"] = descripcion


def normalizar_ruta_guardado(ruta: str | Path) -> Path:
    """Asegura que la ruta tenga extensión .json."""
    p = Path(ruta)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(p.suffix + ".json") if p.suffix else p.with_suffix(".json")
    return p


def guardar_modelo(estructura_modelo: Dict[str, Any], ruta: str | Path) -> str:
    """Guarda los parámetros, fórmula, métricas y descripción del modelo en JSON."""
    ruta_final = normalizar_ruta_guardado(ruta)
    modelo = estructura_modelo["modelo"]

    datos_serializados = {
        "features": estructura_modelo.get("features", []),
        "target": estructura_modelo.get("target", ""),
        "coeficientes": [float(c) for c in modelo.coef_],
        "intercepto": float(modelo.intercept_),
        "formula": estructura_modelo.get("formula", ""),
        "metricas": estructura_modelo.get("metricas", {}),
        "descripcion": estructura_modelo.get("descripcion", ""),
    }

    ruta_final.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_final, "w", encoding="utf-8") as f:
        json.dump(datos_serializados, f, indent=4, ensure_ascii=False)

    return str(ruta_final)


def predecir_salida(
    modelo: LinearRegression,
    nombres_entrada: list,
    valores_crudos: dict[str, str],
    decimales: int = 4
) -> float:
    """
    Valida los valores de entrada (admite punto o coma decimal), realiza la predicción
    con el modelo y devuelve el resultado redondeado al mismo número de decimales que la fórmula.
    """
    if modelo is None:
        raise ValueError("No hay ningún modelo disponible para realizar la predicción.")

    faltantes = []
    no_numericos = []
    valores_numericos = {}

    for col in nombres_entrada:
        val_str = str(valores_crudos.get(col, "")).strip()
        if not val_str:
            faltantes.append(col)
            continue

        val_normalizado = val_str.replace(",", ".")
        try:
            valores_numericos[col] = float(val_normalizado)
        except ValueError:
            no_numericos.append(col)

    errores = []
    if faltantes:
        errores.append(f"Faltan valores para las variables: {', '.join(faltantes)}.")
    if no_numericos:
        errores.append(f"Valores no numéricos en las variables: {', '.join(no_numericos)}.")

    if errores:
        raise ValueError("\n".join(errores))

    df_entrada = pd.DataFrame([valores_numericos], columns=nombres_entrada)
    prediccion = float(modelo.predict(df_entrada)[0])
    return round(prediccion, decimales)