import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any


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

    # 4. Texto de la fórmula
    formula = generar_formula_texto(y_train.name, X_train.columns.tolist(), modelo.coef_, modelo.intercept_)

    return {
        "modelo": modelo,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "formula": formula,
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