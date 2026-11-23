import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class ModeloInvalidoError(ValueError):
    """
    Se lanza al cargar un archivo de modelo que está dañado, no tiene el formato
    esperado o no contiene toda la información del modelo (Historia #115).
    """


# Campos que debe contener un archivo de modelo guardado por esta aplicación.
# La descripción es opcional: si falta se interpreta como vacía.
CLAVES_MODELO_REQUERIDAS = ("features", "target", "coeficientes", "intercepto", "formula", "metricas")
CLAVES_METRICAS_REQUERIDAS = ("r2_train", "r2_test", "ecm_train", "ecm_test")


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


def _es_numero(valor: Any) -> bool:
    """True si el valor es un número JSON (int o float), excluyendo booleanos."""
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def cargar_modelo(ruta: str | Path) -> Dict[str, Any]:
    """
    Carga un modelo guardado previamente con `guardar_modelo` y devuelve la misma
    estructura que produce `entrenar_modelo_regresion` (sin las predicciones, que
    no se almacenan): modelo listo para predecir, fórmula, descripción, columnas
    de entrada, columna de salida y métricas.

    Es independiente de la interfaz gráfica. Funciona con rutas en cualquier
    carpeta (incluidas las que tengan espacios, tildes o la letra ñ).

    Errores:
      - FileNotFoundError / PermissionError / IsADirectoryError: problemas de acceso al archivo.
      - ModeloInvalidoError (subclase de ValueError): archivo dañado, con formato
        distinto al esperado o al que le falta información del modelo.
    """
    ruta_archivo = Path(ruta)

    try:
        # utf-8-sig tolera archivos con y sin marca BOM (p. ej. editados con el Bloc de notas)
        with open(ruta_archivo, "r", encoding="utf-8-sig") as archivo:
            datos = json.load(archivo)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise ModeloInvalidoError(
            "El archivo seleccionado no tiene un formato de modelo válido o está dañado.\n"
            "Asegúrate de elegir un archivo .json guardado desde esta aplicación."
        ) from err

    if not isinstance(datos, dict):
        raise ModeloInvalidoError(
            "El archivo seleccionado no tiene el formato de un modelo guardado por esta aplicación."
        )

    faltantes = [clave for clave in CLAVES_MODELO_REQUERIDAS if clave not in datos]
    if faltantes:
        raise ModeloInvalidoError(
            "Al archivo le falta información del modelo "
            f"({', '.join(faltantes)}), por lo que no se puede cargar."
        )

    features = datos["features"]
    if not isinstance(features, list) or not features or not all(isinstance(c, str) for c in features):
        raise ModeloInvalidoError("Las columnas de entrada guardadas en el archivo no son válidas.")

    target = datos["target"]
    if not isinstance(target, str) or not target:
        raise ModeloInvalidoError("La columna de salida guardada en el archivo no es válida.")

    coeficientes = datos["coeficientes"]
    if (
        not isinstance(coeficientes, list)
        or len(coeficientes) != len(features)
        or not all(_es_numero(c) for c in coeficientes)
    ):
        raise ModeloInvalidoError(
            "Los coeficientes guardados en el archivo no son válidos o no coinciden "
            "con las columnas de entrada."
        )

    intercepto = datos["intercepto"]
    if not _es_numero(intercepto):
        raise ModeloInvalidoError("El término independiente guardado en el archivo no es válido.")

    formula = datos["formula"]
    if not isinstance(formula, str):
        raise ModeloInvalidoError("La fórmula guardada en el archivo no es válida.")

    metricas = datos["metricas"]
    if not isinstance(metricas, dict):
        raise ModeloInvalidoError("Las métricas guardadas en el archivo no son válidas.")
    metricas_faltantes = [c for c in CLAVES_METRICAS_REQUERIDAS if c not in metricas]
    if metricas_faltantes:
        raise ModeloInvalidoError(
            "A las métricas del archivo les falta información "
            f"({', '.join(metricas_faltantes)}), por lo que no se puede cargar el modelo."
        )
    if not all(_es_numero(metricas[c]) for c in CLAVES_METRICAS_REQUERIDAS):
        raise ModeloInvalidoError("Las métricas guardadas en el archivo no son numéricas.")

    descripcion = datos.get("descripcion", "")
    if descripcion is None:
        descripcion = ""
    if not isinstance(descripcion, str):
        raise ModeloInvalidoError("La descripción guardada en el archivo no es válida.")

    modelo = LinearRegression()
    modelo.coef_ = np.array(coeficientes, dtype=float)
    modelo.intercept_ = float(intercepto)
    modelo.n_features_in_ = len(features)
    modelo.feature_names_in_ = np.array(features, dtype=object)

    return {
        "modelo": modelo,
        "formula": formula,
        "descripcion": descripcion,
        "features": list(features),
        "target": target,
        "metricas": dict(metricas),
        "coeficientes": [float(c) for c in coeficientes],
        "intercepto": float(intercepto),
    }


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