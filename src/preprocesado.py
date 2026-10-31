import math
import numbers

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

# Parámetros por defecto de la separación Train/Test
PORCENTAJE_TEST_POR_DEFECTO = 20
SEMILLA_POR_DEFECTO = 42
MIN_FILAS_SEPARACION = 5


class ConjuntoVacioError(ValueError):
    """La eliminación de filas dejaría vacío el conjunto de entrenamiento o el de test."""


def parsear_valor_constante(valor: Any) -> float:
    """Convierte el valor constante (número o texto, admite coma decimal) a float.

    Lanza ValueError si no es un número finito válido.
    """
    if valor is None:
        raise ValueError("Debe proporcionar un valor constante válido.")
    msg = "El valor introducido debe ser un número válido."
    if isinstance(valor, bool):
        raise ValueError(msg)
    try:
        numero = float(valor.strip().replace(",", ".")) if isinstance(valor, str) else float(valor)
    except (ValueError, TypeError):
        raise ValueError(msg) from None
    if not math.isfinite(numero):
        raise ValueError(msg)
    return numero


def hay_valores_inexistentes(df_train: pd.DataFrame, df_test: pd.DataFrame, columnas: list) -> bool:
    """True si queda algún valor inexistente en las columnas indicadas de train o de test."""
    return bool(df_train[columnas].isna().any().any() or df_test[columnas].isna().any().any())


def detectar_valores_inexistentes(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    columnas: list
) -> Dict[str, Dict[str, int]]:
    """Cuenta cuántos valores nulos hay por columna en train y test."""
    nulos_train = df_train[columnas].isna().sum().to_dict()
    nulos_test = df_test[columnas].isna().sum().to_dict()
    
    return {
        "entrenamiento": nulos_train,
        "test": nulos_test
    }


def tratar_valores_inexistentes(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    columnas: list,
    estrategia: str,
    valor_constante: Any = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Aplica la estrategia seleccionada ('eliminar', 'media', 'mediana', 'constante').

    No modifica los DataFrames recibidos: trabaja sobre copias. La media y la mediana se
    calculan únicamente con el conjunto de entrenamiento y se usan para rellenar ambos.

    Devuelve (train_limpio, test_limpio, resumen). El resumen contiene:
    filas_eliminadas_train/test y valores_rellenados_train/test.
    Lanza ConjuntoVacioError (subclase de ValueError) si eliminar filas dejaría vacío
    algún conjunto, y ValueError para cualquier otro parámetro no válido.
    """
    train_clean = df_train.copy()
    test_clean = df_test.copy()
    resumen = {
        "filas_eliminadas_train": 0,
        "filas_eliminadas_test": 0,
        "valores_rellenados_train": 0,
        "valores_rellenados_test": 0,
    }

    if estrategia == "eliminar":
        f_train_antes = len(train_clean)
        f_test_antes = len(test_clean)

        train_clean = train_clean.dropna(subset=columnas)
        test_clean = test_clean.dropna(subset=columnas)

        if train_clean.empty or test_clean.empty:
            raise ConjuntoVacioError("La eliminación de filas dejaría vacío el conjunto de entrenamiento o el de test.")

        resumen["filas_eliminadas_train"] = f_train_antes - len(train_clean)
        resumen["filas_eliminadas_test"] = f_test_antes - len(test_clean)

    elif estrategia in ("media", "mediana"):
        # Primero se calculan todos los valores de relleno (solo con TRAIN, sin data leakage)
        # y se valida todo; así un error no deja el preprocesado a medias.
        valores_relleno = {}
        for col in columnas:
            if not pd.api.types.is_numeric_dtype(train_clean[col]):
                raise ValueError(f"La columna '{col}' no es numérica; no se puede calcular la {estrategia}.")
            valor = train_clean[col].mean() if estrategia == "media" else train_clean[col].median()
            if pd.isna(valor):
                raise ValueError(
                    f"La columna '{col}' no tiene valores en el conjunto de entrenamiento; "
                    f"no se puede calcular la {estrategia}."
                )
            valores_relleno[col] = valor

        resumen["valores_rellenados_train"] = int(train_clean[columnas].isna().sum().sum())
        resumen["valores_rellenados_test"] = int(test_clean[columnas].isna().sum().sum())
        for col, valor in valores_relleno.items():
            train_clean[col] = train_clean[col].fillna(valor)
            test_clean[col] = test_clean[col].fillna(valor)

    elif estrategia == "constante":
        # Si no es un número válido, lanza ValueError con el texto esperado por las pruebas
        val_num = parsear_valor_constante(valor_constante)

        resumen["valores_rellenados_train"] = int(train_clean[columnas].isna().sum().sum())
        resumen["valores_rellenados_test"] = int(test_clean[columnas].isna().sum().sum())
        train_clean[columnas] = train_clean[columnas].fillna(val_num)
        test_clean[columnas] = test_clean[columnas].fillna(val_num)

    else:
        raise ValueError(f"Estrategia de preprocesado no reconocida: {estrategia}")

    # Devuelve los 3 elementos necesarios para que los tests los desempaqueten
    return train_clean, test_clean, resumen


def validar_seleccion_columnas(
    df: pd.DataFrame, 
    features: List[str], 
    target: str) -> Tuple[bool, str]:
    """
    Valida la selección de columnas de entrada (features) y salida (target).
    
    Retorna:
        (True, mensaje_éxito) si la selección cumple todos los requisitos.
        (False, mensaje_error) si falla alguna validación.
    """
    features = list(features or [])
    target = target or ""

    # 1. Comprobar selecciones vacías (si faltan ambas, se indican las dos)
    if not features and not target:
        return False, ("Debes seleccionar al menos una columna de entrada (feature) "
                       "y una columna de salida (target).")
    if not features:
        return False, "Debes seleccionar al menos una columna de entrada (feature)."
    if not target:
        return False, "Debes seleccionar una columna de salida (target)."

    # 2. Entrada y salida no pueden coincidir
    if target in features:
        return False, f"La columna objetivo '{target}' no puede ser seleccionada también como columna de entrada."

    # 3. Comprobar existencia de columnas en el DataFrame
    #    (la GUI maneja los nombres como texto, aunque el DataFrame tenga columnas numéricas)
    columnas = {str(c): c for c in df.columns}
    columnas_faltantes = [col for col in features + [target] if col not in columnas]
    if columnas_faltantes:
        return False, f"Las siguientes columnas no existen en el dataset: {', '.join(columnas_faltantes)}"

    # 4. Comprobar tipos numéricos (los booleanos no son válidos para regresión)
    columnas_no_numericas = [
        col for col in features + [target]
        if not pd.api.types.is_numeric_dtype(df[columnas[col]])
        or pd.api.types.is_bool_dtype(df[columnas[col]])
    ]

    if columnas_no_numericas:
        cols_str = ", ".join(columnas_no_numericas)
        return False, f"Las siguientes columnas no son numéricas y no se pueden usar en el modelo de regresión: {cols_str}"

    msg = (
        f"Selección confirmada correctamente.\n"
        f"Columnas de entrada: {', '.join(features)}\n"
        f"Columna de salida: {target}"
    )
    return True, msg


def parsear_parametros_separacion(texto_porcentaje: str, texto_semilla: str) -> Tuple[float, int]:
    """Convierte los textos de la interfaz en (porcentaje_test, semilla).

    Lanza ValueError con un mensaje explicativo si el porcentaje no es un número
    entre 1 y 99, o si la semilla no es un número entero.
    """
    msg_pct = "El porcentaje de test debe ser un número entre 1 y 99."
    try:
        porcentaje = float(str(texto_porcentaje).strip().replace(",", "."))
    except ValueError:
        raise ValueError(msg_pct) from None
    if not math.isfinite(porcentaje) or not (1 <= porcentaje <= 99):
        raise ValueError(msg_pct)

    try:
        semilla = int(str(texto_semilla).strip())
    except ValueError:
        raise ValueError("La semilla debe ser un número entero.") from None
    if semilla < 0:
        raise ValueError("La semilla debe ser un número entero no negativo.")

    return porcentaje, semilla


def separar_train_test(
    df: pd.DataFrame,
    porcentaje_test: float = PORCENTAJE_TEST_POR_DEFECTO,
    semilla: int = SEMILLA_POR_DEFECTO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separa df en (entrenamiento, test) de forma aleatoria y reproducible.

    - porcentaje_test: entre 1 y 99; el de entrenamiento es el complementario.
    - semilla: entero >= 0. Mismo dataset + mismo porcentaje + misma semilla
      producen siempre los mismos conjuntos.
    Los conjuntos conservan el índice original (y el orden original de las filas),
    no comparten ninguna fila y juntos contienen todo el dataset.
    Lanza ValueError si los parámetros no son válidos, si hay menos de
    MIN_FILAS_SEPARACION filas o si algún conjunto quedaría vacío.
    """
    n = 0 if df is None else len(df)
    if n < MIN_FILAS_SEPARACION:
        raise ValueError(
            f"El dataset debe tener al menos {MIN_FILAS_SEPARACION} filas para poder "
            f"separarlo en entrenamiento y test (tiene {n})."
        )

    if (isinstance(porcentaje_test, bool) or not isinstance(porcentaje_test, numbers.Real)
            or not math.isfinite(porcentaje_test) or not (1 <= porcentaje_test <= 99)):
        raise ValueError("El porcentaje de test debe ser un número entre 1 y 99.")
    if isinstance(semilla, bool) or not isinstance(semilla, numbers.Integral):
        raise ValueError("La semilla debe ser un número entero.")
    if semilla < 0:
        raise ValueError("La semilla debe ser un número entero no negativo.")

    n_test = int(math.floor(n * porcentaje_test / 100 + 0.5))  # redondeo clásico (.5 hacia arriba)
    if n_test < 1 or n_test >= n:
        vacio = "test" if n_test < 1 else "entrenamiento"
        raise ValueError(
            f"Con {n} filas y un {porcentaje_test:g} % de test, el conjunto de {vacio} "
            f"quedaría vacío. Cambia el porcentaje."
        )

    orden = np.random.default_rng(int(semilla)).permutation(n)
    pos_test = np.sort(orden[:n_test])
    pos_train = np.sort(orden[n_test:])
    return df.iloc[pos_train].copy(), df.iloc[pos_test].copy()