# src/procesador_datos_rodrigo_lucas_martin/separacion.py

import pandas as pd


def separar_datos(
    df: pd.DataFrame,
    porcentaje_test: float = 20.0,
    semilla: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame aleatoriamente en conjuntos de entrenamiento y test.

    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset a dividir.
    porcentaje_test : float, opcional
        Porcentaje de datos destinado al conjunto de test (entre 1 y 99). Por defecto 20.0.
    semilla : int, opcional
        Semilla entera para garantizar la reproducibilidad. Por defecto 42.

    Devuelve:
    ---------
    tuple[pd.DataFrame, pd.DataFrame]
        Tupla con (df_entrenamiento, df_test).

    Excepciones:
    ------------
    ValueError:
        - Si el porcentaje no es un número entre 1 y 99.
        - Si la semilla no es un número entero.
        - Si el dataset tiene menos de 5 filas o algún subconjunto queda vacío.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("No hay un dataset válido cargado para separar.")

    # 1. Validar que la semilla sea un número entero (excluyendo booleanos)
    if isinstance(semilla, bool) or not isinstance(semilla, int):
        raise ValueError("La semilla introducida debe ser un número entero.")

    # 2. Validar que el porcentaje sea un número entre 1 y 99
    if isinstance(porcentaje_test, bool) or not isinstance(porcentaje_test, (int, float)):
        raise ValueError("El porcentaje de test debe ser un número entre 1 y 99.")
    if porcentaje_test < 1 or porcentaje_test > 99:
        raise ValueError("El porcentaje de test debe ser un número entre 1 y 99.")

    # 3. Validar tamaño mínimo del dataset (al menos 5 filas)
    total_filas = len(df)
    if total_filas < 5:
        raise ValueError(
            f"El dataset tiene menos de 5 filas ({total_filas}). "
            "Se necesitan al menos 5 filas para realizar la separación."
        )

    # 4. Calcular tamaños de cada subconjunto y comprobar que ninguno quede vacío
    n_test = round(total_filas * (porcentaje_test / 100.0))
    n_entrenamiento = total_filas - n_test

    if n_test == 0 or n_entrenamiento == 0:
        raise ValueError(
            f"Con el porcentaje elegido ({porcentaje_test}%), uno de los conjuntos "
            f"quedaría vacío (Entrenamiento: {n_entrenamiento} filas, Test: {n_test} filas)."
        )

    # 5. Barajar de forma reproducible usando la semilla (ajustada a 32 bits para pandas)
    semilla_normalizada = semilla % (2**32)
    df_barajado = df.sample(frac=1.0, random_state=semilla_normalizada)

    df_test = df_barajado.iloc[:n_test].copy()
    df_entrenamiento = df_barajado.iloc[n_test:].copy()

    return df_entrenamiento, df_test