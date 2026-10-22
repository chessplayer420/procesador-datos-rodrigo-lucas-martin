import pandas as pd
from typing import Tuple, Dict, Any


def detectar_valores_inexistentes(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    columnas: list
) -> Dict[str, Dict[str, int]]:
    """
    Cuenta el número de valores inexistentes (NaN) en las columnas seleccionadas
    tanto para el conjunto de entrenamiento como para el de test.
    """
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
    """
    Aplica la estrategia elegida ('eliminar', 'media', 'mediana', 'constante').
    Calcula la media/mediana ÚNICAMENTE con el conjunto de entrenamiento para evitar fuga de datos.
    """
    train_clean = df_train.copy()
    test_clean = df_test.copy()
    resumen = {"filas_eliminadas_train": 0, "filas_eliminadas_test": 0}

    if estrategia == "eliminar":
        filas_train_antes = len(train_clean)
        filas_test_antes = len(test_clean)

        train_clean = train_clean.dropna(subset=columnas)
        test_clean = test_clean.dropna(subset=columnas)

        if train_clean.empty or test_clean.empty:
            raise ValueError("La eliminación de filas dejaría vacío el conjunto de entrenamiento o el de test.")

        resumen["filas_eliminadas_train"] = filas_train_antes - len(train_clean)
        resumen["filas_eliminadas_test"] = filas_test_antes - len(test_clean)

    elif estrategia == "media":
        for col in columnas:
            if not pd.api.types.is_numeric_dtype(train_clean[col]):
                raise ValueError(f"La columna '{col}' no es numérica; no se puede calcular la media.")
            # Cálculo de la media SOLO en entrenamiento
            media = train_clean[col].mean()
            train_clean[col] = train_clean[col].fillna(media)
            test_clean[col] = test_clean[col].fillna(media)

    elif estrategia == "mediana":
        for col in columnas:
            if not pd.api.types.is_numeric_dtype(train_clean[col]):
                raise ValueError(f"La columna '{col}' no es numérica; no se puede calcular la mediana.")
            # Cálculo de la mediana SOLO en entrenamiento
            mediana = train_clean[col].median()
            train_clean[col] = train_clean[col].fillna(mediana)
            test_clean[col] = test_clean[col].fillna(mediana)

    elif estrategia == "constante":
        if valor_constante is None:
            raise ValueError("Debe proporcionar un valor constante válido.")
        try:
            val_num = float(valor_constante)
        except (ValueError, TypeError):
            raise ValueError("El valor constante introducido debe ser un número válido.")

        train_clean[columnas] = train_clean[columnas].fillna(val_num)
        test_clean[columnas] = test_clean[columnas].fillna(val_num)

    else:
        raise ValueError(f"Estrategia de preprocesado no reconocida: {estrategia}")

    return train_clean, test_clean, resumen