# prototipos/poc_modelo_lineal.py

import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Asegurar que Python encuentre los módulos dentro de src/
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_PAQUETE = RAIZ_PROYECTO / "src" / "procesador_datos_rodrigo_lucas_martin"
RUTA_SRC = RAIZ_PROYECTO / "src"
for ruta in (RUTA_PAQUETE, RUTA_SRC):
    if ruta.exists() and str(ruta) not in sys.path:
        sys.path.insert(0, str(ruta))

try:
    from procesador_datos_rodrigo_lucas_martin.importacion import importar_datos
    from procesador_datos_rodrigo_lucas_martin.preprocesado import (
        separar_train_test,
        PORCENTAJE_TEST_POR_DEFECTO,
        SEMILLA_POR_DEFECTO,
    )
except ImportError:
    from importacion import importar_datos
    from preprocesado import (
        separar_train_test,
        PORCENTAJE_TEST_POR_DEFECTO,
        SEMILLA_POR_DEFECTO,
    )


def obtener_dataset_ejemplo() -> Path:
    """Busca automáticamente el primer dataset compatible dentro de la carpeta local data/."""
    carpeta_data = RAIZ_PROYECTO / "data"
    extensiones = (".csv", ".xlsx", ".xls", ".sqlite", ".db")

    if carpeta_data.exists():
        for archivo in sorted(carpeta_data.iterdir()):
            if archivo.suffix.lower() in extensiones:
                return archivo

    raise FileNotFoundError(
        "No se encontró ningún archivo compatible en la carpeta 'data/'. "
        "Coloca un dataset de ejemplo en 'data/' antes de ejecutar el script."
    )


def ejecutar_prueba_de_concepto():
    # 1. Cargar los datos mediante el módulo de importación (Historia #10)
    ruta_dataset = obtener_dataset_ejemplo()
    print(f"=== 1. Cargando dataset: {ruta_dataset.name} ===")
    df = importar_datos(str(ruta_dataset))

    # Filtrar columnas numéricas y descartar filas con valores inexistentes para entrenar
    df_numerico = df.select_dtypes(include=["number"]).dropna()
    if df_numerico.shape[1] < 2:
        raise ValueError("El dataset necesita al menos 2 columnas numéricas (entradas y salida).")

    columnas_entrada = [str(c) for c in df_numerico.columns[:-1]]
    columna_salida = str(df_numerico.columns[-1])
    df_numerico.columns = [str(c) for c in df_numerico.columns]

    print(f"Columnas de entrada (X): {columnas_entrada}")
    print(f"Columna de salida (y): {columna_salida}\n")

    # 2. Separar los datos en entrenamiento y test usando separar_train_test de preprocesado.py
    df_train, df_test = separar_train_test(
        df_numerico,
        PORCENTAJE_TEST_POR_DEFECTO,
        SEMILLA_POR_DEFECTO
    )
    print("=== 2. Separación de datos ===")
    print(f"Filas de entrenamiento: {len(df_train)} | Filas de test: {len(df_test)}\n")

    X_train = df_train[columnas_entrada]
    y_train = df_train[columna_salida]
    X_test = df_test[columnas_entrada]
    y_test = df_test[columna_salida]

    # 3. Entrenar el modelo con el conjunto de entrenamiento
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 4. Obtener coeficientes, término independiente y escribir la fórmula del modelo
    coeficientes = modelo.coef_
    termino_independiente = modelo.intercept_

    terminos = " + ".join(
        f"{coef:.4f} * {col}" for coef, col in zip(coeficientes, columnas_entrada)
    )
    formula = f"{columna_salida} = {terminos} + ({termino_independiente:.4f})"

    print("=== 3 y 4. Parámetros y Fórmula del Modelo ===")
    for col, coef in zip(columnas_entrada, coeficientes):
        print(f"  Coeficiente de '{col}': {coef:.4f}")
    print(f"  Término independiente: {termino_independiente:.4f}")
    print(f"\n  Fórmula:\n  {formula}\n")

    # 5. Calcular el error cuadrático medio (ECM) y R² en entrenamiento y test
    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    ecm_train = mean_squared_error(y_train, pred_train)
    r2_train = r2_score(y_train, pred_train)

    ecm_test = mean_squared_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)

    print("=== 5. Métricas de Evaluación ===")
    print(f"  [Entrenamiento] ECM: {ecm_train:.4f} | R²: {r2_train:.4f}")
    print(f"  [Test]          ECM: {ecm_test:.4f} | R²: {r2_test:.4f}\n")

    # 6. Realizar una predicción para un valor de entrada nuevo
    nuevo_dato = pd.DataFrame([X_train.mean().values], columns=columnas_entrada)
    prediccion_nueva = modelo.predict(nuevo_dato)[0]

    print("=== 6. Predicción para un valor de entrada nuevo ===")
    print(f"  Valores de entrada (media):\n{nuevo_dato.to_string(index=False)}")
    print(f"  Predicción para '{columna_salida}': {prediccion_nueva:.4f}")


if __name__ == "__main__":
    ejecutar_prueba_de_concepto()