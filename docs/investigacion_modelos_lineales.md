# Investigación de Librerías para Modelos de Regresión Lineal (#13)

## 1. Comparativa de librerías investigadas

Se han evaluado tres alternativas en Python para construir modelos de regresión lineal: `scikit-learn`, `statsmodels` y `numpy`.

| Aspecto evaluado | scikit-learn | statsmodels | numpy |
| :--- | :--- | :--- | :--- |
| Facilidad y documentación | Muy alta, API intuitiva y ejemplos claros | Media, enfocada a inferencia estadística | Baja para flujo de Machine Learning |
| Regresión simple y múltiple | Idéntica mediante `LinearRegression()` | Requiere añadir columna constante (`add_constant`) | `polyfit` (simple) o `linalg.lstsq` (múltiple) |
| Coeficientes e intercepto | Directo con `.coef_` e `.intercept_` | Mediante serie `.params` | Devuelve vectores numéricos sin nombres |
| Métricas de error | Incluye `mean_squared_error` y `r2_score` | Incluidas en el objeto resultado de `OLS` | No incluidas, cálculo manual con arrays |
| Instalación con `uv` | `uv add scikit-learn` | `uv add statsmodels` | `uv add numpy` |

## 2. Librería seleccionada y justificación

Se selecciona **`scikit-learn`** como librería oficial del proyecto por los siguientes motivos:

1. **Soporte unificado para regresión simple y múltiple:** La clase `LinearRegression` recibe directamente un `DataFrame` de `pandas` con una o varias columnas de entrada (`X`) sin cambiar el código.
2. **Extracción directa de la fórmula:** Una vez entrenado el modelo con `.fit(X_train, y_train)`, los pesos se obtienen en `modelo.coef_` y el término independiente en `modelo.intercept_`, permitiendo construir de forma inmediata la ecuación con los nombres de las columnas.
3. **Cálculo integrado de métricas:** El módulo `sklearn.metrics` ofrece funciones estandarizadas para obtener el Error Cuadrático Medio (`mean_squared_error`) y el coeficiente de determinación (`r2_score`).
4. **Integración limpia con `uv`:** Se instala y bloquea en `pyproject.toml` y `uv.lock` mediante `uv add scikit-learn`.

## 3. Prueba de concepto y resultados obtenidos

La prueba de concepto se ha implementado en el script independiente `prototipos/poc_modelo_lineal.py` utilizando el dataset `data/housing.csv`.

### Pasos ejecutados y resultados:
1. **Carga y selección de variables:** Se importó `housing.csv` mediante `importar_datos` (Historia #10). Tras filtrar las columnas numéricas y descartar filas con valores nulos, se seleccionaron como entradas (`X`): `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households` y `median_income`, y como salida (`y`): `median_house_value`.
2. **Separación en entrenamiento y test:** Usando `separar_train_test` (80 % entrenamiento, 20 % test, semilla 42), se obtuvieron **16346 filas de entrenamiento** y **4087 filas de test**.
3. **Entrenamiento y fórmula obtenida:**
   - Coeficiente de `longitude`: `-42408.4312`
   - Coeficiente de `latitude`: `-42345.9709`
   - Coeficiente de `housing_median_age`: `1126.7479`
   - Coeficiente de `total_rooms`: `-8.3001`
   - Coeficiente de `total_bedrooms`: `108.2856`
   - Coeficiente de `population`: `-38.7128`
   - Coeficiente de `households`: `53.2818`
   - Coeficiente de `median_income`: `40273.4003`
   - Término independiente: `-3551005.4055`

   **Fórmula del modelo:**
   `median_house_value = -42408.4312 * longitude + -42345.9709 * latitude + 1126.7479 * housing_median_age + -8.3001 * total_rooms + 108.2856 * total_bedrooms + -38.7128 * population + 53.2818 * households + 40273.4003 * median_income + (-3551005.4055)`

4. **Métricas de evaluación:**
   - **Entrenamiento:** ECM = `4828219699.5146` | $R^2$ = `0.6372`
   - **Test:** ECM = `4880626591.6873` | $R^2$ = `0.6356`

5. **Predicción para un valor nuevo:**
   Para un registro con los valores medios del conjunto de entrenamiento (`longitude = -119.559055`, `latitude = 35.619817`, `housing_median_age = 28.564419`, `total_rooms = 2643.083568`, `total_bedrooms = 539.067111`, `population = 1426.252661`, `households = 500.273094`, `median_income = 3.878653`), el modelo predijo un `median_house_value` de **`207218.8335`**.

### Interpretación de los resultados:
- **Significado de los coeficientes:** Indican el cambio estimado en `median_house_value` por cada unidad que aumenta la variable de entrada, manteniendo el resto constantes. Por ejemplo, un incremento de 1 unidad en los ingresos medios (`median_income`) eleva el valor estimado de la vivienda en `40273.4003` dólares, mientras que una mayor antigüedad (`housing_median_age`) añade `1126.7479` dólares por año.
- **Significado del ECM:** El Error Cuadrático Medio está expresado en dólares al cuadrado (`~4.88e9` en test). Su raíz cuadrada (RMSE) equivale a unos `69861` dólares, lo que representa el margen de error promedio en las predicciones del precio de las viviendas.
- **Significado del $R^2$:** El valor de `0.6356` en test indica que el modelo lineal explica el **63.56 %** de la variabilidad del precio de las viviendas a partir de las 8 variables de entrada.
- **Comparación entre entrenamiento y test:** Las métricas de entrenamiento ($R^2 = 0.6372$) y de test ($R^2 = 0.6356$) son prácticamente idénticas. Esto demuestra que el modelo **no sufre sobreajuste (*overfitting*)** y generaliza con la misma precisión ante datos nuevos no vistos durante el entrenamiento.

## 4. Dificultades encontradas y cómo se resolvieron

1. **Columnas no numéricas y valores inexistentes (`NaN`) en `housing.csv`:** El dataset incluye la columna de texto `ocean_proximity` y valores nulos en `total_bedrooms`, lo que provoca que `LinearRegression.fit()` lance un `ValueError`. Se resolvió filtrando las columnas numéricas (`select_dtypes(include=["number"])`) y eliminando filas incompletas (`dropna()`) antes de separar y entrenar.
2. **Advertencia de nombres de columnas en la predicción:** Al predecir sobre un vector numérico simple, `scikit-learn` emite un `UserWarning` indicando que faltan los nombres de las características (*feature names*). Se solucionó pasando los nuevos valores dentro de un `pd.DataFrame` con `columns=columnas_entrada`.