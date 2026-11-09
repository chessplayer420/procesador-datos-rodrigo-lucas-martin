# Investigación sobre Persistencia de Modelos: Pickle vs JSON

## Tabla Comparativa

| Criterio | `pickle` | `json` |
| :--- | :--- | :--- |
| **Facilidad de uso** | Muy alta. Permite serializar/deserializar objetos complejos de Python (ej. `LinearRegression` de scikit-learn o diccionarios completos) en 2 líneas. | Media. Requiere extraer manualmente los coeficientes, término independiente, nombres de columnas y métricas a un diccionario para convertirlos a texto. |
| **Tamaño del archivo generado** | Pequeño. Formato binario optimizado para estructuras de Python. | Muy pequeño y comprimible. Texto plano estructurado en formato UTF-8. |
| **Legibilidad** | Nula (formato binario inaccesible a simple vista). | Alta (texto plano estructurado en pares clave-valor legible humanamente). |
| **Seguridad con origen desconocido** | **Baja / Insegura.** Cargar archivos `.pkl` arbitrarios ejecuta código Python binario (vulnerable a inyección de código mediante desmarcado). | **Muy alta.** Es una representación pura de datos; no ejecuta código durante la lectura. |
| **Compatibilidad de versiones** | Limitada. Puede fallar si cambia la versión de Python, de `scikit-learn` o de la librería del modelo entre el guardado y la lectura. | **Muy alta.** Al guardar datos numéricos puros (coeficientes, intercepción, nombres), los valores son totalmente independientes de las versiones de las librerías. |
| **Facilidad para guardar metadatos** | Alta. Permite empaquetar en una sola estructura el modelo, nombres de features/target, métricas ($R^2$, $ECM$) y descripción. | Alta. Los datos del modelo y la información requerida por la aplicación se estructuran de forma natural en el objeto JSON. |

## Conclusiones y Elección de la Librería

### Librería Seleccionada: `json` (con arquitectura basada en parámetros)

Aunque `pickle` resulta tentador por la inmediatez al serializar el objeto binario de la regresión, se ha seleccionado **`json`** (o en su defecto un esquema híbrido centrado en guardar metadatos en JSON) por los siguientes motivos principales:

1. **Compatibilidad y Portabilidad:** En proyectos de Software e Inteligencia Artificial, serializar un objeto `LinearRegression` binario genera fuertes acoplamientos con versiones específicas de bibliotecas. Guardar en `JSON` los parámetros numéricos puros (coeficientes, término independiente), nombres de columnas y métricas garantiza que el modelo pueda reconstruirse o consultarse en cualquier versión de Python o entorno sin romper la aplicación.
2. **Seguridad:** En entornos donde la aplicación permite cargar o importar modelos previamente guardados, el uso de `pickle` representa una vulnerabilidad crítica de ejecución remota de código. `JSON` elimina completamente este riesgo al ser un formato pasivo de datos.
3. **Persistencia Completa del Dominio:** `JSON` permite almacenar de manera transparente en un único archivo la fórmula interpretada, los coeficientes redondeados, las métricas de error ($R^2$ y $ECM$) y la descripción del usuario.