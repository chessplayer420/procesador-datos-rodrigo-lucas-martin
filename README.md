# procesador-datos-rodrigo-lucas-martin

## Propósito del Proyecto

Aplicación de escritorio para crear modelos de regresión lineal a partir de datasets, sin necesidad de programar. El usuario carga un archivo de datos, elige las variables de entrada y de salida, prepara los datos y (en versiones futuras) crea y evalúa el modelo.

## Estado del proyecto

Versión **0.1.0 (versión previa)**: incluye la carga y preparación de datos. **Todavía no permite crear modelos.**

### Funcionalidades incluidas
- **Importación de datos** desde archivos CSV, Excel (`.xlsx`, `.xls`) y bases de datos SQLite (`.sqlite`, `.db`).
- **Interfaz gráfica** (PySide6) que carga el dataset y lo muestra en una tabla.
- **Selección de columnas**: varias columnas de entrada y una de salida. Se valida que existan, que sean numéricas y que la salida no sea también entrada.
- **Separación en entrenamiento y test** con porcentaje de test y semilla configurables (por defecto 20 % y semilla 42) y resultado reproducible.
- **Detección de valores inexistentes** por columna, en entrenamiento y test por separado.
- **Tratamiento de valores inexistentes**: eliminar filas, rellenar con la media, con la mediana o con un valor constante. La media y la mediana se calculan solo con el conjunto de entrenamiento.

### Pendiente para próximas versiones
- Creación del modelo de regresión lineal.
- Evaluación del modelo con el conjunto de test.

## Instalación y Configuración del Entorno con `uv`

Este proyecto utiliza `uv` para la gestión rápida del entorno virtual y las dependencias de Python. Sigue estos pasos para configurarlo:

**Clonar el repositorio:**
```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DE_LA_CARPETA>
```

**Instalar uv en el equipo:**
- Windows:
```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- macOS/Linux:
```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Instalar dependencias y crear el entorno:**
```bash
uv sync
```

## Ejecución de la aplicación

Desde la raíz del repositorio:

```bash
uv run <COMANDO DE EJECUCIÓN>   # COMPLETAR con el comando real de arranque
```

### Cómo usarla
1. Pulsa **Cargar Dataset** y elige un archivo CSV, Excel o SQLite.
2. Selecciona las columnas de entrada (selección múltiple) y la columna de salida, y pulsa **Confirmar selección**.
3. Indica el porcentaje de test y la semilla, y pulsa **Separar datos**.
4. Revisa los valores inexistentes detectados, elige una estrategia y pulsa **Aplicar preprocesado**.

> Los datasets no se suben al repositorio: guárdalos en la carpeta local `data/` de tu máquina.

## Pruebas

```bash
uv run pytest
```

## Contribuir

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo de ramas, el estilo de commits y el procedimiento de release.

