# Documentación de Integración Continua con GitHub Actions (#64)

## 1. Cuándo se ejecuta el flujo de trabajo
El flujo de trabajo definido en `.github/workflows/ci.yml` está configurado con el evento `on: push`. Esto significa que se ejecuta automáticamente en los servidores de GitHub cada vez que cualquier miembro del equipo sube cambios (`git push`) a **cualquier rama** del repositorio (`feature/*`, `fix/*`, `docs/*`, `dev` o `main`).

## 2. Qué hace cada paso del flujo de trabajo

El trabajo (`job`) llamado `tests` se ejecuta sobre una máquina virtual limpia con Ubuntu Linux (`runs-on: ubuntu-latest`) y realiza los siguientes pasos en orden:

1. **Descargar el código del repositorio (`uses: actions/checkout@v4`):**
   Clona el código fuente de la rama a la que se acaba de hacer `push` dentro de la máquina virtual de GitHub Actions.
2. **Instalar `uv` (`uses: astral-sh/setup-uv@v5`):**
   Descarga y configura el gestor de paquetes y entornos `uv` en el servidor de integración continua.
3. **Instalar las dependencias del proyecto (`run: uv sync --locked`):**
   Crea el entorno virtual e instala todas las librerías del proyecto usando exactamente las versiones fijadas en `uv.lock`. La opción `--locked` garantiza que la ejecución falle si alguien modificó `pyproject.toml` sin actualizar `uv.lock`.
4. **Ejecutar todas las pruebas (`run: uv run pytest`):**
   Lanza la suite completa de pruebas unitarias dentro del entorno virtual recién instalado. Si alguna prueba falla, el paso devuelve un código de error y marca la ejecución como fallida.

## 3. Cómo consultar los resultados y localizar fallos en GitHub
1. Acceder al repositorio en GitHub y hacer clic en la pestaña superior **"Actions"**.
2. En el listado de ejecuciones aparecen el mensaje del commit, la **rama** en la que se ejecutó y un icono de estado:
   - Tick verde (`✔`): todas las pruebas pasaron correctamente.
   - Cruz roja (`✖`): algún paso o prueba ha fallado.
3. Para localizar qué prueba ha fallado, se hace clic sobre la ejecución fallida, se entra en el trabajo **"Ejecutar pruebas con pytest"** y se despliega el paso **"4. Ejecutar todas las pruebas automatizadas"**, donde `pytest` muestra el resumen (`short test summary info`), el archivo de test, la función exacta y la línea del `AssertionError`.

## 4. Prevención y resolución de diferencias entre entorno local y GitHub Actions
Para asegurar que las pruebas pasen tanto en las máquinas de los desarrolladores (Windows/macOS) como en el servidor Linux de GitHub Actions, se han aplicado las siguientes medidas:
- **No depender de la carpeta local `data/`:** Dado que `data/` está en `.gitignore` y no existe en GitHub, todas las pruebas unitarias crean sus propios datos sintéticos en memoria o usan el fixture `tmp_path` de `pytest`.
- **Rutas compatibles entre sistemas operativos:** Se utiliza `pathlib.Path` en lugar de concatenar barras invertidas (`\`) propias de Windows, evitando fallos en el entorno `ubuntu-latest` de GitHub.
- **Separación estricta entre lógica e interfaz gráfica:** Los módulos de lógica (`importacion.py`, `preprocesado.py`, `modelo.py`) no instancian ventanas de `PySide6` al importarse, evitando errores por falta de pantalla gráfica (*headless server*) en GitHub Actions.