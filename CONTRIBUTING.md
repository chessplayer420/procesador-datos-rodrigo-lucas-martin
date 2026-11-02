# Guía de Contribución y Flujo de Trabajo

## Normas Básicas
- **Datos fuera del repositorio**: Está terminantemente **PROHIBIDO** subir archivos de datos (datasets o modelos generados). Todo archivo de datos debe guardarse localmente en la carpeta `data/` de tu máquina.
- Antes de cada commit, usa `git status` para comprobar que no estás incluyendo archivos de datos por error.

## Modelo de Ramas
- `main`: Contiene solo versiones estables. Únicamente recibe cambios desde `dev` al publicar una release. **Prohibido hacer commits directos**.
- `dev`: Rama de integración del desarrollo. Existe desde el inicio. **Prohibido hacer commits directos**.
- `feature/<numero>-<descripcion>`: Una rama por cada nueva tarea o historia de usuario. Se crean desde `dev`.
- `fix/<descripcion>`: Para corregir errores. Se crean desde `dev`.
- `docs/<descripcion>`: Para cambios en documentación. Se crean desde `dev`.

## Procedimiento para trabajar (Crear, actualizar y fusionar)
1. Sitúate en la rama de desarrollo y actualízala: `git checkout dev` y `git pull origin dev`.
2. Crea tu rama: `git checkout -b feature/XX-tu-tarea`.
3. Haz tus cambios, revisa con `git status`, y haz `git add .` seguido de `git commit`.
4. Antes de fusionar, actualiza tu rama con los últimos cambios de dev: `git merge dev` (en tu rama).
5. Fusiona tu rama en `dev` (`git checkout dev`, `git merge feature/XX-tu-tarea`) y súbelo con `git push`.
6. **NO ELIMINAR LA RAMA**: Una vez fusionada, la rama original (`feature`, `fix` o `docs`) debe conservarse en el repositorio para mantener el historial.

## Estilo de mensajes de Commit
Usa mensajes descriptivos y un prefijo que indique el tipo de trabajo: `feat:`, `fix:`, `docs:`.

## Gestión de Dependencias con uv
- Para añadir un paquete nuevo al proyecto usar **siempre**: `uv add <paquete>`.
- El archivo `uv.lock` debe incluirse siempre en los commits.

## Reglas de fusión (Merge)
Antes de fusionar cualquier rama a dev (o dev a main), el desarrollador debe actualizar su rama con los últimos cambios de dev y ejecutar todas las pruebas en su máquina local. Solo se fusiona si todas las pruebas pasan.

## Instrucciones de ejecución de pruebas
Las pruebas automatizadas del proyecto se ejecutan desde la raíz del repositorio con el comando:

```bash
uv run pytest
```

## Procedimiento de release

Pasos para publicar una versión en GitHub. Se redactó con la release `v0.1.0`; para las siguientes, sustituye `0.1.0` por el número de versión nuevo.

**Versionado (SemVer `MAJOR.MINOR.PATCH`):** mientras la aplicación esté en desarrollo usamos `0.x`. Sube `MINOR` (0.1.0 → 0.2.0) al añadir funcionalidades y `PATCH` (0.1.0 → 0.1.1) al corregir errores. `1.0.0` significa que la aplicación está completa.

### 1. Requisitos previos
- Las historias de la release están terminadas y fusionadas en `dev`.
- Todas las pruebas pasan sobre `dev` (regla de fusión de esta guía):
```bash
  git checkout dev
  git pull origin dev
  uv sync
  uv run pytest
```
- Un miembro del equipo que **no haya implementado** esas historias prueba a mano la aplicación en `dev` con uno de los datasets de ejemplo guardado en `data/`: cargar el archivo, seleccionar columnas, separar los datos y preprocesarlos. Cada error encontrado se corrige en una rama `fix/<descripcion>` creada desde `dev` y fusionada en `dev` antes de continuar.
- El `README.md` describe las funcionalidades de la versión y cómo ejecutarla.

### 2. Actualizar la versión
1. Comprueba la versión actual:
```bash
   grep -n '^version' pyproject.toml
```
2. Si no es la que vas a publicar, corrígela en una rama creada desde `dev`:
```bash
   git checkout dev
   git pull origin dev
   git checkout -b fix/version-0.1.0
   # edita pyproject.toml: version = "0.1.0"
   uv lock
   git add pyproject.toml uv.lock
   git commit -m "fix: actualiza la versión a 0.1.0"
```
3. Fusiónala en `dev` siguiendo el procedimiento de trabajo de esta guía (`git checkout dev`, `git merge fix/version-0.1.0`, `git push origin dev`). **No elimines la rama `fix`.**

### 3. Fusionar `dev` en `main`
```bash
git checkout main
git pull origin main
git merge --no-ff dev -m "Merge dev into main: release 0.1.0"
git push origin main
```
Si prefieres una pull request de `dev` a `main`, fusiónala con **"Create a merge commit"** (no con *squash*, que haría divergir los historiales de `main` y `dev`).

Después, sincroniza `dev` con `main` y comprueba que contienen el mismo código:
```bash
git checkout dev
git merge main
git push origin dev
git diff main dev        # debe salir vacío
```

### 4. Crear y subir la etiqueta
La etiqueta se crea sobre el último commit de `main`:
```bash
git checkout main
git pull origin main
git tag -a v0.1.0 -m "Versión 0.1.0"
git push origin v0.1.0
git show v0.1.0 --stat   # comprueba que apunta al commit correcto
```
Si te equivocas antes de publicar la release, bórrala y vuelve a crearla:
```bash
git tag -d v0.1.0
git push origin --delete v0.1.0
```

### 5. Crear la release en GitHub
1. En el repositorio, entra en **Releases → Draft a new release**.
2. En **Choose a tag**, elige `v0.1.0`.
3. Título: `v0.1.0`.
4. Marca **Set as a pre-release** mientras la aplicación no esté completa (versiones `0.x`).
5. Pega las notas (apartado siguiente) y pulsa **Publish release**.

Alternativa con la CLI de GitHub: `gh release create v0.1.0 --title "v0.1.0" --prerelease --notes-file NOTAS_RELEASE.md`

### 6. Redactar las notas de la release
Incluyen, en este orden:
1. Descripción breve de la aplicación y del estado del proyecto.
2. **Funcionalidades** (commits `feat`) y **correcciones** (commits `fix`).
3. **Pendiente** para próximas versiones.
4. Enlace al `README.md` de esa versión: `https://github.com/chessplayer420/procesador-datos-rodrigo-lucas-martin/blob/v0.1.0/README.md`

Para obtener los commits (reescríbelos como frases legibles, no pegues el log tal cual):
```bash
# Primera release
git log --no-merges --pretty=format:"%s" | grep -E "^(feat|fix)"
# Siguientes releases (desde la etiqueta anterior)
git log v0.1.0..main --no-merges --pretty=format:"%s" | grep -E "^(feat|fix)"
```

### 7. Comprobar la descarga
Lo hace un miembro del equipo, siguiendo **solo** el `README.md`:
```bash
curl -L -o v0.1.0.zip https://github.com/chessplayer420/procesador-datos-rodrigo-lucas-martin/archive/refs/tags/v0.1.0.zip
unzip v0.1.0.zip -d comprobacion-v0.1.0
cd comprobacion-v0.1.0/procesador-datos-rodrigo-lucas-martin-0.1.0
uv sync
uv run <comando de ejecución del README>
```
Comprueba que la ventana abre y que, con un dataset de `data/`, funciona el flujo disponible (cargar, seleccionar columnas, separar y preprocesar). Si algo falla, corrígelo en una rama `fix/<descripcion>` desde `dev` y repite el procedimiento con una versión `PATCH` (por ejemplo, `0.1.1`).

### Lista de comprobación rápida
- [ ] Historias fusionadas en `dev` y `uv run pytest` en verde
- [ ] Prueba manual hecha por alguien que no implementó las historias
- [ ] `README.md` actualizado
- [ ] `version` correcta en `pyproject.toml` y `uv.lock` actualizado
- [ ] `dev` fusionado en `main` con merge commit; `git diff main dev` vacío
- [ ] Etiqueta anotada `vX.Y.Z` subida
- [ ] Release creada (pre-release si procede) con las notas completas
- [ ] Descarga del zip comprobada con `uv sync` y `uv run`

