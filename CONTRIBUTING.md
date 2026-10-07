# Guía de Contribución y Flujo de Trabajo

## Normas Básicas
- **Datos fuera del repositorio**: Está terminantemente **PROHIBIDO** subir archivos de datos (datasets o modelos generados). Todo archivo de datos debe guardarse localmente en la carpeta `data/` de tu máquina (la cual está ignorada por git).
- Antes de cada commit, usa `git status` para comprobar que no estás incluyendo archivos de datos por error.

## Modelo de Ramas
- `main`: Contiene solo versiones estables. Únicamente recibe cambios desde `dev` al publicar una release. **Prohibido hacer commits directos**.
- `dev`: Rama de integración del desarrollo. Existe desde el inicio. **Prohibido hacer commits directos**.
- `feature/<numero>-<descripcion>`: Una rama por cada nueva tarea o historia de usuario (ej. `feature/10-importacion-datos`). Se crean desde `dev`.
- `fix/<descripcion>`: Para corregir errores. Se crean desde `dev`.
- `docs/<descripcion>`: Para cambios en documentación. Se crean desde `dev`.

## Procedimiento para trabajar (Crear, actualizar y fusionar)
1. Sitúate en la rama de desarrollo y actualízala: `git checkout dev` y `git pull origin dev`.
2. Crea tu rama: `git checkout -b feature/XX-tu-tarea`.
3. Haz tus cambios, revisa con `git status`, y haz `git add .` seguido de `git commit`.
4. Antes de fusionar, actualiza tu rama con los últimos cambios de dev: `git merge dev` (en tu rama).
5. Fusiona tu rama en `dev` (`git checkout dev`, `git merge feature/XX-tu-tarea`) y súbelo con `git push`. Opcionalmente, usa Pull Requests en GitHub.
6. Elimina la rama tras la fusión.

## Estilo de mensajes de Commit
Usa mensajes descriptivos y un prefijo que indique el tipo de trabajo:
- `feat: añade el modelo de predicción`
- `fix: corrige error de lectura de CSV`
- `docs: actualiza instrucciones en el README`

## Gestión de Dependencias con uv
- Para añadir un paquete nuevo al proyecto usar **siempre**: `uv add <paquete>`.
- **NUNCA** editar el entorno a mano ni usar `pip install`.
- El archivo `uv.lock` debe incluirse siempre en los commits para que todo el equipo use exactamente las mismas versiones.

## Modificar el README o la Documentación
Cualquier cambio en la documentación debe hacerse creando una rama del tipo `docs/<descripcion>` desde `dev`, siguiendo el mismo flujo de trabajo mencionado arriba.
