# Mejora de Visualización de Tablas Grandes 

## 1. Técnica elegida
Se ha implementado **scroll virtual** mediante la arquitectura Modelo/Vista nativa de PySide6: **`QTableView`** conectado a una subclase personalizada de **`QAbstractTableModel`** (`ModeloTablaDataFrame`).

## 2. Motivo de la elección
Frente a la paginación (que divide el dataset en páginas y obliga a pulsar botones para recorrerlo) o la carga progresiva (que acumula widgets en memoria a medida que el usuario baja), el **scroll virtual** con `QTableView` ofrece:
- **Renderizado bajo demanda:** Qt solicita a `data(index, role)` únicamente las filas y columnas visibles en el área de la tabla en cada instante.
- **Acceso directo e inmediato a todas las filas:** El usuario puede arrastrar la barra de desplazamiento hasta cualquier punto (por ejemplo, la fila 20.000) al instante sin perder contexto ni cambiar de página.
- **Uso constante de memoria:** No se instancian objetos `QTableWidgetItem` por celda, leyendo directamente del `DataFrame` de `pandas` mediante `.iat[fila, col]`.

## 3. Comportamiento con el dataset de ejemplo (`housing.csv`, 20.640 filas)

| Aspecto medido | Antes (`QTableWidget` con bucle) | Después (`QTableView` + `QAbstractTableModel` + `QThread`) |
| :--- | :--- | :--- |
| **Objetos creados al cargar** | > 206.400 `QTableWidgetItem` (20.640 × 10) | 0 objetos por celda (solo las ~15 filas visibles) |
| **Respuesta de la ventana al cargar** | La ventana se congelaba varios segundos y el sistema operativo la marcaba como *"No responde"* | La lectura ocurre en un `QThread` con barra de progreso y la ventana se puede mover con fluidez en todo momento |
| **Desplazamiento vertical** | Lento y con tirones al arrastrar la barra | Fluido e instantáneo desde la fila 1 hasta la fila 20.640 |
| **Confirmación de columnas seleccionadas** | Volvía a crear decenas de miles de celdas para `tabla_seleccion` | Actualización instantánea resaltando la columna de salida mediante `Qt.BackgroundRole` |