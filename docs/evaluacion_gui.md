# Evaluación de Librerías de Interfaz Gráfica para Python

## Tabla Comparativa

| Criterio | Tkinter | PySide6 (Qt for Python) | wxPython |
| :--- | :--- | :--- | :--- |
| **Facilidad de uso y curva de aprendizaje** | Muy alta. Incluida en la librería estándar, sintaxis sencilla. | Media-Alta. Módulos estructurados orientados a objetos y herramientas como Qt Designer. | Media. Estructura tradicional orientada a eventos similar a C++. |
| **Compatibilidad multiplataforma** | Windows, macOS, Linux (aspecto nativo básico). | Windows, macOS, Linux (aspecto moderno y nativo en cada SO). | Windows, macOS, Linux (usa widgets nativos del SO). |
| **Licencia y condiciones de uso** | Python Software Foundation License (Permisiva/Comercial). | LGPLv3 (Permite uso comercial sin publicar código propietario si se usa como librería enlazada). | LGPL con excepción (Permisiva para uso comercial). |
| **Documentación y comunidad** | Extensa y abundante por antigüedad, aunque con tutoriales desactualizados. | Excelente documentación oficial de Qt/Qt Company y amplia comunidad activa. | Documentación moderada, comunidad menor en comparación con Qt. |
| **Componentes requeridos** | Diálogo de archivos (`filedialog`), tablas (`ttk.Treeview`), desplegables (`ttk.Combobox`), texto (`Text`), Matplotlib (`FigureCanvasTkAgg`). | Diálogo (`QFileDialog`), tablas (`QTableWidget`), desplegables (`QComboBox`), texto (`QLineEdit`/`QTextEdit`), Matplotlib (`FigureCanvasQTAgg`). | Diálogo (`FileDialog`), tablas (`Grid`), desplegables (`ComboBox`), texto (`TextCtrl`), Matplotlib (`FigureCanvasWxAgg`). |
| **Instalación con `uv`** | Incluida por defecto en la instalación de Python. | `uv add PySide6` | `uv add wxPython` |

## Selección y Justificación

Se ha seleccionado **PySide6** como la librería para el proyecto por las siguientes razones:
1. **Componentes integrados:** Dispone de un `QTableWidget` muy potente y configurable con scroll automático, soportando la integración fluida de gráficos de Matplotlib mediante `FigureCanvasQTAgg`.
2. **Estabilidad y soporte corporativo:** Es la librería oficial de The Qt Company para Python, lo que garantiza mantenimiento continuo y soporte multiplataforma homogéneo.
3. **Licencia:** La licencia LGPLv3 permite su integración en proyectos comerciales o académicos sin forzar a liberar la totalidad del código fuente del proyecto.

## Dificultades encontradas durante la prueba de concepto
* **Integración de Matplotlib:** Inicialmente el gráfico se redimensionaba de forma no deseada al ajustar la ventana. Se resolvió envolviendo el canvas en un `QVBoxLayout` dentro de un `QWidget` contenedor con políticas de expansión adecuadas.
* **Carga desacoplada de la tabla:** Para mostrar datos del módulo de importación sin duplicar la lógica de lectura, se diseñó la función `cargar_datos_en_tabla` para aceptar directamente un objeto `DataFrame` de Pandas cargado externamente.