## Comparativa de Kivy

| Criterio | Evaluación | Descripción |
|---|---|---|
| Facilidad de uso y curva de aprendizaje | Media | Permite crear interfaces con Python, pero requiere aprender su sistema de layouts, eventos y widgets. |
| Compatibilidad multiplataforma | Alta | Compatible con Windows, macOS y Linux. También permite desarrollar aplicaciones móviles. |
| Licencia | MIT | Permite utilizar, modificar y distribuir el software, respetando las condiciones de la licencia. |
| Documentación y comunidad | Buena | Dispone de documentación oficial, tutoriales, ejemplos y una comunidad de desarrolladores. |
| Diálogo de selección de archivos | Disponible | Ofrece `FileChooserListView` y `FileChooserIconView` para explorar y seleccionar archivos. |
| Tablas con scroll | Requiere implementación | Se pueden construir con `GridLayout`, `ScrollView` o `RecycleView`, aunque no dispone de un componente de tabla convencional completo. |
| Desplegable de selección simple | Disponible | El widget `Spinner` permite seleccionar una opción de una lista. |
| Selección múltiple | Requiere implementación | Puede implementarse mediante `CheckBox` u otros widgets combinados. |
| Áreas de texto | Disponible | `TextInput` permite introducir texto de una o varias líneas. |
| Integración con Matplotlib | Posible | Puede realizarse mediante `kivy-garden.matplotlib`, comprobando la compatibilidad entre versiones. |
| Instalación con uv | Compatible | Se puede añadir al proyecto mediante `uv add kivy`, verificando los requisitos de Python y del sistema operativo. |

## Conclusión

Kivy es una alternativa adecuada para desarrollar interfaces gráficas multiplataforma y personalizables. Sin embargo, para una aplicación orientada al análisis de datos, es necesario evaluar el esfuerzo de implementación de tablas, selección múltiple e integración con Matplotlib antes de tomar una decisión definitiva.

## Referencias

- [Documentación oficial de Kivy](https://kivy.org/doc/stable/)
- [Referencia de widgets](https://kivy.org/doc/stable/api-kivy.uix.html)
- [Guía de instalación](https://kivy.org/doc/stable/gettingstarted/installation.html)
- [Kivy en PyPI](https://pypi.org/project/Kivy/)
- [Licencia de Kivy](https://github.com/kivy/kivy/blob/master/LICENSE)
