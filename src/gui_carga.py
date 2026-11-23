import html
import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from PySide6.QtCore import Qt, QAbstractTableModel, QThread, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QTableView,
    QLineEdit, QMessageBox, QHeaderView, QLabel, QListWidget,
    QComboBox, QAbstractItemView, QGroupBox, QScrollArea,
    QTextEdit, QProgressBar, QFrame
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from src.preprocesado import (
        validar_seleccion_columnas,
        detectar_valores_inexistentes,
        tratar_valores_inexistentes,
        hay_valores_inexistentes,
        parsear_valor_constante,
        ConjuntoVacioError,
        separar_train_test,
        parsear_parametros_separacion,
        PORCENTAJE_TEST_POR_DEFECTO,
        SEMILLA_POR_DEFECTO,
    )
    from src.importacion import importar_datos
    from src.modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
        normalizar_ruta_guardado,
        predecir_salida,
        ModeloInvalidoError,
    )
    try:
        from src.modelo import cargar_modelo as cargar_modelo_archivo
    except ImportError:
        cargar_modelo_archivo = None
except ImportError:
    from preprocesado import (
        validar_seleccion_columnas,
        detectar_valores_inexistentes,
        tratar_valores_inexistentes,
        hay_valores_inexistentes,
        parsear_valor_constante,
        ConjuntoVacioError,
        separar_train_test,
        parsear_parametros_separacion,
        PORCENTAJE_TEST_POR_DEFECTO,
        SEMILLA_POR_DEFECTO,
    )
    from importacion import importar_datos
    from modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
        normalizar_ruta_guardado,
        predecir_salida,
        ModeloInvalidoError,
    )
    try:
        from modelo import cargar_modelo as cargar_modelo_archivo
    except ImportError:
        cargar_modelo_archivo = None


ESTILO_MODERNO = """
QMainWindow {
    background-color: #F3F4F6;
}
QWidget {
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #1F2937;
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    margin-top: 12px;
    font-weight: bold;
    font-size: 14px;
    color: #111827;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    background-color: #FFFFFF;
    color: #2563EB;
}
QGroupBox:disabled {
    color: #9CA3AF;
}
QPushButton {
    background-color: #F3F4F6;
    color: #374151;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #E5E7EB;
    border-color: #9CA3AF;
}
QPushButton:disabled {
    background-color: #F9FAFB;
    color: #9CA3AF;
    border-color: #E5E7EB;
}
QLineEdit, QComboBox, QListWidget, QTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 5px 8px;
}
QLineEdit:focus, QComboBox:focus, QListWidget:focus, QTextEdit:focus {
    border-color: #2563EB;
}
QTextEdit:disabled, QLineEdit:disabled, QComboBox:disabled, QListWidget:disabled {
    background-color: #F9FAFB;
    color: #9CA3AF;
    border-color: #E5E7EB;
}
QTableWidget, QTableView {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    gridline-color: #F3F4F6;
}
QHeaderView::section {
    background-color: #F9FAFB;
    color: #4B5563;
    font-weight: bold;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #E5E7EB;
}
QStatusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E5E7EB;
    color: #374151;
    font-size: 12px;
}
"""


# -------------------------------------------------------------
# Modelo de tabla con Scroll Virtual (Historia #37)
# -------------------------------------------------------------
class ModeloTablaDataFrame(QAbstractTableModel):
    """
    Modelo virtual para QTableView: solo consulta y dibuja las celdas
    visibles en pantalla, permitiendo recorrer datasets de decenas de
    miles de filas con total fluidez.
    """
    def __init__(
        self,
        df: pd.DataFrame = None,
        etiquetas: list[str] = None,
        resaltar_ultima_columna: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self._etiquetas = etiquetas or [str(c) for c in self._df.columns]
        self._resaltar_ultima_columna = resaltar_ultima_columna
        self._fondo_target = QBrush(QColor("#DBEAFE"))
        self._texto_target = QBrush(QColor("#1E40AF"))

    def rowCount(self, parent=None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent=None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return self._df.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        fila = index.row()
        col = index.column()

        if role == Qt.DisplayRole:
            val = self._df.iat[fila, col]
            return "" if pd.isna(val) else str(val)

        if self._resaltar_ultima_columna and col == self.columnCount() - 1:
            if role == Qt.BackgroundRole:
                return self._fondo_target
            if role == Qt.ForegroundRole:
                return self._texto_target

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if 0 <= section < len(self._etiquetas):
                return str(self._etiquetas[section])
            return ""
        return str(section + 1)


# -------------------------------------------------------------
# Hilo secundario para carga asíncrona de archivos (Historia #37)
# -------------------------------------------------------------
class HiloCargaDataset(QThread):
    carga_completada = Signal(object, str)
    carga_fallida = Signal(object)

    def __init__(self, ruta_absoluta: str, parent=None):
        super().__init__(parent)
        self.ruta_absoluta = ruta_absoluta

    def run(self):
        try:
            df_nuevo = importar_datos(self.ruta_absoluta)
            df_nuevo.columns = [str(c) for c in df_nuevo.columns]
            self.carga_completada.emit(df_nuevo, self.ruta_absoluta)
        except Exception as err:
            self.carga_fallida.emit(err)


# -------------------------------------------------------------
# Traducción de errores técnicos a mensajes amigables
# -------------------------------------------------------------
def formatear_error_amigable(error: Exception, accion: str) -> str:
    """
    Convierte excepciones técnicas en mensajes claros que explican qué falló
    y cómo solucionarlo, sin exponer trazas ni mensajes técnicos de librerías.
    """
    if isinstance(error, FileNotFoundError):
        return (
            f"No se encontró el archivo indicado al {accion}.\n"
            "Comprueba que el archivo sigue existiendo en esa ubicación y vuelve a intentarlo."
        )
    if isinstance(error, PermissionError):
        return (
            f"No tienes permisos suficientes para acceder a la ubicación elegida al {accion}.\n"
            "Cierra el archivo si está abierto en otro programa o elige otra carpeta (por ejemplo, Documentos)."
        )
    if isinstance(error, IsADirectoryError):
        return (
            f"La ruta seleccionada corresponde a una carpeta y no a un archivo al {accion}.\n"
            "Selecciona un archivo válido e inténtalo de nuevo."
        )
    if isinstance(error, ModeloInvalidoError):
        # Mensaje ya redactado para el usuario desde modelo.cargar_modelo (Historia #115)
        return str(error)
    if isinstance(error, json.JSONDecodeError):
        return (
            "El archivo seleccionado no tiene un formato de modelo válido o está dañado.\n"
            "Asegúrate de elegir un archivo .json generado por esta aplicación."
        )
    if isinstance(error, UnicodeDecodeError):
        return (
            "No se pudo leer el contenido del archivo debido a su codificación de caracteres.\n"
            "Guarda el archivo en formato UTF-8 y vuelve a cargarlo."
        )
    if isinstance(error, MemoryError):
        return (
            "El archivo es demasiado grande para la memoria disponible del equipo.\n"
            "Cierra otras aplicaciones o utiliza un conjunto de datos más reducido."
        )
    if isinstance(error, ConjuntoVacioError):
        return (
            "Al aplicar esta estrategia uno de los conjuntos queda sin filas.\n"
            "Elige otra estrategia de tratamiento (como rellenar con la media o la mediana) para conservar los datos."
        )

    texto_original = str(error).strip()
    indicadores_tecnicos = (
        "Traceback", "KeyError", "TypeError", "AttributeError",
        "sqlite3.", "pandas", "sklearn", "numpy", "errno", "[Errno",
        "could not convert string to float", "Expected 2D array",
    )
    es_tecnico = (
        not texto_original
        or "\n" in texto_original and "File \"" in texto_original
        or any(ind.lower() in texto_original.lower() for ind in indicadores_tecnicos)
    )

    if isinstance(error, ValueError) and not es_tecnico:
        return f"{texto_original}\nRevisa los valores introducidos y vuelve a intentarlo."

    if not es_tecnico and isinstance(error, OSError):
        return (
            f"No se pudo completar la operación de disco al {accion}.\n"
            "Comprueba que tienes espacio libre y permisos en la carpeta seleccionada."
        )

    return (
        f"No se pudo completar la operación al {accion}.\n"
        "Revisa el archivo o los parámetros seleccionados y vuelve a intentarlo."
    )


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Integrado de Análisis y Regresión Lineal")
        # Dimensiones optimizadas para pantallas de 1366 x 768 px (Historia #37)
        self.resize(1100, 700)
        self.setMinimumSize(860, 560)
        self.setStyleSheet(ESTILO_MODERNO)

        # Estado global
        self.df_cargado: pd.DataFrame = None
        self.df_train: pd.DataFrame = None
        self.df_test: pd.DataFrame = None
        self.ruta_archivo_actual: str = ""
        self.features_confirmadas: list = []
        self.target_confirmado: str = ""
        self.porcentaje_test_usado: float = PORCENTAJE_TEST_POR_DEFECTO
        self.semilla_usada: int = SEMILLA_POR_DEFECTO
        self.resultado_modelo = None
        self.origen_modelo: str = ""  # "creado", "cargado" o ""
        self.nombre_archivo_modelo: str = ""
        self.ruta_archivo_modelo: str = ""  # Ruta completa del último archivo de modelo cargado/guardado (Historia #115)
        self.cambios_sin_guardar: bool = False
        self._cargando_archivo: bool = False
        self._hilo_carga: HiloCargaDataset | None = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #F3F4F6; }")
        self.setCentralWidget(scroll)

        contenedor_scroll = QWidget()
        scroll.setWidget(contenedor_scroll)
        layout_principal = QVBoxLayout(contenedor_scroll)
        layout_principal.setSpacing(14)
        layout_principal.setContentsMargins(14, 14, 14, 14)

        # -------------------------------------------------------------
        # Indicación inicial de cómo empezar (Historia #37)
        # -------------------------------------------------------------
        self.banner_inicio = QFrame()
        self.banner_inicio.setStyleSheet(
            "QFrame { background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 8px; }"
        )
        layout_banner = QHBoxLayout(self.banner_inicio)
        layout_banner.setContentsMargins(10, 6, 10, 6)

        self.lbl_guia_inicio = QLabel(
            "<b>¿Cómo empezar?</b> Puedes <b>cargar un dataset</b> (Paso 1) para entrenar un nuevo modelo "
            "siguiendo los pasos en orden, o <b>cargar un modelo guardado</b> (.json) para consultar su fórmula, "
            "editar su descripción y hacer predicciones directamente."
        )
        self.lbl_guia_inicio.setWordWrap(True)
        self.lbl_guia_inicio.setStyleSheet("border: none; color: #1E3A8A; font-size: 13px;")
        layout_banner.addWidget(self.lbl_guia_inicio, 1)

        self.btn_cargar_modelo = QPushButton("📂 Cargar modelo guardado")
        self.btn_cargar_modelo.setStyleSheet(
            "QPushButton { background-color: #1D4ED8; color: white; border: none; padding: 8px 14px; } "
            "QPushButton:hover { background-color: #1E40AF; } "
            "QPushButton:disabled { background-color: #93C5FD; color: #EFF6FF; }"
        )
        self.btn_cargar_modelo.clicked.connect(self.seleccionar_y_cargar_modelo)
        layout_banner.addWidget(self.btn_cargar_modelo)

        layout_principal.addWidget(self.banner_inicio)

        # -------------------------------------------------------------
        # 1. Cargar datos
        # -------------------------------------------------------------
        self.group_carga = QGroupBox("1. Cargar datos")
        layout_carga = QVBoxLayout()

        layout_superior = QHBoxLayout()
        self.btn_cargar = QPushButton("📁 Cargar Dataset")
        self.btn_cargar.setStyleSheet(
            "QPushButton { background-color: #2563EB; color: white; border: none; } "
            "QPushButton:hover { background-color: #1D4ED8; } "
            "QPushButton:disabled { background-color: #93C5FD; color: #EFF6FF; }"
        )
        self.btn_cargar.clicked.connect(self.seleccionar_y_cargar_archivo)

        self.input_ruta = QLineEdit()
        self.input_ruta.setReadOnly(True)
        self.input_ruta.setPlaceholderText("Ningún archivo cargado...")

        layout_superior.addWidget(self.btn_cargar)
        layout_superior.addWidget(self.input_ruta)
        layout_carga.addLayout(layout_superior)

        # Indicador visual en línea durante la carga de archivos grandes
        layout_progreso_carga = QHBoxLayout()
        self.lbl_progreso_carga = QLabel("")
        self.lbl_progreso_carga.setStyleSheet("color: #2563EB; font-weight: 600;")
        self.lbl_progreso_carga.setVisible(False)
        self.barra_carga = QProgressBar()
        self.barra_carga.setRange(0, 0)  # Modo indeterminado
        self.barra_carga.setFixedHeight(14)
        self.barra_carga.setVisible(False)
        layout_progreso_carga.addWidget(self.lbl_progreso_carga)
        layout_progreso_carga.addWidget(self.barra_carga)
        layout_carga.addLayout(layout_progreso_carga)

        self.lbl_info_tabla = QLabel("Sin datos cargados.")
        self.lbl_info_tabla.setStyleSheet("color: #4B5563; font-size: 12px;")
        layout_carga.addWidget(self.lbl_info_tabla)

        # QTableView con scroll virtual (Historia #37)
        self.tabla_datos = QTableView()
        self.tabla_datos.setAlternatingRowColors(True)
        self.tabla_datos.setMinimumHeight(180)
        self.tabla_datos.setMaximumHeight(240)
        self.tabla_datos.setModel(ModeloTablaDataFrame())
        layout_carga.addWidget(self.tabla_datos)

        self.group_carga.setLayout(layout_carga)
        layout_principal.addWidget(self.group_carga)

        # -------------------------------------------------------------
        # 2. Seleccionar columnas
        # -------------------------------------------------------------
        self.group_seleccion_col = QGroupBox("2. Seleccionar columnas")
        layout_sel_box = QVBoxLayout()

        layout_seleccion = QHBoxLayout()
        layout_features = QVBoxLayout()
        self.lbl_features = QLabel("Columnas de entrada (Features):")
        self.lbl_features.setStyleSheet("font-weight: 600; color: #374151;")
        self.lista_features = QListWidget()
        self.lista_features.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lista_features.setFixedHeight(100)
        layout_features.addWidget(self.lbl_features)
        layout_features.addWidget(self.lista_features)

        layout_target = QVBoxLayout()
        self.lbl_target = QLabel("Columna de salida (Target):")
        self.lbl_target.setStyleSheet("font-weight: 600; color: #374151;")
        self.combo_target = QComboBox()
        self.combo_target.currentTextChanged.connect(self.al_cambiar_target)
        layout_target.addWidget(self.lbl_target)
        layout_target.addWidget(self.combo_target)
        layout_target.addStretch()

        layout_seleccion.addLayout(layout_features, 2)
        layout_seleccion.addLayout(layout_target, 1)
        layout_sel_box.addLayout(layout_seleccion)

        self.btn_confirmar = QPushButton("✓ Confirmar Selección")
        self.btn_confirmar.setEnabled(False)
        self.btn_confirmar.clicked.connect(self.confirmar_seleccion_columnas)
        layout_sel_box.addWidget(self.btn_confirmar)

        self.lbl_seleccion_confirmada = QLabel()
        self.lbl_seleccion_confirmada.setWordWrap(True)
        layout_sel_box.addWidget(self.lbl_seleccion_confirmada)

        # QTableView con scroll virtual para las columnas seleccionadas
        self.tabla_seleccion = QTableView()
        self.tabla_seleccion.setAlternatingRowColors(True)
        self.tabla_seleccion.setMinimumHeight(130)
        self.tabla_seleccion.setMaximumHeight(180)
        self.tabla_seleccion.setModel(ModeloTablaDataFrame())
        layout_sel_box.addWidget(self.tabla_seleccion)
        self.mostrar_seleccion_confirmada()

        self.group_seleccion_col.setLayout(layout_sel_box)
        layout_principal.addWidget(self.group_seleccion_col)

        # -------------------------------------------------------------
        # 3. Separar datos
        # -------------------------------------------------------------
        self.group_separacion = QGroupBox("3. Separar datos (Train / Test)")
        layout_sep = QVBoxLayout()

        layout_params = QHBoxLayout()
        layout_params.addWidget(QLabel("% Test:"))
        self.input_porcentaje_test = QLineEdit(str(PORCENTAJE_TEST_POR_DEFECTO))
        self.input_porcentaje_test.setFixedWidth(65)
        self.input_porcentaje_test.textChanged.connect(self.actualizar_porcentaje_train)
        layout_params.addWidget(self.input_porcentaje_test)

        self.lbl_porcentaje_train = QLabel()
        self.lbl_porcentaje_train.setStyleSheet("font-weight: 600;")
        layout_params.addWidget(self.lbl_porcentaje_train)

        layout_params.addWidget(QLabel("Semilla:"))
        self.input_semilla = QLineEdit(str(SEMILLA_POR_DEFECTO))
        self.input_semilla.setFixedWidth(65)
        layout_params.addWidget(self.input_semilla)
        layout_params.addStretch()

        layout_sep.addLayout(layout_params)

        self.btn_separar = QPushButton("⚡ Separar Datos")
        self.btn_separar.setEnabled(False)
        self.btn_separar.clicked.connect(self.separar_datos)
        layout_sep.addWidget(self.btn_separar)

        self.lbl_resultado_separacion = QLabel()
        self.lbl_resultado_separacion.setWordWrap(True)
        layout_sep.addWidget(self.lbl_resultado_separacion)

        self.group_separacion.setLayout(layout_sep)
        layout_principal.addWidget(self.group_separacion)
        self.actualizar_porcentaje_train(self.input_porcentaje_test.text())

        # -------------------------------------------------------------
        # 4. Preprocesar
        # -------------------------------------------------------------
        self.group_preprocesado = QGroupBox("4. Preprocesar (Valores Nulos)")
        layout_prep = QVBoxLayout()

        self.lbl_reporte_nulos = QLabel("Estado: Pendiente de separación en Train/Test")
        self.lbl_reporte_nulos.setWordWrap(True)
        layout_prep.addWidget(self.lbl_reporte_nulos)

        layout_estrategia = QHBoxLayout()
        layout_estrategia.addWidget(QLabel("Estrategia:"))
        self.combo_estrategia = QComboBox()
        self.combo_estrategia.addItems([
            "Eliminar filas con valores inexistentes",
            "Rellenar con la media",
            "Rellenar con la mediana",
            "Rellenar con valor constante"
        ])
        self.combo_estrategia.currentIndexChanged.connect(self.al_cambiar_estrategia)

        self.input_constante = QLineEdit()
        self.input_constante.setPlaceholderText("Valor constante (ej. 0)")
        self.input_constante.setEnabled(False)

        layout_estrategia.addWidget(self.combo_estrategia)
        layout_estrategia.addWidget(self.input_constante)
        layout_prep.addLayout(layout_estrategia)

        self.btn_aplicar_preprocesado = QPushButton("🛠️ Aplicar Preprocesado")
        self.btn_aplicar_preprocesado.setEnabled(False)
        self.btn_aplicar_preprocesado.clicked.connect(self.ejecutar_preprocesado)
        layout_prep.addWidget(self.btn_aplicar_preprocesado)

        self.group_preprocesado.setLayout(layout_prep)
        layout_principal.addWidget(self.group_preprocesado)

        # -------------------------------------------------------------
        # 5. Crear y guardar modelo
        # -------------------------------------------------------------
        self.group_modelo = QGroupBox("5. Crear y evaluar modelo")
        layout_mod = QVBoxLayout()

        layout_botones_modelo = QHBoxLayout()

        self.btn_crear_modelo = QPushButton("🚀 Crear Modelo")
        self.btn_crear_modelo.setFixedHeight(40)
        self.btn_crear_modelo.setStyleSheet("""
            QPushButton { font-size: 14px; font-weight: bold; background-color: #059669; color: white; border: none; border-radius: 6px; }
            QPushButton:hover { background-color: #047857; }
            QPushButton:disabled { background-color: #E5E7EB; color: #9CA3AF; }
        """)
        self.btn_crear_modelo.clicked.connect(self.ejecutar_creacion_modelo)
        layout_botones_modelo.addWidget(self.btn_crear_modelo)

        self.btn_guardar_modelo = QPushButton("💾 Guardar modelo")
        self.btn_guardar_modelo.setFixedHeight(40)
        self.btn_guardar_modelo.setEnabled(False)
        self.btn_guardar_modelo.setStyleSheet("""
            QPushButton { font-size: 14px; font-weight: bold; background-color: #2563EB; color: white; border: none; border-radius: 6px; }
            QPushButton:hover { background-color: #1D4ED8; }
            QPushButton:disabled { background-color: #E5E7EB; color: #9CA3AF; }
        """)
        self.btn_guardar_modelo.clicked.connect(self.ejecutar_guardado_modelo)
        layout_botones_modelo.addWidget(self.btn_guardar_modelo)

        layout_mod.addLayout(layout_botones_modelo)

        self.lbl_formula = QLabel("<b>Fórmula del modelo:</b> <i>(Pendiente de creación)</i>")
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setStyleSheet("font-size: 13px; padding: 10px; background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;")
        layout_mod.addWidget(self.lbl_formula)

        # Columnas de entrada y de salida del modelo (Historia #115)
        self.lbl_columnas_modelo = QLabel("")
        self.lbl_columnas_modelo.setWordWrap(True)
        self.lbl_columnas_modelo.setTextFormat(Qt.RichText)
        self.lbl_columnas_modelo.setStyleSheet("font-size: 13px; padding: 6px 10px; color: #374151;")
        self.lbl_columnas_modelo.setVisible(False)
        layout_mod.addWidget(self.lbl_columnas_modelo)

        self.tabla_metricas = QTableWidget(2, 2)
        self.tabla_metricas.setHorizontalHeaderLabels(["Entrenamiento", "Test"])
        self.tabla_metricas.setVerticalHeaderLabels(["R² (Coef. Determinación)", "Error Cuadrático Medio (ECM)"])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla_metricas.setFixedHeight(90)
        layout_mod.addWidget(self.tabla_metricas)

        self.lbl_descripcion_modelo = QLabel("Descripción (opcional):")
        self.lbl_descripcion_modelo.setStyleSheet("font-weight: 600; color: #374151;")
        layout_mod.addWidget(self.lbl_descripcion_modelo)

        self.input_descripcion_modelo = QTextEdit()
        self.input_descripcion_modelo.setPlaceholderText(
            "Descripción opcional: añade notas sobre el ajuste, las métricas o las variables del modelo..."
        )
        self.input_descripcion_modelo.setFixedHeight(75)
        self.input_descripcion_modelo.setEnabled(False)
        self.input_descripcion_modelo.textChanged.connect(self.al_cambiar_descripcion_modelo)
        layout_mod.addWidget(self.input_descripcion_modelo)

        self.canvas_modelo = FigureCanvas(Figure(figsize=(7, 4)))
        self.canvas_modelo.setMinimumHeight(340)
        layout_mod.addWidget(self.canvas_modelo)

        self.group_modelo.setLayout(layout_mod)
        layout_principal.addWidget(self.group_modelo)

        # -------------------------------------------------------------
        # 6. Realizar predicciones
        # -------------------------------------------------------------
        self.group_prediccion = QGroupBox("6. Realizar predicciones")
        self.group_prediccion.setEnabled(False)
        layout_pred = QVBoxLayout()

        self.layout_campos_prediccion = QVBoxLayout()
        layout_pred.addLayout(self.layout_campos_prediccion)
        self.inputs_prediccion: dict[str, QLineEdit] = {}

        self.btn_predecir = QPushButton("🔮 Realizar predicción")
        self.btn_predecir.setFixedHeight(38)
        self.btn_predecir.setStyleSheet("""
            QPushButton { font-size: 14px; font-weight: bold; background-color: #2563EB; color: white; border: none; border-radius: 6px; }
            QPushButton:hover { background-color: #1D4ED8; }
            QPushButton:disabled { background-color: #E5E7EB; color: #9CA3AF; }
        """)
        self.btn_predecir.clicked.connect(self.ejecutar_prediccion)
        layout_pred.addWidget(self.btn_predecir)

        self.lbl_resultado_prediccion = QLabel("Resultado: (No hay modelo disponible)")
        self.lbl_resultado_prediccion.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; color: #1E3A8A;"
        )
        layout_pred.addWidget(self.lbl_resultado_prediccion)

        self.group_prediccion.setLayout(layout_pred)
        layout_principal.addWidget(self.group_prediccion)

        # -------------------------------------------------------------
        # Barra de estado persistente (Historia #37)
        # -------------------------------------------------------------
        barra_estado = self.statusBar()
        self.lbl_estado_archivo = QLabel()
        self.lbl_estado_modelo = QLabel()
        self.lbl_estado_guardado = QLabel()

        for lbl in (self.lbl_estado_archivo, self.lbl_estado_modelo, self.lbl_estado_guardado):
            lbl.setStyleSheet("padding: 2px 8px;")
            barra_estado.addPermanentWidget(lbl)

        self.invalidar_separacion()
        self.actualizar_barra_estado()

    # -------------------------------------------------------------
    # Sistema unificado de mensajes al usuario (Historia #37)
    # -------------------------------------------------------------
    def mostrar_exito(self, titulo: str, mensaje: str):
        QMessageBox.information(self, titulo, mensaje)

    def mostrar_aviso(self, titulo: str, mensaje: str):
        QMessageBox.warning(self, titulo, mensaje)

    def mostrar_error(self, titulo: str, error_o_mensaje, accion: str = "realizar la operación"):
        if isinstance(error_o_mensaje, Exception):
            texto = formatear_error_amigable(error_o_mensaje, accion)
        else:
            texto = str(error_o_mensaje)
        QMessageBox.critical(self, titulo, texto)

    # -------------------------------------------------------------
    # Utilidades de la Historia #115 (cargar modelo guardado)
    # -------------------------------------------------------------
    def _confirmar_descartar_modelo(self) -> bool:
        """
        Pide confirmación antes de sustituir/descartar el modelo actual cuando
        tiene cambios sin guardar (modelo creado sin guardar o modelo cargado
        cuya descripción se ha modificado). Devuelve True si se puede continuar.
        """
        if self.resultado_modelo is None or not self.cambios_sin_guardar:
            return True

        respuesta = QMessageBox.question(
            self,
            "Cambios sin guardar",
            "El modelo actual tiene cambios sin guardar y se perderán si continúas.\n"
            "¿Deseas continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return respuesta == QMessageBox.Yes

    def _mostrar_secciones_datos(self, visible: bool):
        """
        Muestra u oculta las secciones de trabajo con datos. El botón
        «Cargar Dataset» (dentro del Paso 1) se mantiene siempre visible para
        poder volver a trabajar con datos tras cargar un modelo.
        """
        for widget in (
            self.tabla_datos,
            self.lbl_info_tabla,
            self.group_seleccion_col,
            self.group_separacion,
            self.group_preprocesado,
            self.canvas_modelo,
            self.btn_crear_modelo,
        ):
            widget.setVisible(visible)

    def _resetear_datos(self):
        """Descarta el dataset y todo el flujo asociado (columnas, separación, preprocesado y modelo)."""
        self.df_cargado = None
        self.ruta_archivo_actual = ""
        self.input_ruta.clear()
        self.lbl_info_tabla.setText("Sin datos cargados.")
        self.tabla_datos.setModel(ModeloTablaDataFrame())
        self.actualizar_selectores_columnas(None)  # También limpia el estado del modelo

    def _actualizar_columnas_modelo(self):
        """Muestra los nombres de las columnas de entrada y de salida del modelo actual."""
        if self.resultado_modelo is None or not isinstance(self.resultado_modelo, dict):
            self.lbl_columnas_modelo.setText("")
            self.lbl_columnas_modelo.setVisible(False)
            return

        features = self.resultado_modelo.get("features") or self.features_confirmadas
        target = self.resultado_modelo.get("target") or self.target_confirmado
        entradas = ", ".join(html.escape(str(c)) for c in features)
        self.lbl_columnas_modelo.setText(
            f"<b>Columnas de entrada:</b> {entradas}<br>"
            f"<b>Columna de salida:</b> {html.escape(str(target))}"
        )
        self.lbl_columnas_modelo.setVisible(True)

    # -------------------------------------------------------------
    # Barra de estado y control global del flujo
    # -------------------------------------------------------------
    def actualizar_barra_estado(self):
        if self._cargando_archivo:
            self.statusBar().showMessage("Cargando archivo, por favor espera...")
        else:
            self.statusBar().clearMessage()

        if self.ruta_archivo_actual:
            nombre_ds = Path(self.ruta_archivo_actual).name
            filas = len(self.df_cargado) if self.df_cargado is not None else 0
            self.lbl_estado_archivo.setText(f"Dataset: {nombre_ds} ({filas:,} filas)".replace(",", "."))
        else:
            self.lbl_estado_archivo.setText("Dataset: Ninguno")

        if self.resultado_modelo is None:
            self.lbl_estado_modelo.setText("Modelo: Ninguno")
            self.lbl_estado_guardado.setText("")
        else:
            if self.origen_modelo == "cargado":
                sufijo = f" ({self.nombre_archivo_modelo})" if self.nombre_archivo_modelo else ""
                self.lbl_estado_modelo.setText(f"Modelo: Cargado{sufijo}")
            else:
                self.lbl_estado_modelo.setText("Modelo: Creado")

            if self.cambios_sin_guardar:
                self.lbl_estado_guardado.setText("● Cambios sin guardar")
                self.lbl_estado_guardado.setStyleSheet("padding: 2px 8px; color: #B45309; font-weight: bold;")
            else:
                self.lbl_estado_guardado.setText("✓ Guardado")
                self.lbl_estado_guardado.setStyleSheet("padding: 2px 8px; color: #047857; font-weight: bold;")

    def establecer_modo_carga(self, cargando: bool, mensaje: str = ""):
        """Muestra el indicador de operación en curso y bloquea operaciones incompatibles."""
        self._cargando_archivo = cargando
        self.lbl_progreso_carga.setText(mensaje)
        self.lbl_progreso_carga.setVisible(cargando)
        self.barra_carga.setVisible(cargando)
        self.actualizar_estado_flujo()
        self.actualizar_barra_estado()

    def actualizar_estado_flujo(self):
        ocupado = self._cargando_archivo
        hay_dataset = self.df_cargado is not None and not self.df_cargado.empty
        hay_seleccion = bool(self.features_confirmadas and self.target_confirmado)
        hay_separacion = self.df_train is not None and self.df_test is not None
        hay_modelo = self.resultado_modelo is not None

        self.btn_cargar.setEnabled(not ocupado)
        self.btn_cargar_modelo.setEnabled(not ocupado)
        self.group_seleccion_col.setEnabled((not ocupado) and hay_dataset)
        self.btn_confirmar.setEnabled((not ocupado) and hay_dataset)
        self.group_separacion.setEnabled((not ocupado) and hay_dataset and hay_seleccion)
        self.btn_separar.setEnabled((not ocupado) and hay_dataset and hay_seleccion)
        self.group_preprocesado.setEnabled((not ocupado) and hay_separacion)
        self.btn_crear_modelo.setEnabled((not ocupado) and self.listo_para_crear_modelo())
        self.input_descripcion_modelo.setEnabled((not ocupado) and hay_modelo)
        self.btn_guardar_modelo.setEnabled((not ocupado) and hay_modelo)
        self.group_prediccion.setEnabled((not ocupado) and hay_modelo)
        self.btn_predecir.setEnabled((not ocupado) and hay_modelo)

    def limpiar_estado_modelo(self):
        self.resultado_modelo = None
        self.origen_modelo = ""
        self.nombre_archivo_modelo = ""
        self.ruta_archivo_modelo = ""
        self.cambios_sin_guardar = False

        if hasattr(self, "input_descripcion_modelo"):
            self.input_descripcion_modelo.blockSignals(True)
            self.input_descripcion_modelo.clear()
            self.input_descripcion_modelo.blockSignals(False)
            self.input_descripcion_modelo.setEnabled(False)
        if hasattr(self, "btn_guardar_modelo"):
            self.btn_guardar_modelo.setEnabled(False)
        if hasattr(self, "lbl_formula"):
            self.lbl_formula.setText("<b>Fórmula del modelo:</b> <i>(Pendiente de creación)</i>")
        if hasattr(self, "lbl_columnas_modelo"):
            self._actualizar_columnas_modelo()
        if hasattr(self, "tabla_metricas"):
            self.tabla_metricas.clearContents()
        if hasattr(self, "canvas_modelo"):
            self.canvas_modelo.figure.clear()
            self.canvas_modelo.draw()
        if hasattr(self, "group_prediccion"):
            self.actualizar_panel_prediccion()
        if hasattr(self, "lbl_estado_modelo"):
            self.actualizar_barra_estado()

    def al_cambiar_descripcion_modelo(self):
        if self.resultado_modelo is not None:
            actualizar_descripcion_modelo(
                self.resultado_modelo,
                self.input_descripcion_modelo.toPlainText()
            )
            self.cambios_sin_guardar = True
            self.actualizar_barra_estado()

    def _cargar_modelo_json_respaldo(self, ruta_archivo: str) -> dict:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)

        claves_requeridas = {"coeficientes", "intercepto", "formula", "features", "target", "metricas"}
        if not isinstance(datos, dict) or not claves_requeridas.issubset(datos.keys()):
            raise ValueError("El archivo JSON no contiene todos los campos de un modelo válido.")

        modelo_rec = LinearRegression()
        modelo_rec.coef_ = np.array(datos["coeficientes"], dtype=float)
        modelo_rec.intercept_ = float(datos["intercepto"])
        modelo_rec.n_features_in_ = len(datos["features"])
        modelo_rec.feature_names_in_ = np.array(datos["features"], dtype=object)

        return {
            "modelo": modelo_rec,
            "coeficientes": [float(c) for c in datos["coeficientes"]],
            "intercepto": float(datos["intercepto"]),
            "formula": str(datos["formula"]),
            "features": list(datos["features"]),
            "columnas_entrada": list(datos["features"]),
            "target": str(datos["target"]),
            "columna_salida": str(datos["target"]),
            "metricas": datos["metricas"],
            "descripcion": str(datos.get("descripcion", "")),
        }

    def seleccionar_y_cargar_modelo(self):
        if self._cargando_archivo:
            return

        ruta_seleccionada, _ = QFileDialog.getOpenFileName(
            self,
            "Cargar modelo guardado",
            "",
            "Archivos de modelo JSON (*.json);;Todos los archivos (*)",
        )
        if not ruta_seleccionada:
            return

        # Historia #115: confirmar antes de sustituir un modelo con cambios sin guardar
        if not self._confirmar_descartar_modelo():
            return

        try:
            if cargar_modelo_archivo is not None:
                estructura = cargar_modelo_archivo(ruta_seleccionada)
            else:
                estructura = self._cargar_modelo_json_respaldo(ruta_seleccionada)
            self.cargar_modelo_en_interfaz(
                estructura,
                nombre_archivo=Path(ruta_seleccionada).name,
                ruta_archivo=ruta_seleccionada,
            )
            self.mostrar_exito(
                "Modelo cargado",
                f"El modelo «{Path(ruta_seleccionada).name}» se ha cargado correctamente.\n"
                "Ya puedes consultar su fórmula y métricas, editar su descripción o realizar predicciones."
            )
        except Exception as err:
            self.mostrar_error("Error al cargar el modelo", err, "cargar el modelo")

    def cargar_modelo_en_interfaz(self, estructura_modelo: dict, nombre_archivo: str = "", ruta_archivo: str = ""):
        # Historia #115: el archivo del modelo no contiene datos, así que se descarta
        # el dataset actual y se ocultan las secciones de trabajo con datos.
        # Se hace ANTES de asignar el modelo porque el reseteo limpia el estado del modelo.
        self._resetear_datos()
        self._mostrar_secciones_datos(False)

        if not nombre_archivo and ruta_archivo:
            nombre_archivo = Path(ruta_archivo).name

        self.resultado_modelo = estructura_modelo
        self.origen_modelo = "cargado"
        self.nombre_archivo_modelo = nombre_archivo
        self.ruta_archivo_modelo = ruta_archivo
        self.cambios_sin_guardar = False

        descripcion = estructura_modelo.get("descripcion", "") if isinstance(estructura_modelo, dict) else ""
        self.input_descripcion_modelo.blockSignals(True)
        self.input_descripcion_modelo.setPlainText(descripcion)
        self.input_descripcion_modelo.blockSignals(False)

        self.lbl_formula.setText(f"<b>Fórmula del modelo:</b> {estructura_modelo.get('formula', '')}")
        self._actualizar_columnas_modelo()

        m = estructura_modelo.get("metricas", {})
        self._rellenar_tabla_metricas(m)

        # Limpiar gráfico previo e informar que el archivo no guarda datos del dataset
        fig = self.canvas_modelo.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Modelo cargado desde archivo.\nLos modelos guardados no almacenan los datos del dataset original.",
            ha="center", va="center", fontsize=11, color="#4B5563"
        )
        self.canvas_modelo.draw()

        self.actualizar_panel_prediccion()
        self.actualizar_estado_flujo()
        self.actualizar_barra_estado()

    def _rellenar_tabla_metricas(self, m: dict):
        valores = [
            (0, 0, m.get("r2_train")),
            (0, 1, m.get("r2_test")),
            (1, 0, m.get("ecm_train")),
            (1, 1, m.get("ecm_test")),
        ]
        for fila, col, valor in valores:
            texto = f"{float(valor):.4f}" if valor is not None else "—"
            item = QTableWidgetItem(texto)
            item.setTextAlignment(Qt.AlignCenter)
            self.tabla_metricas.setItem(fila, col, item)

    def actualizar_panel_prediccion(self):
        while self.layout_campos_prediccion.count():
            item = self.layout_campos_prediccion.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()
        self.inputs_prediccion.clear()

        if self.resultado_modelo is None or not isinstance(self.resultado_modelo, dict):
            self.group_prediccion.setEnabled(False)
            self.lbl_resultado_prediccion.setText("Resultado: (No hay modelo disponible)")
            return

        features = self.resultado_modelo.get("features") or self.features_confirmadas
        target = self.resultado_modelo.get("target") or self.target_confirmado

        for col in features:
            fila = QHBoxLayout()
            lbl = QLabel(f"{col}:")
            lbl.setMinimumWidth(150)
            lbl.setStyleSheet("font-weight: 600;")
            campo = QLineEdit()
            campo.setPlaceholderText(f"Introduce {col} (ej. 3.5 o 3,5)")
            fila.addWidget(lbl)
            fila.addWidget(campo)
            self.layout_campos_prediccion.addLayout(fila)
            self.inputs_prediccion[col] = campo

        self.lbl_resultado_prediccion.setText(f"{target} = (Pendiente de cálculo)")
        self.group_prediccion.setEnabled(not self._cargando_archivo)

    def ejecutar_prediccion(self):
        if self.resultado_modelo is None:
            return

        modelo = self.resultado_modelo.get("modelo")
        features = self.resultado_modelo.get("features") or self.features_confirmadas
        target = self.resultado_modelo.get("target") or self.target_confirmado
        valores_crudos = {col: inp.text() for col, inp in self.inputs_prediccion.items()}

        try:
            valor_predicho = predecir_salida(modelo, features, valores_crudos, decimales=4)
            self.lbl_resultado_prediccion.setText(f"{target} = {valor_predicho}")
        except ValueError as err:
            self.mostrar_aviso("Datos de entrada no válidos", formatear_error_amigable(err, "realizar la predicción"))
        except Exception as err:
            self.mostrar_error("Error al predecir", err, "realizar la predicción")

    def invalidar_separacion(self):
        self.df_train = None
        self.df_test = None
        self.limpiar_estado_modelo()
        self.lbl_resultado_separacion.setText("Pendiente: confirma la selección de columnas y pulsa «Separar Datos».")
        self.actualizar_estado_flujo()
        self.actualizar_reporte_nulos()
        self.actualizar_barra_estado()

    def listo_para_crear_modelo(self) -> bool:
        columnas = self.columnas_evaluadas()
        if self.df_train is None or self.df_test is None or not columnas:
            return False
        return not hay_valores_inexistentes(self.df_train, self.df_test, columnas)

    def ejecutar_creacion_modelo(self):
        if not self.listo_para_crear_modelo():
            self.mostrar_aviso(
                "Requisitos incompletos",
                "Para crear el modelo debes haber separado los datos y eliminado los valores inexistentes."
            )
            return

        try:
            X_train = self.df_train[self.features_confirmadas]
            y_train = self.df_train[self.target_confirmado]
            X_test = self.df_test[self.features_confirmadas]
            y_test = self.df_test[self.target_confirmado]

            res = entrenar_modelo_regresion(X_train, y_train, X_test, y_test)
            res.setdefault("descripcion", "")
            res.setdefault("features", list(self.features_confirmadas))
            res.setdefault("target", self.target_confirmado)
            self.resultado_modelo = res
            self.origen_modelo = "creado"
            self.nombre_archivo_modelo = ""
            self.ruta_archivo_modelo = ""
            self.cambios_sin_guardar = True

            self.input_descripcion_modelo.blockSignals(True)
            self.input_descripcion_modelo.clear()
            self.input_descripcion_modelo.blockSignals(False)

            self.lbl_formula.setText(f"<b>Fórmula del modelo:</b> {res['formula']}")
            self._actualizar_columnas_modelo()
            self._rellenar_tabla_metricas(res["metricas"])
            self.dibujar_grafico_modelo(
                X_train, y_train, X_test, y_test,
                res["y_pred_train"], res["y_pred_test"], res["modelo"]
            )
            self.actualizar_panel_prediccion()
            self.actualizar_estado_flujo()
            self.actualizar_barra_estado()
            self.mostrar_exito(
                "Modelo creado",
                "El modelo de regresión lineal se ha creado y evaluado correctamente."
            )

        except Exception as err:
            self.mostrar_error("Error al crear el modelo", err, "crear el modelo")

    def ejecutar_guardado_modelo(self):
        if self.resultado_modelo is None:
            return

        descripcion_actual = self.input_descripcion_modelo.toPlainText().strip()
        actualizar_descripcion_modelo(self.resultado_modelo, self.input_descripcion_modelo.toPlainText())

        if not descripcion_actual:
            respuesta_aviso = QMessageBox.warning(
                self,
                "Descripción vacía",
                "El modelo no tiene descripción. ¿Deseas guardar el modelo sin descripción?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if respuesta_aviso == QMessageBox.No:
                return

        # Historia #115: si el modelo viene de un archivo, se propone su misma carpeta y nombre
        nombre_sugerido = self.ruta_archivo_modelo or self.nombre_archivo_modelo or "modelo_lineal.json"
        ruta_elegida, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar modelo",
            nombre_sugerido,
            "Archivos JSON (*.json);;Todos los archivos (*)",
            options=QFileDialog.DontConfirmOverwrite,
        )
        if not ruta_elegida:
            return

        ruta_final = normalizar_ruta_guardado(ruta_elegida)

        if ruta_final.exists():
            confirmar_sobrescritura = QMessageBox.question(
                self,
                "Confirmar sobrescritura",
                f"El archivo «{ruta_final.name}» ya existe.\n¿Deseas sobrescribirlo?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirmar_sobrescritura != QMessageBox.Yes:
                return

        try:
            ruta_guardada = guardar_modelo(self.resultado_modelo, ruta_final)
            self.cambios_sin_guardar = False
            self.nombre_archivo_modelo = Path(ruta_guardada).name
            self.ruta_archivo_modelo = str(ruta_guardada)
            self.actualizar_barra_estado()
            self.mostrar_exito(
                "Modelo guardado",
                f"El modelo se ha guardado correctamente en:\n{ruta_guardada}",
            )
        except Exception as err:
            self.mostrar_error("Error al guardar el modelo", err, "guardar el modelo")

    def dibujar_grafico_modelo(self, X_tr, y_tr, X_te, y_te, y_pred_tr, y_pred_te, modelo):
        fig = self.canvas_modelo.figure
        fig.clear()
        ax = fig.add_subplot(111)

        es_regresion_simple = X_tr.shape[1] == 1

        if es_regresion_simple:
            col_x = X_tr.columns[0]
            ax.scatter(X_tr[col_x], y_tr, color="#2563EB", label="Entrenamiento", alpha=0.6, edgecolors="none", s=30)
            ax.scatter(X_te[col_x], y_te, color="#F97316", label="Test", alpha=0.6, edgecolors="none", s=30)

            x_min = min(X_tr[col_x].min(), X_te[col_x].min())
            x_max = max(X_tr[col_x].max(), X_te[col_x].max())
            x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            y_line = modelo.predict(pd.DataFrame(x_line, columns=[col_x]))

            ax.plot(x_line, y_line, color="#DC2626", linewidth=2.5, label="Recta de ajuste")
            ax.set_xlabel(col_x, fontsize=10, fontweight="bold")
            ax.set_ylabel(y_tr.name, fontsize=10, fontweight="bold")
        else:
            ax.scatter(y_tr, y_pred_tr, color="#2563EB", label="Entrenamiento", alpha=0.6, edgecolors="none", s=30)
            ax.scatter(y_te, y_pred_te, color="#F97316", label="Test", alpha=0.6, edgecolors="none", s=30)

            val_min = min(y_tr.min(), y_te.min())
            val_max = max(y_tr.max(), y_te.max())
            ax.plot([val_min, val_max], [val_min, val_max], color="#DC2626", linestyle="--", linewidth=2, label="Valor real = Predicho")

            ax.set_xlabel("Valor real", fontsize=10, fontweight="bold")
            ax.set_ylabel("Valor predicho", fontsize=10, fontweight="bold")

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E5E7EB")
        ax.set_title("Gráfico de Ajuste del Modelo", fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        self.canvas_modelo.draw()

    def actualizar_porcentaje_train(self, texto: str):
        try:
            pct_test = float(texto.strip().replace(",", "."))
            if 1 <= pct_test <= 99:
                self.lbl_porcentaje_train.setText(f"→ Entrenamiento: {100 - pct_test:g} %")
                return
        except ValueError:
            pass
        self.lbl_porcentaje_train.setText("→ Entrenamiento: —")

    def mostrar_resultado_separacion(self):
        n_train, n_test = len(self.df_train), len(self.df_test)
        self.lbl_resultado_separacion.setText(
            f"<b>División realizada correctamente</b> "
            f"(test: {self.porcentaje_test_usado:g} %, entrenamiento: {100 - self.porcentaje_test_usado:g} %, "
            f"semilla: {self.semilla_usada}).<br>"
            f"<b>Entrenamiento:</b> {n_train} filas | <b>Test:</b> {n_test} filas"
        )

    def separar_datos(self):
        if self.df_cargado is None or not (self.features_confirmadas and self.target_confirmado):
            self.mostrar_aviso("Paso previo pendiente", "Primero debes cargar un dataset y confirmar la selección de columnas.")
            return

        try:
            pct_test, semilla = parsear_parametros_separacion(
                self.input_porcentaje_test.text(), self.input_semilla.text()
            )
            df_train, df_test = separar_train_test(self.df_cargado, pct_test, semilla)
        except Exception as err:
            self.mostrar_error("Error en la separación", err, "separar los datos")
            return

        self.df_train = df_train
        self.df_test = df_test
        self.limpiar_estado_modelo()
        self.porcentaje_test_usado = pct_test
        self.semilla_usada = semilla

        self.mostrar_resultado_separacion()
        self.actualizar_estado_flujo()
        self.actualizar_reporte_nulos()
        self.mostrar_exito(
            "Separación completada",
            f"Datos divididos correctamente.\nEntrenamiento: {len(df_train)} filas | Test: {len(df_test)} filas"
        )

    def al_cambiar_estrategia(self, index: int):
        self.input_constante.setEnabled(index == 3)

    def columnas_evaluadas(self) -> list:
        return self.features_confirmadas + ([self.target_confirmado] if self.target_confirmado else [])

    def actualizar_reporte_nulos(self):
        if self.df_train is None or self.df_test is None:
            self.lbl_reporte_nulos.setText("Estado: Pendiente de separación en Train/Test.")
            self.btn_aplicar_preprocesado.setEnabled(False)
            self.btn_crear_modelo.setEnabled(False)
            return

        columnas_evaluar = self.columnas_evaluadas()
        if not columnas_evaluar:
            self.lbl_reporte_nulos.setText("Selecciona y confirma las columnas primero.")
            self.btn_aplicar_preprocesado.setEnabled(False)
            self.btn_crear_modelo.setEnabled(False)
            return

        reporte = detectar_valores_inexistentes(self.df_train, self.df_test, columnas_evaluar)
        reporte_train, reporte_test = reporte["entrenamiento"], reporte["test"]
        total_nulos_train, total_nulos_test = sum(reporte_train.values()), sum(reporte_test.values())

        if total_nulos_train == 0 and total_nulos_test == 0:
            self.lbl_reporte_nulos.setText(
                "✓ No hay valores inexistentes en las columnas seleccionadas. "
                "Los conjuntos están listos para crear el modelo."
            )
            self.btn_aplicar_preprocesado.setEnabled(False)
        else:
            filas = "".join(
                f"<tr><td>{html.escape(str(col))}</td>"
                f"<td align='center'>{reporte_train[col]}</td>"
                f"<td align='center'>{reporte_test[col]}</td></tr>"
                for col in columnas_evaluar
            )
            txt = (
                "<b>Valores inexistentes detectados:</b>"
                "<table border='1' cellspacing='0' cellpadding='4' style='margin-top: 6px; border-color: #E5E7EB;'>"
                f"<tr style='background-color: #F9FAFB;'><th>Columna</th><th>Entrenamiento ({len(self.df_train)} filas)</th>"
                f"<th>Test ({len(self.df_test)} filas)</th></tr>"
                f"{filas}"
                f"<tr><td><b>Total</b></td><td align='center'><b>{total_nulos_train}</b></td>"
                f"<td align='center'><b>{total_nulos_test}</b></td></tr>"
                "</table>"
            )
            self.lbl_reporte_nulos.setText(txt)
            self.btn_aplicar_preprocesado.setEnabled(not self._cargando_archivo)

        self.btn_crear_modelo.setEnabled((not self._cargando_archivo) and self.listo_para_crear_modelo())

    def ejecutar_preprocesado(self):
        if self.df_train is None or self.df_test is None:
            return

        columnas_evaluar = self.columnas_evaluadas()
        estrategias_map = {0: "eliminar", 1: "media", 2: "mediana", 3: "constante"}
        estrategia = estrategias_map.get(self.combo_estrategia.currentIndex(), "eliminar")

        valor_constante = None
        if estrategia == "constante":
            texto_constante = self.input_constante.text().strip()
            if not texto_constante:
                self.mostrar_aviso("Valor requerido", "Introduce un valor numérico constante antes de aplicar el preprocesado.")
                return
            try:
                valor_constante = parsear_valor_constante(texto_constante)
            except Exception as err:
                self.mostrar_error("Valor constante no válido", err, "interpretar el valor constante")
                return

        try:
            df_train_limpio, df_test_limpio, resumen = tratar_valores_inexistentes(
                self.df_train, self.df_test, columnas_evaluar, estrategia, valor_constante
            )
        except Exception as err:
            self.mostrar_error("Error en Preprocesado", err, "aplicar el preprocesado")
            return

        self.df_train = df_train_limpio
        self.df_test = df_test_limpio
        self.limpiar_estado_modelo()

        self.mostrar_resultado_separacion()
        self.actualizar_reporte_nulos()
        self.mostrar_exito("Preprocesado aplicado", "El tratamiento de valores inexistentes se ha completado correctamente.")

    # -------------------------------------------------------------
    # Carga asíncrona de dataset (sin bloquear la ventana)
    # -------------------------------------------------------------
    def seleccionar_y_cargar_archivo(self):
        if self._cargando_archivo:
            return

        filtro_extensiones = (
            "Archivos Compatibles (*.csv *.xlsx *.xls *.sqlite *.db);;"
            "Archivos CSV (*.csv);;"
            "Archivos Excel (*.xlsx *.xls);;"
            "Bases de Datos SQLite (*.sqlite *.db)"
        )
        ruta_seleccionada, _ = QFileDialog.getOpenFileName(self, "Seleccionar Dataset", "", filtro_extensiones)
        if not ruta_seleccionada:
            return

        # Historia #115: cargar un dataset descarta el modelo actual; confirmar si tiene cambios sin guardar
        if not self._confirmar_descartar_modelo():
            return

        ruta_absoluta = os.path.abspath(ruta_seleccionada)
        self.establecer_modo_carga(True, f"Cargando «{Path(ruta_absoluta).name}»...")

        self._hilo_carga = HiloCargaDataset(ruta_absoluta, self)
        self._hilo_carga.carga_completada.connect(self._al_completar_carga_dataset)
        self._hilo_carga.carga_fallida.connect(self._al_fallar_carga_dataset)
        self._hilo_carga.finished.connect(self._al_finalizar_hilo_carga)
        self._hilo_carga.start()

    def _al_completar_carga_dataset(self, df_nuevo: pd.DataFrame, ruta_absoluta: str):
        # Historia #115: al cargar un dataset se vuelven a mostrar las secciones de trabajo con datos
        self._mostrar_secciones_datos(True)

        self.df_cargado = df_nuevo
        self.ruta_archivo_actual = ruta_absoluta
        self.input_ruta.setText(self.ruta_archivo_actual)

        self.mostrar_dataframe_en_tabla(self.df_cargado, self.tabla_datos)
        self.lbl_info_tabla.setText(
            f"Mostrando todas las filas con scroll virtual: {len(df_nuevo):,} filas × {df_nuevo.shape[1]} columnas".replace(",", ".")
        )
        self.actualizar_selectores_columnas(self.df_cargado)
        self.actualizar_reporte_nulos()
        self.establecer_modo_carga(False)
        self.mostrar_exito(
            "Dataset cargado",
            f"El archivo «{Path(ruta_absoluta).name}» se ha cargado correctamente "
            f"({len(df_nuevo):,} filas y {df_nuevo.shape[1]} columnas).".replace(",", ".")
        )

    def _al_fallar_carga_dataset(self, error: Exception):
        self.establecer_modo_carga(False)
        self.mostrar_error("Error al cargar el dataset", error, "cargar el archivo de datos")

    def _al_finalizar_hilo_carga(self):
        if self._hilo_carga is not None:
            self._hilo_carga.deleteLater()
            self._hilo_carga = None

    def mostrar_dataframe_en_tabla(
        self,
        df: pd.DataFrame,
        tabla: QTableView = None,
        etiquetas: list = None,
        resaltar_ultima_columna: bool = False,
    ):
        if tabla is None:
            tabla = self.tabla_datos
        modelo_tabla = ModeloTablaDataFrame(
            df=df,
            etiquetas=etiquetas,
            resaltar_ultima_columna=resaltar_ultima_columna,
            parent=tabla,
        )
        tabla.setModel(modelo_tabla)
        tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def actualizar_selectores_columnas(self, df: pd.DataFrame):
        self.combo_target.blockSignals(True)
        self.lista_features.clear()
        self.combo_target.clear()
        self.features_confirmadas = []
        self.target_confirmado = ""
        self.invalidar_separacion()

        if df is not None and not df.empty:
            columnas = [str(col) for col in df.columns]
            self.lista_features.addItems(columnas)
            self.combo_target.addItem("-- Seleccionar salida --")
            self.combo_target.addItems(columnas)
        self.combo_target.blockSignals(False)
        self.mostrar_seleccion_confirmada()
        self.actualizar_estado_flujo()

    def mostrar_seleccion_confirmada(self):
        if not self.features_confirmadas or not self.target_confirmado:
            self.lbl_seleccion_confirmada.setText("<b>Selección confirmada:</b> ninguna todavía.")
            self.tabla_seleccion.setModel(ModeloTablaDataFrame(parent=self.tabla_seleccion))
            return
        entradas = ", ".join(html.escape(c) for c in self.features_confirmadas)
        salida = html.escape(self.target_confirmado)
        self.lbl_seleccion_confirmada.setText(
            f"<b>Selección confirmada:</b><br>"
            f"<b>Entrada (features):</b> {entradas}<br>"
            f"<b>Salida (target):</b> {salida}"
        )
        self.mostrar_tabla_seleccion()

    def mostrar_tabla_seleccion(self):
        nombres = {str(c): c for c in self.df_cargado.columns}
        orden = self.features_confirmadas + [self.target_confirmado]
        df_sel = self.df_cargado[[nombres[c] for c in orden]]
        etiquetas = [f"{c}\n(entrada)" for c in self.features_confirmadas] + [f"{self.target_confirmado}\n(salida)"]
        self.mostrar_dataframe_en_tabla(
            df_sel,
            self.tabla_seleccion,
            etiquetas=etiquetas,
            resaltar_ultima_columna=True,
        )

    def al_cambiar_target(self, texto: str):
        for i in range(self.lista_features.count()):
            item = self.lista_features.item(i)
            es_target = bool(texto) and texto != "-- Seleccionar salida --" and item.text() == texto
            if es_target:
                item.setSelected(False)
            flags = item.flags()
            item.setFlags(flags & ~Qt.ItemIsEnabled if es_target else flags | Qt.ItemIsEnabled)

    def confirmar_seleccion_columnas(self):
        if self.df_cargado is None:
            self.mostrar_aviso("Dataset requerido", "Primero debes cargar un dataset en el Paso 1.")
            return

        features_seleccionadas = [item.text() for item in self.lista_features.selectedItems()]
        target_seleccionado = self.combo_target.currentText()
        if target_seleccionado == "-- Seleccionar salida --":
            target_seleccionado = ""

        es_valido, mensaje = validar_seleccion_columnas(
            self.df_cargado, features_seleccionadas, target_seleccionado
        )

        if es_valido:
            cambio = set(features_seleccionadas) != set(self.features_confirmadas) or target_seleccionado != self.target_confirmado
            self.features_confirmadas = features_seleccionadas
            self.target_confirmado = target_seleccionado
            self.mostrar_seleccion_confirmada()
            if cambio:
                self.invalidar_separacion()
            else:
                self.actualizar_estado_flujo()
            self.mostrar_exito("Selección confirmada", mensaje)
        else:
            self.mostrar_aviso("Error en la selección", mensaje)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaCargaDatos()
    ventana.show()
    sys.exit(app.exec())