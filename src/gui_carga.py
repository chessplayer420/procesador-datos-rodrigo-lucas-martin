import html
import sys
import os
import pandas as pd
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLineEdit, QMessageBox, QHeaderView, QLabel, QListWidget,
    QComboBox, QAbstractItemView, QGroupBox, QSizePolicy, QScrollArea
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importaciones limpias con el prefijo src. para ejecuciones por módulo
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
    from src.modelo import entrenar_modelo_regresion
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
    from modelo import entrenar_modelo_regresion


# Estilo Global Moderno (CSS / QSS)
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
QLineEdit, QComboBox, QListWidget {
    background-color: #FFFFFF;
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 5px 8px;
}
QLineEdit:focus, QComboBox:focus, QListWidget:focus {
    border-color: #2563EB;
}
QTableWidget {
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
"""


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema Integrado de Análisis y Regresión Lineal")
        self.resize(1100, 850)
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

        # Scroll principal para evitar solapamientos
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #F3F4F6; }")
        self.setCentralWidget(scroll)

        contenedor_scroll = QWidget()
        scroll.setWidget(contenedor_scroll)
        layout_principal = QVBoxLayout(contenedor_scroll)
        layout_principal.setSpacing(16)
        layout_principal.setContentsMargins(16, 16, 16, 16)

        # -------------------------------------------------------------
        # 1. Carga de archivo
        # -------------------------------------------------------------
        group_carga = QGroupBox("1. Carga del Dataset")
        layout_carga = QVBoxLayout()
        
        layout_superior = QHBoxLayout()
        self.btn_cargar = QPushButton("📁 Cargar Dataset")
        self.btn_cargar.setStyleSheet("QPushButton { background-color: #2563EB; color: white; border: none; } QPushButton:hover { background-color: #1D4ED8; }")
        self.btn_cargar.clicked.connect(self.seleccionar_y_cargar_archivo)

        self.input_ruta = QLineEdit()
        self.input_ruta.setReadOnly(True)
        self.input_ruta.setPlaceholderText("Ningún archivo cargado...")

        layout_superior.addWidget(self.btn_cargar)
        layout_superior.addWidget(self.input_ruta)
        layout_carga.addLayout(layout_superior)

        self.tabla_datos = QTableWidget()
        self.tabla_datos.setFixedHeight(160)
        layout_carga.addWidget(self.tabla_datos)
        
        group_carga.setLayout(layout_carga)
        layout_principal.addWidget(group_carga)

        # -------------------------------------------------------------
        # 2. Selección de columnas
        # -------------------------------------------------------------
        group_seleccion_col = QGroupBox("2. Selección de Variables")
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
        self.btn_confirmar.clicked.connect(self.confirmar_seleccion_columnas)
        layout_sel_box.addWidget(self.btn_confirmar)

        self.lbl_seleccion_confirmada = QLabel()
        self.lbl_seleccion_confirmada.setWordWrap(True)
        layout_sel_box.addWidget(self.lbl_seleccion_confirmada)

        self.tabla_seleccion = QTableWidget()
        self.tabla_seleccion.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla_seleccion.setFixedHeight(120)
        layout_sel_box.addWidget(self.tabla_seleccion)
        self.mostrar_seleccion_confirmada()

        group_seleccion_col.setLayout(layout_sel_box)
        layout_principal.addWidget(group_seleccion_col)

        # -------------------------------------------------------------
        # 3. Separación en Entrenamiento y Test
        # -------------------------------------------------------------
        group_separacion = QGroupBox("3. División de Datasets (Train / Test)")
        layout_sep = QVBoxLayout()

        layout_params = QHBoxLayout()
        layout_params.addWidget(QLabel("% Test:"))
        self.input_porcentaje_test = QLineEdit(str(PORCENTAJE_TEST_POR_DEFECTO))
        self.input_porcentaje_test.setFixedWidth(60)
        self.input_porcentaje_test.textChanged.connect(self.actualizar_porcentaje_train)
        layout_params.addWidget(self.input_porcentaje_test)

        self.lbl_porcentaje_train = QLabel()
        self.lbl_porcentaje_train.setStyleSheet("font-weight: 600;")
        layout_params.addWidget(self.lbl_porcentaje_train)

        layout_params.addWidget(QLabel("Semilla:"))
        self.input_semilla = QLineEdit(str(SEMILLA_POR_DEFECTO))
        self.input_semilla.setFixedWidth(60)
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

        group_separacion.setLayout(layout_sep)
        layout_principal.addWidget(group_separacion)
        self.actualizar_porcentaje_train(self.input_porcentaje_test.text())

        # -------------------------------------------------------------
        # 4. Tratamiento de Valores Inexistentes (NaN)
        # -------------------------------------------------------------
        self.group_preprocesado = QGroupBox("4. Tratamiento de Valores Nulos")
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
        # 5. Modelo de Regresión Lineal
        # -------------------------------------------------------------
        self.group_modelo = QGroupBox("5. Creación y Evaluación del Modelo de Regresión Lineal")
        layout_mod = QVBoxLayout()

        self.btn_crear_modelo = QPushButton("🚀 Crear Modelo")
        self.btn_crear_modelo.setFixedHeight(42)
        self.btn_crear_modelo.setStyleSheet("""
            QPushButton { font-size: 15px; font-weight: bold; background-color: #059669; color: white; border: none; border-radius: 6px; }
            QPushButton:hover { background-color: #047857; }
            QPushButton:disabled { background-color: #E5E7EB; color: #9CA3AF; }
        """)
        self.btn_crear_modelo.clicked.connect(self.ejecutar_creacion_modelo)
        layout_mod.addWidget(self.btn_crear_modelo)

        self.lbl_formula = QLabel("<b>Fórmula del modelo:</b> <i>(Pendiente de creación)</i>")
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setStyleSheet("font-size: 13px; padding: 10px; background-color: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 6px;")
        layout_mod.addWidget(self.lbl_formula)

        self.tabla_metricas = QTableWidget(2, 2)
        self.tabla_metricas.setHorizontalHeaderLabels(["Entrenamiento", "Test"])
        self.tabla_metricas.setVerticalHeaderLabels(["R² (Coef. Determinación)", "Error Cuadrático Medio (ECM)"])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.setFixedHeight(90)
        layout_mod.addWidget(self.tabla_metricas)

        self.canvas_modelo = FigureCanvas(Figure(figsize=(7, 4.5)))
        self.canvas_modelo.setMinimumHeight(400)
        layout_mod.addWidget(self.canvas_modelo)

        self.group_modelo.setLayout(layout_mod)
        layout_principal.addWidget(self.group_modelo)

        # Estado inicial
        self.invalidar_separacion()

    # -------------------------------------------------------------
    # Métodos de la canalización de datos y modelo
    # -------------------------------------------------------------
    def actualizar_estado_flujo(self):
        hay_seleccion = bool(self.features_confirmadas and self.target_confirmado)
        hay_separacion = self.df_train is not None and self.df_test is not None
        
        self.btn_separar.setEnabled(self.df_cargado is not None and hay_seleccion)
        self.group_preprocesado.setEnabled(hay_separacion)
        self.btn_crear_modelo.setEnabled(self.listo_para_crear_modelo())

    def invalidar_separacion(self):
        self.df_train = None
        self.df_test = None
        self.resultado_modelo = None
        self.lbl_resultado_separacion.setText("Pendiente: confirma la selección de columnas y pulsa «Separar datos».")
        self.actualizar_estado_flujo()
        self.actualizar_reporte_nulos()

    def listo_para_crear_modelo(self) -> bool:
        columnas = self.columnas_evaluadas()
        if self.df_train is None or self.df_test is None or not columnas:
            return False
        return not hay_valores_inexistentes(self.df_train, self.df_test, columnas)

    def ejecutar_creacion_modelo(self):
        if not self.listo_para_crear_modelo():
            QMessageBox.warning(self, "Aviso", "No se cumplen las condiciones para crear el modelo.")
            return

        try:
            X_train = self.df_train[self.features_confirmadas]
            y_train = self.df_train[self.target_confirmado]
            X_test = self.df_test[self.features_confirmadas]
            y_test = self.df_test[self.target_confirmado]

            res = entrenar_modelo_regresion(X_train, y_train, X_test, y_test)
            self.resultado_modelo = res

            self.lbl_formula.setText(f"<b>Fórmula del modelo:</b> {res['formula']}")

            m = res["metricas"]
            item_r2_tr = QTableWidgetItem(f"{m['r2_train']:.4f}")
            item_r2_te = QTableWidgetItem(f"{m['r2_test']:.4f}")
            item_ecm_tr = QTableWidgetItem(f"{m['ecm_train']:.4f}")
            item_ecm_te = QTableWidgetItem(f"{m['ecm_test']:.4f}")

            for item in [item_r2_tr, item_r2_te, item_ecm_tr, item_ecm_te]:
                item.setTextAlignment(Qt.AlignCenter)

            self.tabla_metricas.setItem(0, 0, item_r2_tr)
            self.tabla_metricas.setItem(0, 1, item_r2_te)
            self.tabla_metricas.setItem(1, 0, item_ecm_tr)
            self.tabla_metricas.setItem(1, 1, item_ecm_te)

            self.dibujar_grafico_modelo(X_train, y_train, X_test, y_test, res["y_pred_train"], res["y_pred_test"], res["modelo"])
            QMessageBox.information(self, "Éxito", "El modelo de regresión lineal se ha creado y evaluado correctamente.")

        except Exception as err:
            QMessageBox.critical(self, "Error al crear el modelo", f"Ocurrió un error inesperado:\n{str(err)}")

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
            QMessageBox.warning(self, "Aviso", "Primero debes cargar un dataset y confirmar la selección de columnas.")
            return

        try:
            pct_test, semilla = parsear_parametros_separacion(
                self.input_porcentaje_test.text(), self.input_semilla.text()
            )
            df_train, df_test = separar_train_test(self.df_cargado, pct_test, semilla)
        except ValueError as err:
            QMessageBox.critical(self, "Error en la separación", str(err))
            return

        self.df_train = df_train
        self.df_test = df_test
        self.resultado_modelo = None
        self.porcentaje_test_usado = pct_test
        self.semilla_usada = semilla

        self.mostrar_resultado_separacion()
        self.actualizar_estado_flujo()
        self.actualizar_reporte_nulos()

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
            self.btn_aplicar_preprocesado.setEnabled(True)

        self.btn_crear_modelo.setEnabled(self.listo_para_crear_modelo())

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
                QMessageBox.warning(self, "Error de Entrada", "Debes introducir un valor constante.")
                return
            try:
                valor_constante = parsear_valor_constante(texto_constante)
            except ValueError as err:
                QMessageBox.critical(self, "Error de Tipo", str(err))
                return

        try:
            df_train_limpio, df_test_limpio, resumen = tratar_valores_inexistentes(
                self.df_train, self.df_test, columnas_evaluar, estrategia, valor_constante
            )
        except Exception as err:
            QMessageBox.critical(self, "Error en Preprocesado", str(err))
            return

        self.df_train = df_train_limpio
        self.df_test = df_test_limpio
        self.resultado_modelo = None

        QMessageBox.information(self, "Preprocesado Exitoso", "Preprocesado aplicado correctamente.")
        self.mostrar_resultado_separacion()
        self.actualizar_reporte_nulos()

    def seleccionar_y_cargar_archivo(self):
        filtro_extensiones = (
            "Archivos Compatibles (*.csv *.xlsx *.xls *.sqlite *.db);;"
            "Archivos CSV (*.csv);;"
            "Archivos Excel (*.xlsx *.xls);;"
            "Bases de Datos SQLite (*.sqlite *.db)"
        )
        ruta_seleccionada, _ = QFileDialog.getOpenFileName(self, "Seleccionar Dataset", "", filtro_extensiones)

        if not ruta_seleccionada:
            return

        try:
            ruta_absoluta = os.path.abspath(ruta_seleccionada)
            df_nuevo = importar_datos(ruta_absoluta)
            df_nuevo.columns = [str(c) for c in df_nuevo.columns]

            self.df_cargado = df_nuevo
            self.ruta_archivo_actual = ruta_absoluta
            self.input_ruta.setText(self.ruta_archivo_actual)

            self.mostrar_dataframe_en_tabla(self.df_cargado)
            self.actualizar_selectores_columnas(self.df_cargado)
            self.actualizar_reporte_nulos()

        except Exception as error:
            QMessageBox.critical(self, "Error al cargar archivo", f"No se pudo cargar el dataset:\n{str(error)}")

    def mostrar_dataframe_en_tabla(self, df: pd.DataFrame, tabla: QTableWidget = None, etiquetas: list = None):
        if tabla is None:
            tabla = self.tabla_datos
        tabla.clear()
        filas, columnas = df.shape
        tabla.setRowCount(filas)
        tabla.setColumnCount(columnas)
        tabla.setHorizontalHeaderLabels(etiquetas or [str(col) for col in df.columns])

        for r in range(filas):
            for c in range(columnas):
                val = df.iat[r, c]
                texto = "" if pd.isna(val) else str(val)
                tabla.setItem(r, c, QTableWidgetItem(texto))

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

    def mostrar_seleccion_confirmada(self):
        if not self.features_confirmadas or not self.target_confirmado:
            self.lbl_seleccion_confirmada.setText("<b>Selección confirmada:</b> ninguna todavía.")
            self.tabla_seleccion.clear()
            self.tabla_seleccion.setRowCount(0)
            self.tabla_seleccion.setColumnCount(0)
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
        self.mostrar_dataframe_en_tabla(df_sel, self.tabla_seleccion, etiquetas)

        col_target = len(orden) - 1
        fondo = QBrush(QColor("#DBEAFE"))
        texto = QBrush(QColor("#1E40AF"))
        for r in range(self.tabla_seleccion.rowCount()):
            item = self.tabla_seleccion.item(r, col_target)
            if item:
                item.setBackground(fondo)
                item.setForeground(texto)

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
            QMessageBox.warning(self, "Error", "Primero debes cargar un dataset.")
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
            QMessageBox.information(self, "Selección Confirmada", mensaje)
        else:
            QMessageBox.warning(self, "Error en Selección", mensaje)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaCargaDatos()
    ventana.show()
    sys.exit(app.exec())