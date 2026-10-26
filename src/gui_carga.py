import sys
import os
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLineEdit, QMessageBox, QHeaderView, QLabel, QListWidget,
    QComboBox, QAbstractItemView, QGroupBox
)

from preprocesado import (
    validar_seleccion_columnas,
    detectar_valores_inexistentes,
    tratar_valores_inexistentes
)
from importacion import importar_datos


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga y Visualización de Datasets")
        self.resize(900, 750)

        # Estado global para mantener los datos disponibles
        self.df_cargado: pd.DataFrame = None
        self.df_train: pd.DataFrame = None
        self.df_test: pd.DataFrame = None
        self.ruta_archivo_actual: str = ""
        self.features_confirmadas: list = []
        self.target_confirmado: str = ""

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)

        # 1. Carga de archivo
        layout_superior = QHBoxLayout()
        self.btn_cargar = QPushButton("Cargar Dataset")
        self.btn_cargar.clicked.connect(self.seleccionar_y_cargar_archivo)

        self.input_ruta = QLineEdit()
        self.input_ruta.setReadOnly(True)
        self.input_ruta.setPlaceholderText("Ningún archivo cargado...")

        layout_superior.addWidget(self.btn_cargar)
        layout_superior.addWidget(self.input_ruta)
        layout_principal.addLayout(layout_superior)

        # 2. Tabla de visualización
        self.tabla_datos = QTableWidget()
        layout_principal.addWidget(self.tabla_datos)

        # 3. Selección de columnas (Features / Target)
        layout_seleccion = QHBoxLayout()

        layout_features = QVBoxLayout()
        self.lbl_features = QLabel("Columnas de entrada (Features):")
        self.lista_features = QListWidget()
        self.lista_features.setSelectionMode(QAbstractItemView.MultiSelection)
        layout_features.addWidget(self.lbl_features)
        layout_features.addWidget(self.lista_features)

        layout_target = QVBoxLayout()
        self.lbl_target = QLabel("Columna de salida (Target):")
        self.combo_target = QComboBox()
        layout_target.addWidget(self.lbl_target)
        layout_target.addWidget(self.combo_target)
        layout_target.addStretch()

        layout_seleccion.addLayout(layout_features)
        layout_seleccion.addLayout(layout_target)
        layout_principal.addLayout(layout_seleccion)

        self.btn_confirmar = QPushButton("Confirmar selección")
        self.btn_confirmar.clicked.connect(self.confirmar_seleccion_columnas)
        layout_principal.addWidget(self.btn_confirmar)

        # -------------------------------------------------------------
        # TAREA 2: Conexión de Lógica y Control de Errores de Preprocesado
        # -------------------------------------------------------------
        group_preprocesado = QGroupBox("Tratamiento de Valores Inexistentes (NaN)")
        layout_prep = QVBoxLayout()

        # Detección e información de nulos
        self.lbl_reporte_nulos = QLabel("Estado: Pendiente de separación en Train/Test")
        self.lbl_reporte_nulos.setWordWrap(True)
        layout_prep.addWidget(self.lbl_reporte_nulos)

        # Opciones de tratamiento
        layout_estrategia = QHBoxLayout()
        lbl_est = QLabel("Estrategia:")
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

        layout_estrategia.addWidget(lbl_est)
        layout_estrategia.addWidget(self.combo_estrategia)
        layout_estrategia.addWidget(self.input_constante)
        layout_prep.addLayout(layout_estrategia)

        # Botón aplicar preprocesado (conectado a ejecutar_preprocesado)
        self.btn_aplicar_preprocesado = QPushButton("Aplicar preprocesado")
        self.btn_aplicar_preprocesado.setEnabled(False)
        self.btn_aplicar_preprocesado.clicked.connect(self.ejecutar_preprocesado)
        layout_prep.addWidget(self.btn_aplicar_preprocesado)

        group_preprocesado.setLayout(layout_prep)
        layout_principal.addWidget(group_preprocesado)

    def al_cambiar_estrategia(self, index: int):
        """Habilita el campo constante solo cuando se elige la opción 3 (Valor constante)."""
        self.input_constante.setEnabled(index == 3)

    def actualizar_reporte_nulos(self):
        """Muestra el reporte de nulos por columna en Train y Test."""
        if self.df_train is None or self.df_test is None:
            self.lbl_reporte_nulos.setText("Estado: Pendiente de separación en Train/Test.")
            self.btn_aplicar_preprocesado.setEnabled(False)
            return

        columnas_evaluar = self.features_confirmadas + ([self.target_confirmado] if self.target_confirmado else [])
        if not columnas_evaluar:
            self.lbl_reporte_nulos.setText("Selecciona y confirma las columnas primero.")
            self.btn_aplicar_preprocesado.setEnabled(False)
            return

        reporte_train, reporte_test = detectar_valores_inexistentes(
            self.df_train, self.df_test, columnas_evaluar
        )

        total_nulos_train = sum(reporte_train.values())
        total_nulos_test = sum(reporte_test.values())

        if total_nulos_train == 0 and total_nulos_test == 0:
            self.lbl_reporte_nulos.setText("No hay valores inexistentes en las columnas seleccionadas. Puedes continuar sin preprocesar.")
            self.btn_aplicar_preprocesado.setEnabled(False)
        else:
            txt = f"<b>Valores inexistentes detectados:</b><br>"
            txt += f"• <b>Entrenamiento (Train):</b> {reporte_train}<br>"
            txt += f"• <b>Prueba (Test):</b> {reporte_test}"
            self.lbl_reporte_nulos.setText(txt)
            self.btn_aplicar_preprocesado.setEnabled(True)

    def ejecutar_preprocesado(self):
        """Ejecuta la estrategia de tratamiento de nulos seleccionada."""
        if self.df_train is None or self.df_test is None:
            QMessageBox.warning(self, "Aviso", "No existen los conjuntos de Entrenamiento y Test.")
            return

        columnas_evaluar = self.features_confirmadas + ([self.target_confirmado] if self.target_confirmado else [])
        if not columnas_evaluar:
            QMessageBox.warning(self, "Aviso", "Debes confirmar las columnas a evaluar.")
            return

        # Mapeo del selector a nombres de estrategia
        estrategias_map = {
            0: "eliminar",
            1: "media",
            2: "mediana",
            3: "constante"
        }
        idx = self.combo_estrategia.currentIndex()
        estrategia = estrategias_map.get(idx, "eliminar")

        valor_constante = None
        if estrategia == "constante":
            texto_constante = self.input_constante.text().strip()
            if not texto_constante:
                QMessageBox.warning(self, "Error de Entrada", "Debes introducir un valor constante.")
                return
            try:
                valor_constante = float(texto_constante)
            except ValueError:
                QMessageBox.critical(
                    self,
                    "Error de Tipo",
                    "El valor constante introducido debe ser un número válido."
                )
                return

        filas_train_antes = len(self.df_train)
        filas_test_antes = len(self.df_test)

        try:
            df_train_limpio, df_test_limpio = tratar_valores_inexistentes(
                self.df_train,
                self.df_test,
                columnas_evaluar,
                estrategia,
                valor_constante
            )

            # Comprobar que no se hayan vaciado los conjuntos al eliminar filas
            if df_train_limpio.empty or df_test_limpio.empty:
                QMessageBox.warning(
                    self,
                    "Operación Cancelada",
                    "La eliminación de filas dejaría vacío el conjunto de Entrenamiento o Test. Cambia de estrategia."
                )
                return

            # Actualizar estado interno
            self.df_train = df_train_limpio
            self.df_test = df_test_limpio

            # Generar mensaje de éxito
            if estrategia == "eliminar":
                diff_train = filas_train_antes - len(self.df_train)
                diff_test = filas_test_antes - len(self.df_test)
                msg = f"Se han eliminado {diff_train} filas en Train y {diff_test} filas en Test."
            else:
                msg = f"Valores inexistentes rellenados correctamente usando la estrategia '{estrategia}'."

            QMessageBox.information(self, "Preprocesado Exitoso", msg)
            self.actualizar_reporte_nulos()

        except Exception as err:
            QMessageBox.critical(self, "Error en Preprocesado", f"Ocurrió un error al procesar:\n{str(err)}")

    def seleccionar_y_cargar_archivo(self):
        filtro_extensiones = (
            "Archivos Compatibles (*.csv *.xlsx *.xls *.sqlite *.db);;"
            "Archivos CSV (*.csv);;"
            "Archivos Excel (*.xlsx *.xls);;"
            "Bases de Datos SQLite (*.sqlite *.db)"
        )

        ruta_seleccionada, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Dataset", "", filtro_extensiones
        )

        if not ruta_seleccionada:
            return

        try:
            ruta_absoluta = os.path.abspath(ruta_seleccionada)
            df_nuevo = importar_datos(ruta_absoluta)

            self.df_cargado = df_nuevo
            self.ruta_archivo_actual = ruta_absoluta
            self.input_ruta.setText(self.ruta_archivo_actual)

            self.mostrar_dataframe_en_tabla(self.df_cargado)
            self.actualizar_selectores_columnas(self.df_cargado)
            self.actualizar_reporte_nulos()

        except Exception as error:
            QMessageBox.critical(
                self,
                "Error al cargar archivo",
                f"No se pudo cargar el dataset:\n{str(error)}"
            )

    def mostrar_dataframe_en_tabla(self, df: pd.DataFrame):
        self.tabla_datos.clear()
        filas, columnas = df.shape
        self.tabla_datos.setRowCount(filas)
        self.tabla_datos.setColumnCount(columnas)
        self.tabla_datos.setHorizontalHeaderLabels([str(col) for col in df.columns])

        for r in range(filas):
            for c in range(columnas):
                val = df.iat[r, c]
                texto = "" if pd.isna(val) else str(val)
                self.tabla_datos.setItem(r, c, QTableWidgetItem(texto))

        self.tabla_datos.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def actualizar_selectores_columnas(self, df: pd.DataFrame):
        self.lista_features.clear()
        self.combo_target.clear()
        self.features_confirmadas = []
        self.target_confirmado = ""

        if df is None or df.empty:
            return

        columnas = [str(col) for col in df.columns]
        self.lista_features.addItems(columnas)
        self.combo_target.addItem("-- Seleccionar salida --")
        self.combo_target.addItems(columnas)

    def confirmar_seleccion_columnas(self):
        if self.df_cargado is None:
            QMessageBox.warning(self, "Error", "Primero debes cargar un dataset.")
            return

        features_seleccionadas = [item.text() for item in self.lista_features.selectedItems()]
        target_seleccionado = self.combo_target.currentText()
        if target_seleccionado == "-- Seleccionar salida --":
            target_seleccionado = ""

        es_valido, mensaje = validar_seleccion_columnas(
            self.df_cargado,
            features_seleccionadas,
            target_seleccionado
        )

        if es_valido:
            self.features_confirmadas = features_seleccionadas
            self.target_confirmado = target_seleccionado
            QMessageBox.information(self, "Selección Confirmada", mensaje)
            self.actualizar_reporte_nulos()
        else:
            QMessageBox.warning(self, "Error en Selección", mensaje)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaCargaDatos()
    ventana.show()
    sys.exit(app.exec())