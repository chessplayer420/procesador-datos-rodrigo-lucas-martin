import sys
import os
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLineEdit, QMessageBox, QHeaderView, QLabel, QListWidget,
    QComboBox, QAbstractItemView
)

from preprocesado import validar_seleccion_columnas
from importacion import importar_datos


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga y Visualización de Datasets")
        self.resize(900, 600)

        # Estado global para mantener los datos disponibles en los siguientes pasos
        self.df_cargado: pd.DataFrame = None
        self.ruta_archivo_actual: str = ""
        self.features_confirmadas: list = []
        self.target_confirmado: str = ""

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)

        # Botón y campo de texto para mostrar la ruta absoluta
        layout_superior = QHBoxLayout()
        self.btn_cargar = QPushButton("Cargar Dataset")
        self.btn_cargar.clicked.connect(self.seleccionar_y_cargar_archivo)

        self.input_ruta = QLineEdit()
        self.input_ruta.setReadOnly(True)
        self.input_ruta.setPlaceholderText("Ningún archivo cargado...")

        layout_superior.addWidget(self.btn_cargar)
        layout_superior.addWidget(self.input_ruta)
        layout_principal.addLayout(layout_superior)

        # Tabla con scroll horizontal y vertical
        self.tabla_datos = QTableWidget()
        layout_principal.addWidget(self.tabla_datos)

        # Seleccion de columnas de entrada (features) y salida (target)
        layout_seleccion = QHBoxLayout()

        # Selector de Features (Entrada)
        layout_features = QVBoxLayout()
        self.lbl_features = QLabel("Columnas de entrada (Features):")
        self.lista_features = QListWidget()
        # Permitir selección múltiple
        self.lista_features.setSelectionMode(QAbstractItemView.MultiSelection)
        layout_features.addWidget(self.lbl_features)
        layout_features.addWidget(self.lista_features)

        # Selector de Target (Salida)
        layout_target = QVBoxLayout()
        self.lbl_target = QLabel("Columna de salida (Target):")
        self.combo_target = QComboBox()
        layout_target.addWidget(self.lbl_target)
        layout_target.addWidget(self.combo_target)
        layout_target.addStretch()

        layout_seleccion.addLayout(layout_features)
        layout_seleccion.addLayout(layout_target)
        layout_principal.addLayout(layout_seleccion)

        # Botón Confirmar Selección
        self.btn_confirmar = QPushButton("Confirmar selección")
        self.btn_confirmar.clicked.connect(self.confirmar_seleccion_columnas)
        layout_principal.addWidget(self.btn_confirmar)

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
            
            # Delegación al módulo de importación
            df_nuevo = importar_datos(ruta_absoluta)

            self.df_cargado = df_nuevo
            self.ruta_archivo_actual = ruta_absoluta
            self.input_ruta.setText(self.ruta_archivo_actual)

            self.mostrar_dataframe_en_tabla(self.df_cargado)

            self.actualizar_selectores_columnas(self.df_cargado)

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
        """Actualiza los selectores con las columnas del dataset y resetea selecciones previas."""
        self.lista_features.clear()
        self.combo_target.clear()
        self.features_confirmadas = []
        self.target_confirmado = ""

        if df is None or df.empty:
            return

        columnas = [str(col) for col in df.columns]
        
        # Poblar lista de features y combo de target
        self.lista_features.addItems(columnas)
        self.combo_target.addItem("-- Seleccionar salida --")
        self.combo_target.addItems(columnas)

    def confirmar_seleccion_columnas(self):
        """Lee los elementos seleccionados en la GUI y valida con la función independiente."""
        if self.df_cargado is None:
            QMessageBox.warning(self, "Error", "Primero debes cargar un dataset.")
            return

        # Obtener las columnas seleccionadas en la GUI
        features_seleccionadas = [item.text() for item in self.lista_features.selectedItems()]
        
        target_seleccionado = self.combo_target.currentText()
        if target_seleccionado == "-- Seleccionar salida --":
            target_seleccionado = ""

        # Delegar la validación a la lógica pura
        es_valido, mensaje = validar_seleccion_columnas(
            self.df_cargado,
            features_seleccionadas,
            target_seleccionado
        )

        if es_valido:
            self.features_confirmadas = features_seleccionadas
            self.target_confirmado = target_seleccionado
            QMessageBox.information(self, "Selección Confirmada", mensaje)
        else:
            QMessageBox.warning(self, "Error en Selección", mensaje)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaCargaDatos()
    ventana.show()
    sys.exit(app.exec())