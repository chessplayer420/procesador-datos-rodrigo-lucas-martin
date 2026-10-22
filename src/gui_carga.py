import sys
import os
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLineEdit, QMessageBox, QHeaderView
)

from src.importacion import cargar_dataset


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga y Visualización de Datasets")
        self.resize(900, 600)

        # Estado global para mantener los datos disponibles en los siguientes pasos
        self.df_cargado: pd.DataFrame = None
        self.ruta_archivo_actual: str = ""

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
            df_nuevo = cargar_dataset(ruta_absoluta)

            self.df_cargado = df_nuevo
            self.ruta_archivo_actual = ruta_absoluta
            self.input_ruta.setText(self.ruta_archivo_actual)

            self.mostrar_dataframe_en_tabla(self.df_cargado)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaCargaDatos()
    ventana.show()
    sys.exit(app.exec())