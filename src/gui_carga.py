# src/procesador_datos_rodrigo_lucas_martin/gui_carga.py

import sys
import os
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLineEdit, QMessageBox, QHeaderView, QGroupBox, QLabel
)

from importacion import importar_datos
from separacion import separar_datos


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga y Procesamiento de Datasets")
        self.resize(950, 700)

        # Estado global para mantener los datos disponibles en los siguientes pasos
        self.df_cargado: pd.DataFrame = None
        self.ruta_archivo_actual: str = ""
        self.columnas_confirmadas: bool = False

        # Subconjuntos almacenados tras la separación (Historia #41)
        self.df_entrenamiento: pd.DataFrame = None
        self.df_test: pd.DataFrame = None

        # Estado de pasos posteriores (se descartan si se repite la separación)
        self.datos_preprocesados = None
        self.modelo = None

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)

        # =========================================================
        # 1. PANEL SUPERIOR: CARGA DE ARCHIVO
        # =========================================================
        layout_superior = QHBoxLayout()
        self.btn_cargar = QPushButton("Cargar Dataset")
        self.btn_cargar.clicked.connect(self.seleccionar_y_cargar_archivo)

        self.input_ruta = QLineEdit()
        self.input_ruta.setReadOnly(True)
        self.input_ruta.setPlaceholderText("Ningún archivo cargado...")

        # Botón para confirmar columnas (habilita el paso de separación)
        self.btn_confirmar_columnas = QPushButton("Confirmar Selección de Columnas")
        self.btn_confirmar_columnas.setEnabled(False)
        self.btn_confirmar_columnas.clicked.connect(self.confirmar_seleccion_columnas)

        layout_superior.addWidget(self.btn_cargar)
        layout_superior.addWidget(self.input_ruta)
        layout_superior.addWidget(self.btn_confirmar_columnas)
        layout_principal.addLayout(layout_superior)

        # =========================================================
        # 2. TABLA DE VISUALIZACIÓN
        # =========================================================
        self.tabla_datos = QTableWidget()
        layout_principal.addWidget(self.tabla_datos)

        # =========================================================
        # 3. PANEL DE SEPARACIÓN ENTRENAMIENTO / TEST (Historia #41)
        # =========================================================
        self.grupo_separacion = QGroupBox("Separación de datos (Entrenamiento y Test)")
        self.grupo_separacion.setEnabled(False)  # Solo disponible tras confirmar columnas
        layout_separacion = QVBoxLayout(self.grupo_separacion)

        layout_controles_sep = QHBoxLayout()

        # Porcentaje de test (por defecto 20 %)
        layout_controles_sep.addWidget(QLabel("Porcentaje Test (%):"))
        self.input_porcentaje_test = QLineEdit("20")
        self.input_porcentaje_test.setFixedWidth(60)
        self.input_porcentaje_test.textChanged.connect(self.actualizar_porcentaje_entrenamiento)
        layout_controles_sep.addWidget(self.input_porcentaje_test)

        # Porcentaje de entrenamiento calculado automáticamente como complementario (80 %)
        self.lbl_porcentaje_entrenamiento = QLabel("Porcentaje Entrenamiento: 80 %")
        layout_controles_sep.addWidget(self.lbl_porcentaje_entrenamiento)

        layout_controles_sep.addSpacing(20)

        # Semilla (por defecto 42)
        layout_controles_sep.addWidget(QLabel("Semilla:"))
        self.input_semilla = QLineEdit("42")
        self.input_semilla.setFixedWidth(80)
        layout_controles_sep.addWidget(self.input_semilla)

        layout_controles_sep.addSpacing(20)

        # Botón "Separar datos"
        self.btn_separar = QPushButton("Separar datos")
        self.btn_separar.clicked.connect(self.ejecutar_separacion_datos)
        layout_controles_sep.addWidget(self.btn_separar)
        layout_controles_sep.addStretch()

        layout_separacion.addLayout(layout_controles_sep)

        # Etiquetas de visualización de resultados de la separación
        self.lbl_info_filas = QLabel("Filas -> Entrenamiento: - | Test: -")
        self.lbl_confirmacion_sep = QLabel("")
        layout_separacion.addWidget(self.lbl_info_filas)
        layout_separacion.addWidget(self.lbl_confirmacion_sep)

        layout_principal.addWidget(self.grupo_separacion)

    # =========================================================
    # LÓGICA DE INTERFAZ Y FLUJO DE TRABAJO
    # =========================================================

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

            # Al cargar un nuevo dataset, se reinicia la confirmación de columnas y la separación
            self.columnas_confirmadas = False
            self.reiniciar_estado_separacion()
            self.grupo_separacion.setEnabled(False)
            self.btn_confirmar_columnas.setEnabled(True)

        except Exception as error:
            QMessageBox.critical(
                self,
                "Error al cargar archivo",
                f"No se pudo cargar el dataset:\n{str(error)}"
            )

    def confirmar_seleccion_columnas(self):
        """
        Se ejecuta cuando el usuario confirma las columnas de entrada y salida.
        Si se cambia la selección de columnas, reinicia la separación previa.
        """
        if self.df_cargado is None:
            return

        self.columnas_confirmadas = True
        self.reiniciar_estado_separacion()
        self.grupo_separacion.setEnabled(True)

    def reiniciar_estado_separacion(self):
        """Descarta la separación actual y los pasos posteriores (preprocesado y modelo)."""
        self.df_entrenamiento = None
        self.df_test = None
        self.datos_preprocesados = None
        self.modelo = None
        self.lbl_info_filas.setText("Filas -> Entrenamiento: - | Test: -")
        self.lbl_confirmacion_sep.setText("")

    def actualizar_porcentaje_entrenamiento(self, texto: str):
        """Calcula automáticamente el porcentaje de entrenamiento como el complementario."""
        try:
            valor_test = float(texto)
            if 1 <= valor_test <= 99:
                valor_train = 100 - valor_test
                # Mostrar sin decimales si es entero
                texto_train = f"{int(valor_train)}" if valor_train.is_integer() else f"{valor_train:.2f}"
                self.lbl_porcentaje_entrenamiento.setText(f"Porcentaje Entrenamiento: {texto_train} %")
            else:
                self.lbl_porcentaje_entrenamiento.setText("Porcentaje Entrenamiento: - %")
        except ValueError:
            self.lbl_porcentaje_entrenamiento.setText("Porcentaje Entrenamiento: - %")

    def ejecutar_separacion_datos(self):
        """Valida las entradas de la GUI, llama a separar_datos y muestra el resultado."""
        if not self.columnas_confirmadas or self.df_cargado is None:
            QMessageBox.warning(
                self,
                "Flujo de trabajo",
                "Debe confirmar la selección de columnas antes de separar los datos."
            )
            return

        # 1. Validar que el porcentaje sea un número entre 1 y 99
        texto_porcentaje = self.input_porcentaje_test.text().strip()
        try:
            porcentaje_test = float(texto_porcentaje)
        except ValueError:
            QMessageBox.critical(
                self,
                "Error en el porcentaje",
                "El porcentaje introducido debe ser un número entre 1 y 99."
            )
            return

        # 2. Validar que la semilla sea un número entero
        texto_semilla = self.input_semilla.text().strip()
        try:
            semilla = int(texto_semilla)
        except ValueError:
            QMessageBox.critical(
                self,
                "Error en la semilla",
                "La semilla introducida debe ser un número entero."
            )
            return

        # 3. Llamar a la función lógica independiente
        try:
            df_train, df_test = separar_datos(
                self.df_cargado,
                porcentaje_test=porcentaje_test,
                semilla=semilla
            )

            # Al repetir la separación, se descartan el preprocesado y el modelo anteriores
            self.datos_preprocesados = None
            self.modelo = None

            # Guardar los conjuntos en la aplicación para los pasos siguientes
            self.df_entrenamiento = df_train
            self.df_test = df_test

            # Mostrar el número de filas de cada conjunto y el mensaje de confirmación
            filas_train = len(self.df_entrenamiento)
            filas_test = len(self.df_test)
            self.lbl_info_filas.setText(
                f"Filas -> Entrenamiento: {filas_train} | Test: {filas_test}"
            )
            self.lbl_confirmacion_sep.setText(
                "✔ La división en entrenamiento y test se ha realizado correctamente."
            )

            QMessageBox.information(
                self,
                "Separación completada",
                f"La división se ha realizado correctamente.\n\n"
                f"• Filas de entrenamiento: {filas_train}\n"
                f"• Filas de test: {filas_test}"
            )

        except ValueError as error:
            QMessageBox.critical(
                self,
                "Error al separar los datos",
                str(error)
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