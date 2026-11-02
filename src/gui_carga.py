import html
import sys
import os
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QLineEdit, QMessageBox, QHeaderView, QLabel, QListWidget,
    QComboBox, QAbstractItemView, QGroupBox
)

try:
    from procesador_datos_rodrigo_lucas_martin.preprocesado import (
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
    from procesador_datos_rodrigo_lucas_martin.importacion import importar_datos
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


class VentanaCargaDatos(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Carga y Visualización de Datasets")
        self.resize(900, 950)

        # Estado global para mantener los datos disponibles
        self.df_cargado: pd.DataFrame = None
        self.df_train: pd.DataFrame = None
        self.df_test: pd.DataFrame = None
        self.ruta_archivo_actual: str = ""
        self.features_confirmadas: list = []
        self.target_confirmado: str = ""
        self.porcentaje_test_usado: float = PORCENTAJE_TEST_POR_DEFECTO
        self.semilla_usada: int = SEMILLA_POR_DEFECTO
        self.modelo = None  # Se descartará al repetir la separación (paso de creación del modelo)

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
        self.lbl_features = QLabel("Columnas de entrada (Features) - selección múltiple:")
        self.lista_features = QListWidget()
        self.lista_features.setSelectionMode(QAbstractItemView.MultiSelection)
        layout_features.addWidget(self.lbl_features)
        layout_features.addWidget(self.lista_features)

        layout_target = QVBoxLayout()
        self.lbl_target = QLabel("Columna de salida (Target) - selección única:")
        self.combo_target = QComboBox()
        self.combo_target.currentTextChanged.connect(self.al_cambiar_target)
        layout_target.addWidget(self.lbl_target)
        layout_target.addWidget(self.combo_target)
        layout_target.addStretch()

        layout_seleccion.addLayout(layout_features)
        layout_seleccion.addLayout(layout_target)
        layout_principal.addLayout(layout_seleccion)

        self.btn_confirmar = QPushButton("Confirmar selección")
        self.btn_confirmar.clicked.connect(self.confirmar_seleccion_columnas)
        layout_principal.addWidget(self.btn_confirmar)

        # Resumen persistente de la selección confirmada
        self.lbl_seleccion_confirmada = QLabel()
        self.lbl_seleccion_confirmada.setWordWrap(True)
        layout_principal.addWidget(self.lbl_seleccion_confirmada)

        # Tabla con solo las columnas confirmadas (entradas + salida), con todos los datos
        self.tabla_seleccion = QTableWidget()
        self.tabla_seleccion.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout_principal.addWidget(self.tabla_seleccion)
        self.mostrar_seleccion_confirmada()

        # -------------------------------------------------------------
        # Separación en Entrenamiento y Test
        # -------------------------------------------------------------
        group_separacion = QGroupBox("Separación en Entrenamiento y Test")
        layout_sep = QVBoxLayout()

        layout_params = QHBoxLayout()
        layout_params.addWidget(QLabel("% Test:"))
        self.input_porcentaje_test = QLineEdit(str(PORCENTAJE_TEST_POR_DEFECTO))
        self.input_porcentaje_test.textChanged.connect(self.actualizar_porcentaje_train)
        layout_params.addWidget(self.input_porcentaje_test)
        self.lbl_porcentaje_train = QLabel()
        layout_params.addWidget(self.lbl_porcentaje_train)
        layout_params.addWidget(QLabel("Semilla:"))
        self.input_semilla = QLineEdit(str(SEMILLA_POR_DEFECTO))
        layout_params.addWidget(self.input_semilla)
        layout_sep.addLayout(layout_params)

        self.btn_separar = QPushButton("Separar datos")
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
        # Tratamiento de Valores Inexistentes (NaN)
        # -------------------------------------------------------------
        self.group_preprocesado = QGroupBox("Tratamiento de Valores Inexistentes (NaN)")
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

        self.group_preprocesado.setLayout(layout_prep)
        layout_principal.addWidget(self.group_preprocesado)

        # Estado inicial del flujo: sin separación, preprocesado y modelo deshabilitados
        self.invalidar_separacion()

    # -------------------------------------------------------------
    # Separación Train/Test y orden del flujo
    # -------------------------------------------------------------
    def actualizar_porcentaje_train(self, texto: str):
        """Calcula automáticamente el porcentaje de entrenamiento (complementario al de test)."""
        try:
            pct_test = float(texto.strip().replace(",", "."))
            if 1 <= pct_test <= 99:
                self.lbl_porcentaje_train.setText(f"→ Entrenamiento: {100 - pct_test:g} %")
                return
        except ValueError:
            pass
        self.lbl_porcentaje_train.setText("→ Entrenamiento: —")

    def actualizar_estado_flujo(self):
        """Separar solo tras confirmar columnas; preprocesado y modelo solo tras separar."""
        hay_seleccion = bool(self.features_confirmadas and self.target_confirmado)
        hay_separacion = self.df_train is not None and self.df_test is not None
        self.btn_separar.setEnabled(self.df_cargado is not None and hay_seleccion)
        self.group_preprocesado.setEnabled(hay_separacion)

    def invalidar_separacion(self):
        """Descarta la separación y todo lo que dependa de ella (preprocesado y modelo)."""
        self.df_train = None
        self.df_test = None
        self.modelo = None
        self.lbl_resultado_separacion.setText("Pendiente: confirma la selección de columnas y pulsa «Separar datos».")
        self.actualizar_estado_flujo()
        self.actualizar_reporte_nulos()

    def mostrar_resultado_separacion(self):
        n_train, n_test = len(self.df_train), len(self.df_test)
        self.lbl_resultado_separacion.setText(
            f"<b>División realizada correctamente</b> "
            f"(test: {self.porcentaje_test_usado:g} %, entrenamiento: {100 - self.porcentaje_test_usado:g} %, "
            f"semilla: {self.semilla_usada}).<br>"
            f"<b>Entrenamiento:</b> {n_train} filas<br>"
            f"<b>Test:</b> {n_test} filas"
        )

    def separar_datos(self):
        if self.df_cargado is None or not (self.features_confirmadas and self.target_confirmado):
            QMessageBox.warning(
                self, "Aviso",
                "Primero debes cargar un dataset y confirmar la selección de columnas."
            )
            return

        try:
            pct_test, semilla = parsear_parametros_separacion(
                self.input_porcentaje_test.text(), self.input_semilla.text()
            )
            df_train, df_test = separar_train_test(self.df_cargado, pct_test, semilla)
        except ValueError as err:
            QMessageBox.critical(self, "Error en la separación", str(err))
            return

        # Una nueva separación sustituye a la anterior y descarta preprocesado y modelo previos
        self.df_train = df_train
        self.df_test = df_test
        self.modelo = None
        self.porcentaje_test_usado = pct_test
        self.semilla_usada = semilla

        self.mostrar_resultado_separacion()
        self.actualizar_estado_flujo()
        self.actualizar_reporte_nulos()
        QMessageBox.information(
            self,
            "Separación realizada",
            f"La división se ha realizado correctamente.\n"
            f"Entrenamiento: {len(df_train)} filas\n"
            f"Test: {len(df_test)} filas"
        )

    def al_cambiar_estrategia(self, index: int):
        """Habilita el campo constante solo cuando se elige la opción 3 (Valor constante)."""
        self.input_constante.setEnabled(index == 3)

    def columnas_evaluadas(self) -> list:
        """Columnas de entrada y de salida confirmadas."""
        return self.features_confirmadas + ([self.target_confirmado] if self.target_confirmado else [])

    def listo_para_crear_modelo(self) -> bool:
        """La creación del modelo solo está disponible con la separación hecha y sin
        valores inexistentes en las columnas seleccionadas (ni en train ni en test)."""
        columnas = self.columnas_evaluadas()
        if self.df_train is None or self.df_test is None or not columnas:
            return False
        return not hay_valores_inexistentes(self.df_train, self.df_test, columnas)

    def actualizar_reporte_nulos(self):
        """Muestra el número de valores inexistentes por columna, en Train y en Test por separado."""
        if self.df_train is None or self.df_test is None:
            self.lbl_reporte_nulos.setText("Estado: Pendiente de separación en Train/Test.")
            self.btn_aplicar_preprocesado.setEnabled(False)
            return

        columnas_evaluar = self.columnas_evaluadas()
        if not columnas_evaluar:
            self.lbl_reporte_nulos.setText("Selecciona y confirma las columnas primero.")
            self.btn_aplicar_preprocesado.setEnabled(False)
            return

        reporte = detectar_valores_inexistentes(
            self.df_train, self.df_test, columnas_evaluar
        )
        reporte_train = reporte["entrenamiento"]
        reporte_test = reporte["test"]

        total_nulos_train = sum(reporte_train.values())
        total_nulos_test = sum(reporte_test.values())

        if total_nulos_train == 0 and total_nulos_test == 0:
            self.lbl_reporte_nulos.setText(
                "No hay valores inexistentes en las columnas seleccionadas. "
                "Puedes continuar sin preprocesar: los conjuntos están listos para crear el modelo."
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
                "<b>Valores inexistentes detectados en las columnas seleccionadas:</b>"
                "<table border='1' cellspacing='0' cellpadding='4'>"
                f"<tr><th>Columna</th><th>Entrenamiento ({len(self.df_train)} filas)</th>"
                f"<th>Test ({len(self.df_test)} filas)</th></tr>"
                f"{filas}"
                f"<tr><td><b>Total</b></td><td align='center'><b>{total_nulos_train}</b></td>"
                f"<td align='center'><b>{total_nulos_test}</b></td></tr>"
                "</table>"
                "La creación del modelo no estará disponible mientras queden valores inexistentes."
            )
            self.lbl_reporte_nulos.setText(txt)
            self.btn_aplicar_preprocesado.setEnabled(True)

    def ejecutar_preprocesado(self):
        """Ejecuta la estrategia de tratamiento de nulos seleccionada."""
        if self.df_train is None or self.df_test is None:
            QMessageBox.warning(self, "Aviso", "No existen los conjuntos de Entrenamiento y Test.")
            return

        columnas_evaluar = self.columnas_evaluadas()
        if not columnas_evaluar:
            QMessageBox.warning(self, "Aviso", "Debes confirmar las columnas a evaluar.")
            return

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
                valor_constante = parsear_valor_constante(texto_constante)
            except ValueError as err:
                QMessageBox.critical(self, "Error de Tipo", str(err))
                return

        try:
            df_train_limpio, df_test_limpio, resumen = tratar_valores_inexistentes(
                self.df_train,
                self.df_test,
                columnas_evaluar,
                estrategia,
                valor_constante
            )
        except ConjuntoVacioError as err:
            QMessageBox.warning(
                self,
                "Operación Cancelada",
                f"{err}\nNo se ha aplicado ningún cambio. Cambia de estrategia."
            )
            return
        except ValueError as err:
            QMessageBox.critical(self, "Error en Preprocesado", str(err))
            return
        except Exception as err:
            QMessageBox.critical(self, "Error en Preprocesado", f"Ocurrió un error al procesar:\n{str(err)}")
            return

        # Actualizar estado interno (el archivo original en disco no se toca)
        self.df_train = df_train_limpio
        self.df_test = df_test_limpio

        # Mensaje de éxito con el resultado en cada conjunto
        if estrategia == "eliminar":
            msg = (
                "Preprocesado aplicado: se han eliminado filas con valores inexistentes.\n"
                f"Entrenamiento: {resumen['filas_eliminadas_train']} filas eliminadas "
                f"({len(self.df_train)} restantes)\n"
                f"Test: {resumen['filas_eliminadas_test']} filas eliminadas "
                f"({len(self.df_test)} restantes)"
            )
        else:
            if estrategia == "constante":
                con_que = f"el valor constante {valor_constante:g}"
            else:
                con_que = f"la {estrategia} del conjunto de entrenamiento"
            msg = (
                f"Preprocesado aplicado: valores inexistentes rellenados con {con_que}.\n"
                f"Entrenamiento: {resumen['valores_rellenados_train']} valores rellenados\n"
                f"Test: {resumen['valores_rellenados_test']} valores rellenados"
            )

        QMessageBox.information(self, "Preprocesado Exitoso", msg)
        self.mostrar_resultado_separacion()
        self.actualizar_reporte_nulos()

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
            df_nuevo.columns = [str(c) for c in df_nuevo.columns]

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
        """Muestra en la ventana las columnas confirmadas (o que aún no hay selección)."""
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
        """Muestra como un CSV solo las columnas confirmadas: entradas primero y salida al final."""
        nombres = {str(c): c for c in self.df_cargado.columns}
        orden = self.features_confirmadas + [self.target_confirmado]
        df_sel = self.df_cargado[[nombres[c] for c in orden]]
        etiquetas = [f"{c}\n(entrada)" for c in self.features_confirmadas] + [f"{self.target_confirmado}\n(salida)"]
        self.mostrar_dataframe_en_tabla(df_sel, self.tabla_seleccion, etiquetas)

        # Resaltar la columna de salida
        col_target = len(orden) - 1
        fondo = QBrush(QColor("#cfe8ff"))
        texto = QBrush(QColor("#000000"))
        for r in range(self.tabla_seleccion.rowCount()):
            item = self.tabla_seleccion.item(r, col_target)
            item.setBackground(fondo)
            item.setForeground(texto)

    def al_cambiar_target(self, texto: str):
        """La columna elegida como salida no puede ser también de entrada:
        se deselecciona y se deshabilita en la lista de features."""
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
            self.df_cargado,
            features_seleccionadas,
            target_seleccionado
        )

        if es_valido:
            cambio = (
                set(features_seleccionadas) != set(self.features_confirmadas)
                or target_seleccionado != self.target_confirmado
            )
            self.features_confirmadas = features_seleccionadas
            self.target_confirmado = target_seleccionado
            self.mostrar_seleccion_confirmada()
            if cambio:
                # Otra selección de columnas obliga a repetir la separación
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