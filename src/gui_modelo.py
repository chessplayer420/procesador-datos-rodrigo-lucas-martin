import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QTextEdit, QFileDialog
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from src.modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
        normalizar_ruta_guardado,
    )
except ImportError:
    from modelo import (
        entrenar_modelo_regresion,
        actualizar_descripcion_modelo,
        guardar_modelo,
        normalizar_ruta_guardado,
    )


class VentanaModeloRegresion(QMainWindow):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.setWindowTitle("Creación y Evaluación del Modelo de Regresión Lineal")
        self.resize(950, 820)
        self.setMinimumSize(800, 650)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.resultado_modelo = None

        # Contenedor y Layout Principal
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout(widget_central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 1. Botones "Crear modelo" y "Guardar modelo"
        layout_botones = QHBoxLayout()

        self.btn_crear = QPushButton("Crear modelo")
        self.btn_crear.setFixedHeight(40)
        self.btn_crear.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background-color: #0078D4;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.btn_crear.clicked.connect(self.crear_modelo)
        layout_botones.addWidget(self.btn_crear)

        self.btn_guardar = QPushButton("Guardar modelo")
        self.btn_guardar.setFixedHeight(40)
        self.btn_guardar.setEnabled(False)  # Solo habilitado cuando hay un modelo creado o cargado
        self.btn_guardar.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: bold;
                background-color: #107C41;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0B5A2E;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        self.btn_guardar.clicked.connect(self.ejecutar_guardado_modelo)
        layout_botones.addWidget(self.btn_guardar)

        layout.addLayout(layout_botones)

        # Comprobar estado inicial del botón
        self.validar_requisitos_modelo()

        # 2. Etiqueta para la Fórmula del Modelo
        self.lbl_formula = QLabel("<b>Fórmula del modelo:</b> <i>(Pendiente de creación)</i>")
        self.lbl_formula.setWordWrap(True)
        self.lbl_formula.setStyleSheet("font-size: 13px; padding: 5px; background-color: #F3F3F3; border-radius: 4px;")
        layout.addWidget(self.lbl_formula)

        # 3. Tabla para Métricas (R² y ECM)
        self.tabla_metricas = QTableWidget(2, 2)
        self.tabla_metricas.setHorizontalHeaderLabels(["Entrenamiento", "Test"])
        self.tabla_metricas.setVerticalHeaderLabels(["R² (Coef. Determinación)", "Error Cuadrático Medio (ECM)"])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.setFixedHeight(95)
        layout.addWidget(self.tabla_metricas)

        # 4. Área de texto de varias líneas para la Descripción del Modelo (#114)
        self.lbl_descripcion = QLabel("<b>Descripción (opcional):</b>")
        layout.addWidget(self.lbl_descripcion)

        self.input_descripcion = QTextEdit()
        self.input_descripcion.setPlaceholderText(
            "Descripción (opcional): escribe aquí observaciones sobre el modelo, las variables o las métricas..."
        )
        self.input_descripcion.setFixedHeight(75)
        self.input_descripcion.setEnabled(False)
        self.input_descripcion.textChanged.connect(self.al_cambiar_descripcion)
        layout.addWidget(self.input_descripcion)

        # 5. Canvas de Matplotlib para el Gráfico de Ajuste
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

    def validar_requisitos_modelo(self):
        requisitos_ok = bool(
            self.X_train is not None and self.y_train is not None and
            self.X_test is not None and self.y_test is not None and
            self.X_train.isna().sum().sum() == 0 and
            self.y_train.isna().sum() == 0 and
            self.X_test.isna().sum().sum() == 0 and
            self.y_test.isna().sum() == 0
        )
        self.btn_crear.setEnabled(requisitos_ok)

    def al_cambiar_descripcion(self):
        if self.resultado_modelo is not None:
            actualizar_descripcion_modelo(
                self.resultado_modelo,
                self.input_descripcion.toPlainText()
            )

    def cargar_modelo(self, estructura_modelo: dict):
        self.resultado_modelo = estructura_modelo
        descripcion = estructura_modelo.get("descripcion", "") if isinstance(estructura_modelo, dict) else ""
        self.input_descripcion.blockSignals(True)
        self.input_descripcion.setPlainText(descripcion)
        self.input_descripcion.blockSignals(False)
        hay_modelo = self.resultado_modelo is not None
        self.input_descripcion.setEnabled(hay_modelo)
        self.btn_guardar.setEnabled(hay_modelo)

    def crear_modelo(self):
        try:
            res = entrenar_modelo_regresion(self.X_train, self.y_train, self.X_test, self.y_test)
            res.setdefault("descripcion", "")
            self.resultado_modelo = res

            # Al crear un nuevo modelo, vaciar descripción y habilitar descripción y guardado
            self.input_descripcion.blockSignals(True)
            self.input_descripcion.clear()
            self.input_descripcion.blockSignals(False)
            self.input_descripcion.setEnabled(True)
            self.btn_guardar.setEnabled(True)

            # Actualizar Texto de la Fórmula
            self.lbl_formula.setText(f"<b>Fórmula del modelo:</b> {res['formula']}")

            # Actualizar Tabla de Métricas en la GUI
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

            # Renderizar Gráfico de Ajuste
            self.dibujar_grafico(res["y_pred_train"], res["y_pred_test"], res["modelo"])
            QMessageBox.information(self, "Éxito", "El modelo de regresión lineal se ha creado y evaluado correctamente.")

        except Exception as err:
            QMessageBox.critical(self, "Error al crear el modelo", f"Ocurrió un error inesperado:\n{str(err)}")

    def ejecutar_guardado_modelo(self):
        if self.resultado_modelo is None:
            return

        descripcion_actual = self.input_descripcion.toPlainText().strip()
        actualizar_descripcion_modelo(self.resultado_modelo, self.input_descripcion.toPlainText())

        # Aviso si la descripción está vacía (sin impedir el guardado)
        if not descripcion_actual:
            respuesta = QMessageBox.warning(
                self,
                "Descripción vacía",
                "El modelo no tiene descripción. ¿Deseas guardar el modelo sin descripción?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if respuesta == QMessageBox.No:
                return

        ruta_elegida, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar modelo",
            "modelo_lineal.json",
            "Archivos JSON (*.json);;Todos los archivos (*)",
            options=QFileDialog.DontConfirmOverwrite,
        )

        if not ruta_elegida:
            return

        ruta_final = normalizar_ruta_guardado(ruta_elegida)

        # Confirmar antes de sobrescribir si el archivo ya existe
        if ruta_final.exists():
            confirmar = QMessageBox.question(
                self,
                "Confirmar sobrescritura",
                f"El archivo «{ruta_final.name}» ya existe.\n¿Deseas sobrescribirlo?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if confirmar != QMessageBox.Yes:
                return

        try:
            ruta_guardada = guardar_modelo(self.resultado_modelo, ruta_final)
            QMessageBox.information(
                self,
                "Modelo guardado",
                f"El modelo se ha guardado correctamente en:\n{ruta_guardada}",
            )
        except Exception as err:
            QMessageBox.critical(
                self,
                "Error al guardar el modelo",
                f"No se pudo guardar el archivo en la ubicación seleccionada:\n{str(err)}",
            )

    def dibujar_grafico(self, y_pred_train, y_pred_test, modelo):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        es_regresion_simple = self.X_train.shape[1] == 1

        if es_regresion_simple:
            col_x = self.X_train.columns[0]
            ax.scatter(self.X_train[col_x], self.y_train, color="#1f77b4", label="Entrenamiento", alpha=0.7, edgecolors="none")
            ax.scatter(self.X_test[col_x], self.y_test, color="#ff7f0e", label="Test", alpha=0.7, edgecolors="none")

            x_min = min(self.X_train[col_x].min(), self.X_test[col_x].min())
            x_max = max(self.X_train[col_x].max(), self.X_test[col_x].max())
            x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            y_line = modelo.predict(pd.DataFrame(x_line, columns=[col_x]))

            ax.plot(x_line, y_line, color="#d62728", linewidth=2, label="Recta de ajuste")
            ax.set_xlabel(col_x)
            ax.set_ylabel(self.y_train.name)
        else:
            ax.scatter(self.y_train, y_pred_train, color="#1f77b4", label="Entrenamiento", alpha=0.7, edgecolors="none")
            ax.scatter(self.y_test, y_pred_test, color="#ff7f0e", label="Test", alpha=0.7, edgecolors="none")

            val_min = min(self.y_train.min(), self.y_test.min())
            val_max = max(self.y_train.max(), self.y_test.max())
            ax.plot([val_min, val_max], [val_min, val_max], color="#d62728", linestyle="--", label="Valor real = Predicho")

            ax.set_xlabel("Valor real")
            ax.set_ylabel("Valor predicho")

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        ax.set_title("Gráfico de Ajuste del Modelo")
        fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    X_tr = pd.DataFrame({"Entrada": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_tr = pd.Series([2.1, 3.9, 6.1, 8.2, 9.8], name="Salida")
    X_te = pd.DataFrame({"Entrada": [6.0, 7.0]})
    y_te = pd.Series([11.9, 14.1], name="Salida")

    win = VentanaModeloRegresion(X_tr, y_tr, X_te, y_te)
    win.show()
    sys.exit(app.exec())