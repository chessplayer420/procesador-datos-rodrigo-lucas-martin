import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from modelo import entrenar_modelo_regresion


class VentanaModeloRegresion(QMainWindow):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.setWindowTitle("Creación y Evaluación del Modelo de Regresión Lineal")
        self.resize(900, 700)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout(widget_central)

        # 1. Botón "Crear modelo"
        self.btn_crear = QPushButton("Crear modelo")
        self.btn_crear.clicked.connect(self.crear_modelo)
        layout.addWidget(self.btn_crear)

        # Habilitación del botón solo si se cumplen los requisitos
        self.validar_requisitos_modelo()

        # 2. Etiqueta para la Fórmula
        self.lbl_formula = QLabel("Fórmula del modelo: (Pendiente de creación)")
        self.lbl_formula.setWordWrap(True)
        layout.addWidget(self.lbl_formula)

        # 3. Tabla para Métricas (R² y ECM)
        self.tabla_metricas = QTableWidget(2, 2)
        self.tabla_metricas.setHorizontalHeaderLabels(["Entrenamiento", "Test"])
        self.tabla_metricas.setVerticalHeaderLabels(["R²", "Error Cuadrático Medio (ECM)"])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tabla_metricas)

        # 4. Canvas de Matplotlib para el gráfico de ajuste
        self.canvas = FigureCanvas(Figure(figsize=(6, 4)))
        layout.addWidget(self.canvas)

    def validar_requisitos_modelo(self):
        # Comprobar si existen datos y si no quedan NaNs
        requisitos_ok = bool(
            self.X_train is not None and self.y_train is not None and
            self.X_test is not None and self.y_test is not None and
            self.X_train.isna().sum().sum() == 0 and
            self.y_train.isna().sum() == 0 and
            self.X_test.isna().sum().sum() == 0 and
            self.y_test.isna().sum() == 0
        )
        self.btn_crear.setEnabled(requisitos_ok)

    def crear_modelo(self):
        try:
            res = entrenar_modelo_regresion(self.X_train, self.y_train, self.X_test, self.y_test)

            # Actualizar Fórmula y Métricas
            self.lbl_formula.setText(f"<b>Fórmula:</b> {res['formula']}")
            m = res["metricas"]
            self.tabla_metricas.setItem(0, 0, QTableWidgetItem(f"{m['r2_train']:.4f}"))
            self.tabla_metricas.setItem(0, 1, QTableWidgetItem(f"{m['r2_test']:.4f}"))
            self.tabla_metricas.setItem(1, 0, QTableWidgetItem(f"{m['ecm_train']:.4f}"))
            self.tabla_metricas.setItem(1, 1, QTableWidgetItem(f"{m['ecm_test']:.4f}"))

            # Renderizar Gráfico
            self.dibujar_grafico(res["y_pred_train"], res["y_pred_test"], res["modelo"])
            QMessageBox.information(self, "Éxito", "El modelo se ha creado y ajustado correctamente.")

        except Exception as err:
            QMessageBox.critical(self, "Error al crear el modelo", f"Ocurrió un error:\n{str(err)}")

    def dibujar_grafico(self, y_pred_train, y_pred_test, modelo):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if self.X_train.shape[1] == 1:
            col_x = self.X_train.columns[0]
            ax.scatter(self.X_train[col_x], self.y_train, color="blue", label="Entrenamiento", alpha=0.7)
            ax.scatter(self.X_test[col_x], self.y_test, color="orange", label="Test", alpha=0.7)

            x_min = min(self.X_train[col_x].min(), self.X_test[col_x].min())
            x_max = max(self.X_train[col_x].max(), self.X_test[col_x].max())
            x_line = np.linspace(x_min, x_max, 100).reshape(-1, 1)
            y_line = modelo.predict(pd.DataFrame(x_line, columns=[col_x]))

            ax.plot(x_line, y_line, color="red", linewidth=2, label="Recta de ajuste")
            ax.set_xlabel(col_x)
            ax.set_ylabel(self.y_train.name)
        else:
            ax.scatter(self.y_train, y_pred_train, color="blue", label="Entrenamiento", alpha=0.7)
            ax.scatter(self.y_test, y_pred_test, color="orange", label="Test", alpha=0.7)

            val_min = min(self.y_train.min(), self.y_test.min())
            val_max = max(self.y_train.max(), self.y_test.max())
            ax.plot([val_min, val_max], [val_min, val_max], color="red", linestyle="--", label="Valor real = Predicho")

            ax.set_xlabel("Valor real")
            ax.set_ylabel("Valor predicho")

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