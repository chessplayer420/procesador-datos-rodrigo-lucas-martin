import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QComboBox, QLineEdit, QLabel, QMessageBox, QHeaderView
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class VentanaPrototipo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prueba de Concepto GUI - PySide6")
        self.resize(800, 600)

        # Contenedor principal
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QVBoxLayout(widget_central)

        # 1. Diálogo de selección de archivos
        layout_archivo = QHBoxLayout()
        btn_archivo = QPushButton("Seleccionar Archivo")
        btn_archivo.clicked.connect(self.abrir_dialogo_archivo)
        self.lbl_ruta_archivo = QLabel("Ningún archivo seleccionado")
        
        layout_archivo.addWidget(btn_archivo)
        layout_archivo.addWidget(self.lbl_ruta_archivo)
        layout_principal.addLayout(layout_archivo)

        # 2. Desplegable y campo de texto con botón
        layout_entradas = QHBoxLayout()
        self.combo_opciones = QComboBox()
        self.combo_opciones.addItems(["Opción 1", "Opción 2", "Opción 3"])

        self.input_texto = QLineEdit()
        self.input_texto.setPlaceholderText("Introduce un texto aquí...")

        btn_mostrar = QPushButton("Mostrar Selección")
        btn_mostrar.clicked.connect(self.mostrar_mensaje_seleccion)

        layout_entradas.addWidget(QLabel("Desplegable:"))
        layout_entradas.addWidget(self.combo_opciones)
        layout_entradas.addWidget(QLabel("Texto:"))
        layout_entradas.addWidget(self.input_texto)
        layout_entradas.addWidget(btn_mostrar)
        layout_principal.addLayout(layout_entradas)

        # 3. Tabla con scroll para mostrar datos
        self.tabla_datos = QTableWidget()
        layout_principal.addWidget(self.tabla_datos)
        self.cargar_datos_ejemplo()

        # 4. Integración de gráfico de Matplotlib
        self.canvas_matplotlib = FigureCanvas(Figure(figsize=(5, 3)))
        layout_principal.addWidget(self.canvas_matplotlib)
        self.dibujar_grafico_ejemplo()

    def abrir_dialogo_archivo(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo", "", "Todos los archivos (*)")
        if ruta:
            self.lbl_ruta_archivo.setText(ruta)

    def mostrar_mensaje_seleccion(self):
        opcion = self.combo_opciones.currentText()
        texto = self.input_texto.text()
        QMessageBox.information(
            self,
            "Valores Introducidos",
            f"Desplegable: {opcion}\nTexto ingresado: {texto}"
        )

    def cargar_datos_ejemplo(self, df: pd.DataFrame = None):
        if df is None:
            datos = {
                "ID": [1, 2, 3, 4, 5],
                "Nombre": ["Elemento A", "Elemento B", "Elemento C", "Elemento D", "Elemento E"],
                "Valor": [10.5, 23.0, 15.2, 42.8, 31.1]
            }
            df = pd.DataFrame(datos)

        self.tabla_datos.setRowCount(len(df))
        self.tabla_datos.setColumnCount(len(df.columns))
        self.tabla_datos.setHorizontalHeaderLabels(list(df.columns))

        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                self.tabla_datos.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

        self.tabla_datos.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def dibujar_grafico_ejemplo(self):
        fig = self.canvas_matplotlib.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3, 4], [10, 20, 15, 25], marker='o', color='b')
        ax.set_title("Muestra de Gráfico Integrado (Matplotlib)")
        self.canvas_matplotlib.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrototipo()
    ventana.show()
    sys.exit(app.exec())