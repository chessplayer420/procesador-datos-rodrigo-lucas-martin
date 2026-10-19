
from pathlib import Path

from kivy.app import App
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput


class KivyPOCApp(App):
    def build(self):
        self.title = "Prueba de concepto - Kivy"

        root = BoxLayout(
            orientation="vertical",
            padding=dp(12),
            spacing=dp(8),
        )

        # Selector de archivos
        self.file_label = Label(
            text="Ningún archivo seleccionado",
            size_hint_y=None,
            height=dp(35),
        )

        file_button = Button(
            text="Seleccionar archivo",
            size_hint_y=None,
            height=dp(45),
        )
        file_button.bind(on_release=self.open_file_dialog)

        root.add_widget(file_button)
        root.add_widget(self.file_label)

        # Tabla de ejemplo con desplazamiento vertical
        table_scroll = ScrollView()

        table = GridLayout(
            cols=3,
            spacing=dp(1),
            size_hint_y=None,
        )
        table.bind(minimum_height=table.setter("height"))

        rows = [
            ("ID", "Nombre", "Valor"),
            ("1", "Elemento A", "10.5"),
            ("2", "Elemento B", "20.3"),
            ("3", "Elemento C", "15.8"),
            ("4", "Elemento D", "31.2"),
            ("5", "Elemento E", "12.4"),
            ("6", "Elemento F", "18.9"),
            ("7", "Elemento G", "25.1"),
            ("8", "Elemento H", "16.7"),
            ("9", "Elemento I", "22.6"),
            ("10", "Elemento J", "19.4"),
        ]

        for row in rows:
            for value in row:
                cell = Label(
                    text=value,
                    size_hint_y=None,
                    height=dp(38),
                )
                table.add_widget(cell)

        table_scroll.add_widget(table)
        root.add_widget(table_scroll)

        # Desplegable de selección simple
        self.options = Spinner(
            text="Selecciona una opción",
            values=("Opción A", "Opción B", "Opción C"),
            size_hint_y=None,
            height=dp(45),
        )
        root.add_widget(self.options)

        # Campo de texto
        self.text_input = TextInput(
            hint_text="Introduce un valor",
            multiline=False,
            size_hint_y=None,
            height=dp(45),
        )
        root.add_widget(self.text_input)

        # Botón de confirmación
        result_button = Button(
            text="Mostrar selección",
            size_hint_y=None,
            height=dp(45),
        )
        result_button.bind(on_release=self.show_result)
        root.add_widget(result_button)

        return root

    def open_file_dialog(self, _button):
        chooser = FileChooserListView(
            path=str(Path.home()),
            filters=["*.csv", "*.txt", "*.xlsx", "*.json"],
        )

        content = BoxLayout(
            orientation="vertical",
            spacing=dp(8),
            padding=dp(8),
        )
        content.add_widget(chooser)

        actions = BoxLayout(
            size_hint_y=None,
            height=dp(45),
            spacing=dp(8),
        )

        popup = Popup(
            title="Seleccionar archivo",
            content=content,
            size_hint=(0.9, 0.9),
        )

        def select_file(_button):
            if chooser.selection:
                self.file_label.text = chooser.selection[0]
                popup.dismiss()

        select_button = Button(text="Seleccionar")
        select_button.bind(on_release=select_file)

        cancel_button = Button(text="Cancelar")
        cancel_button.bind(on_release=popup.dismiss)

        actions.add_widget(select_button)
        actions.add_widget(cancel_button)
        content.add_widget(actions)

        popup.open()

    def show_result(self, _button):
        selected = self.options.text
        entered = self.text_input.text.strip()

        message = (
            f"Opción seleccionada: {selected}\n"
            f"Valor introducido: {entered or '(vacío)'}"
        )

        Popup(
            title="Resultado",
            content=Label(text=message),
            size_hint=(0.8, 0.35),
        ).open() 


if __name__ == "__main__":
    KivyPOCApp().run()
