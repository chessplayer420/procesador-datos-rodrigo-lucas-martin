# src/procesador_datos_rodrigo_lucas_martin/importacion.py

import pandas as pd
import sqlite3
from pathlib import Path

def importar_datos(ruta_archivo: str) -> pd.DataFrame:
    """
    Importa datos desde un archivo CSV, Excel o base de datos SQLite.
    """
    ruta = Path(ruta_archivo)

    # 1. Comprobar si el archivo existe
    if not ruta.is_file():
        raise FileNotFoundError(f"El archivo no existe: {ruta_archivo}")

    extension = ruta.suffix.lower()

    # 2. Comprobar extensión soportada (Fuera del try)
    if extension not in ['.csv', '.xlsx', '.xls', '.sqlite', '.db']:
        raise ValueError(f"La extensión del archivo no está soportada: {extension}")

    # 3. Comprobar si el archivo está físicamente vacío (0 bytes)
    # Lo ponemos DESPUÉS de la extensión, para que archivos de 0 bytes no soportados
    # den el error de extensión primero.
    if ruta.stat().st_size == 0:
        if extension in ['.sqlite', '.db']:
             raise ValueError("La base de datos SQLite no contiene ninguna tabla.")
        raise ValueError("El archivo no contiene datos (está vacío).")

    try:
        # 4. Leer según el formato
        if extension == '.csv':
            df = pd.read_csv(ruta)
            
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(ruta)
            
        elif extension in ['.sqlite', '.db']:
            with sqlite3.connect(ruta) as conn:
                query_tablas = "SELECT name FROM sqlite_master WHERE type='table';"
                tablas = pd.read_sql(query_tablas, conn)
                
                if len(tablas) == 0:
                    raise ValueError("La base de datos SQLite no contiene ninguna tabla.")
                if len(tablas) > 1:
                    raise ValueError("La base de datos SQLite contiene más de una tabla.")
                
                nombre_tabla = tablas.iloc[0]['name']
                df = pd.read_sql(f"SELECT * FROM {nombre_tabla}", conn)

    except ValueError as e:
        mensaje = str(e)
        # Si es un error en inglés de pandas sobre el formato de Excel, lo convertimos a nuestro error
        if "Excel file format cannot be determined" in mensaje:
            raise Exception("El archivo está dañado o su contenido no corresponde a su extensión")
        # Si es nuestro propio ValueError (ej. de sqlite), lo dejamos pasar
        raise e
    except Exception as e:
        # Cualquier otro error interno de lectura
        raise Exception("El archivo está dañado o su contenido no corresponde a su extensión")

    # 5. Comprobar si se leyó correctamente pero está vacío lógicamente
    if df.empty and len(df.columns) == 0:
        raise ValueError("El archivo no contiene datos.")

    return df