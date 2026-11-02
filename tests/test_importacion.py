# tests/test_importacion.py

import pytest
import pandas as pd
import sqlite3
from importacion import importar_datos

# ==========================================
# TEST DE ÉXITO 
# ==========================================

def test_importar_csv_exito(tmp_path):
    archivo = tmp_path / "datos.csv"
    archivo.write_text("id,valor\n1,10.5\n2,20.0")
    df = importar_datos(str(archivo))
    assert len(df) == 2
    assert "valor" in df.columns

def test_importar_excel_exito(tmp_path):
    archivo = tmp_path / "datos.xlsx"
    df_dummy = pd.DataFrame({"id": [1, 2], "valor": [10.5, 20.0]})
    df_dummy.to_excel(archivo, index=False)
    
    df = importar_datos(str(archivo))
    assert len(df) == 2
    assert "valor" in df.columns

def test_importar_sqlite_exito(tmp_path):
    archivo = tmp_path / "datos.sqlite"
    with sqlite3.connect(archivo) as conn:
        conn.execute("CREATE TABLE mi_tabla (id INTEGER, valor REAL)")
        conn.execute("INSERT INTO mi_tabla VALUES (1, 10.5), (2, 20.0)")
    
    df = importar_datos(str(archivo))
    assert len(df) == 2
    assert "valor" in df.columns

# ==========================================
# TEST DE CASOS DE ERROR
# ==========================================

def test_error_archivo_no_existe():
    with pytest.raises(FileNotFoundError, match="El archivo no existe"):
        importar_datos("ruta_inventada.csv")

def test_error_archivo_vacio(tmp_path):
    archivo_vacio = tmp_path / "vacio.csv"
    archivo_vacio.write_text("")  # 0 bytes
    with pytest.raises(ValueError, match="El archivo no contiene datos"):
        importar_datos(str(archivo_vacio))

def test_error_extension_no_soportada(tmp_path):
    archivo_txt = tmp_path / "datos.txt"
    archivo_txt.write_text("texto aleatorio")
    with pytest.raises(ValueError, match="La extensión del archivo no está soportada"):
        importar_datos(str(archivo_txt))

def test_error_archivo_danado_o_incompatible(tmp_path):
    archivo_danado = tmp_path / "falso_excel.xlsx"
    archivo_danado.write_text("Esto no es un archivo excel real, provocará un error de lectura.")
    
    with pytest.raises(Exception, match="El archivo está dañado o su contenido no corresponde a su extensión"):
        importar_datos(str(archivo_danado))

def test_error_sqlite_sin_tablas(tmp_path):
    archivo = tmp_path / "vacia.db"
    # Crea el archivo de 0 bytes (sin tablas)
    archivo.write_text("")
    
    with pytest.raises(ValueError, match="La base de datos SQLite no contiene ninguna tabla."):
        importar_datos(str(archivo))

def test_error_sqlite_multiples_tablas(tmp_path):
    archivo = tmp_path / "multi.db"
    with sqlite3.connect(archivo) as conn:
        conn.execute("CREATE TABLE tabla1 (id INTEGER)")
        conn.execute("CREATE TABLE tabla2 (id INTEGER)")
    
    with pytest.raises(ValueError, match="La base de datos SQLite contiene más de una tabla."):
        importar_datos(str(archivo))