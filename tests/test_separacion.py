# tests/test_separacion.py

import pytest
import pandas as pd
from separacion import separar_datos


def crear_df_ejemplo(filas: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(filas),
        "x": [i * 2.5 for i in range(filas)],
        "y": [i + 10 for i in range(filas)]
    })


# ==========================================
# PRUEBAS DE ÉXITO Y CRITERIOS PRINCIPALES
# ==========================================

def test_tamanos_corresponden_al_porcentaje():
    df = crear_df_ejemplo(100)
    df_train, df_test = separar_datos(df, porcentaje_test=25, semilla=42)

    assert len(df_train) == 75
    assert len(df_test) == 25
    assert len(df_train) + len(df_test) == len(df)


def test_ninguna_fila_aparece_en_ambos_conjuntos():
    df = crear_df_ejemplo(50)
    df_train, df_test = separar_datos(df, porcentaje_test=20, semilla=123)

    # Comprobar que los índices y los IDs son disjuntos (sin elementos en común)
    indices_train = set(df_train.index)
    indices_test = set(df_test.index)
    assert indices_train.isdisjoint(indices_test)

    ids_train = set(df_train["id"])
    ids_test = set(df_test["id"])
    assert ids_train.isdisjoint(ids_test)


def test_misma_semilla_produce_misma_separacion():
    df = crear_df_ejemplo(30)
    train_1, test_1 = separar_datos(df, porcentaje_test=20, semilla=99)
    train_2, test_2 = separar_datos(df, porcentaje_test=20, semilla=99)

    pd.testing.assert_frame_equal(train_1, train_2)
    pd.testing.assert_frame_equal(test_1, test_2)


def test_distinta_semilla_produce_distinta_separacion():
    df = crear_df_ejemplo(30)
    train_1, _ = separar_datos(df, porcentaje_test=20, semilla=1)
    train_2, _ = separar_datos(df, porcentaje_test=20, semilla=2)

    assert not train_1.index.equals(train_2.index)


# ==========================================
# PRUEBAS DE MANEJO DE ERRORES
# ==========================================

@pytest.mark.parametrize("porcentaje_invalido", [0, -10, 100, 150, "veinte", None])
def test_error_porcentaje_fuera_de_rango(porcentaje_invalido):
    df = crear_df_ejemplo(20)
    with pytest.raises(ValueError, match="número entre 1 y 99"):
        separar_datos(df, porcentaje_test=porcentaje_invalido, semilla=42)


@pytest.mark.parametrize("semilla_invalida", [3.14, "42", None, True])
def test_error_semilla_no_entera(semilla_invalida):
    df = crear_df_ejemplo(20)
    with pytest.raises(ValueError, match="número entero"):
        separar_datos(df, porcentaje_test=20, semilla=semilla_invalida)


def test_error_dataset_menos_de_5_filas():
    df_pequeno = crear_df_ejemplo(4)
    with pytest.raises(ValueError, match="menos de 5 filas"):
        separar_datos(df_pequeno, porcentaje_test=20, semilla=42)


def test_error_conjunto_quedaria_vacio():
    # Con 5 filas y un 5% de test, round(5 * 0.05) = 0 filas en test
    df = crear_df_ejemplo(5)
    with pytest.raises(ValueError, match="quedaría vacío"):
        separar_datos(df, porcentaje_test=5, semilla=42)