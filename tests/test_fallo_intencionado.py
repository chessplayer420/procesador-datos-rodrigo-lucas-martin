# tests/test_fallo_intencionado.py

def test_comprobacion_fallo_ci():
    """Prueba diseñada para fallar a propósito y verificar que GitHub Actions detecta el error."""
    resultado_obtenido = 2 + 2
    resultado_esperado = 5
    assert resultado_obtenido == resultado_esperado, "Fallo intencionado para comprobar GitHub Actions"