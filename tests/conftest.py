import sys
from pathlib import Path

# Añade la carpeta src al sys.path para que los tests encuentren los módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))