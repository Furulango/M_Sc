packages = {
    "scikit-learn": "sklearn",
    "TensorFlow": "tensorflow",
    "Seaborn": "seaborn"
}

for name, module in packages.items():
    try:
        pkg = __import__(module)
        print(f"{name} está instalado. Versión: {pkg.__version__}")
    except ImportError:
        print(f"{name} NO está instalado. Instálalo con: pip install {module}")
