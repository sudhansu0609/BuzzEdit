import sys

def test_python_version():
    assert sys.version_info >= (3, 11)

def test_imports():
    import fastapi
    import faster_whisper
    import onnxruntime
    import pydantic
    assert fastapi.__version__ is not None
