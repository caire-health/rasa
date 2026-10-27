import sys
from importlib.metadata import distributions


def test_tensorflow_text_install():
    installed_packages = {d.metadata["Name"].lower() for d in distributions()}
    tf_text_installed = "tensorflow-text" in installed_packages

    if sys.platform == "win32":
        assert not tf_text_installed
    else:
        assert tf_text_installed
