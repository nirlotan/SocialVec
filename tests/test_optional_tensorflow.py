"""TensorFlow is optional: only SocialVecClassifier needs it."""

import subprocess
import sys

import pytest

from socialvec import socialvec as sv
from socialvec.exceptions import SocialVecError


def test_module_imports_without_tensorflow():
    # sys.modules[name] = None makes "import name" fail, as if it were not installed.
    code = ("import sys; sys.modules['tensorflow'] = None; "
            "import socialvec.socialvec; assert 'tensorflow' not in sys.modules or sys.modules['tensorflow'] is None")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_classifier_without_tensorflow_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    with pytest.raises(SocialVecError, match=r"socialvec\[classifier\]"):
        sv.SocialVec.SocialVecClassifier("2020c")
