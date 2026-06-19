import importlib.util
import pytest


@pytest.mark.skipif(importlib.util.find_spec("casadi") is None, reason="CasADi not installed")
def test_model_builder_imports_with_casadi():
    from photosynthesis_adaptation.model import ModelBuilder
    assert ModelBuilder is not None
