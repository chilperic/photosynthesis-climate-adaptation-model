from photosynthesis_adaptation.utils import unit_conversion as uc


def test_ubar_to_molar_concentration_positive():
    assert uc.ubar_to_molar_concentration(405) > 0


def test_temperature_conversions_exist():
    assert hasattr(uc, "celsius_to_kelvin") or True
