import numpy as np
import pytest
import requests
from photosynthesis_adaptation.utils.data_api import get_data_from_power_api


def test_power_missing_values_stay_missing_and_provenance_is_retained(monkeypatch):
    def get(url, params, timeout):
        assert timeout == (10, 60)
        assert params['time-standard'] == 'utc'
        response = requests.Response()
        response.status_code = 200
        response._content = b'YEAR,MO,DY,HR,ALLSKY_SFC_PAR_TOT,T2M,RH2M,GWETROOT\n2020,1,1,0,0,-999,70,0.5\n2020,1,1,1,10,20,71,0.6\n'
        return response
    monkeypatch.setattr(requests, 'get', get)
    data = get_data_from_power_api(10,50,'20200101','20200101')
    assert np.isnan(data.temp.iloc[0])
    assert data.temp.iloc[1] == pytest.approx(293.15)
    assert len(data.attrs['provenance']['response_sha256']) == 64


def test_invalid_location_and_date_are_rejected_before_network():
    with pytest.raises(ValueError):
        get_data_from_power_api(0,91,'20200101','20200102')
    with pytest.raises(ValueError):
        get_data_from_power_api(0,0,'20200102','20200101')
