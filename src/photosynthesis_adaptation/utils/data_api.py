import hashlib
import io
from datetime import datetime

import numpy as np
import pandas as pd
import requests


def get_data_from_power_api(
    longitude: float, latitude: float, start_date: str, end_date: str, export_to_csv: bool = False
):
    """
    Fetch hourly environmental data from NASA's POWER API for a specific dates and location.

    Args:
        latitude (float): Geographic latitude in decimal degrees.
        longitude (float): Geographic longitude in decimal degrees.
        start_date (str): The date in 'YYYYMMDD' format for which to retrieve the data.
        end_date (str): The end date in 'YYYYMMDD' format for which to retrieve the data.
        csv_file (bool): If True, saves the CSV to 'power_data.csv'. Default is False.

    Returns:
        pandas.DataFrame:
            A DataFrame containing the environmental data if successful, raises an exception on failure.
    """

    if (
        not np.isfinite(latitude)
        or not -90 <= latitude <= 90
        or not np.isfinite(longitude)
        or not -180 <= longitude <= 180
    ):
        raise ValueError("Invalid latitude/longitude")
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    if end < start:
        raise ValueError("end_date precedes start_date")
    # Define the API endpoint and parameters
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "start": start_date,
        "end": end_date,
        "latitude": latitude,
        "longitude": longitude,
        "community": "sb",
        "parameters": "ALLSKY_SFC_PAR_TOT,T2M,RH2M,GWETROOT",
        "format": "csv",
        "header": "false",
        "time-standard": "utc",
    }

    df = _pull_data(url, params, latitude, longitude, start_date, export_to_csv)

    if df is None:
        return None

    df = _rename_columns(df)
    convert_units(df)

    return df


def _pull_data(
    url: str,
    params: dict,
    latitude: float,
    longitude: float,
    start_date: str,
    export_to_csv: bool = False,
):
    """
    Helper function to pull data from the POWER API via GET request.

    Args:
        url (str): The API endpoint URL.
        params (dict): The parameters for the API request.
        latitude (float): Geographic latitude in decimal degrees.
        longitude (float): Geographic longitude in decimal degrees.
        start_date (str): The start date in 'YYYYMMDD' format.
        csv_file (bool): If True, saves the CSV to a file.

    Returns:
        pandas.DataFrame:
            A DataFrame containing the environmental data if successful, raises an exception on failure.
    """

    response = requests.get(url, params=params, timeout=(10, 60))
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), na_values=[-999, -999.0])
    required = {"YEAR", "MO", "DY", "HR", "ALLSKY_SFC_PAR_TOT", "T2M", "RH2M", "GWETROOT"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("POWER response lacks required hourly columns or rows")
    df.attrs["provenance"] = {
        "url": url,
        "parameters": dict(params),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "time_standard": "UTC",
        "missing_value": -999,
    }
    if export_to_csv:
        from pathlib import Path

        Path(f"power_data_{latitude:.3f}_{longitude:.3f}_{start_date}.csv").write_bytes(
            response.content
        )
    return df


def _rename_columns(df):
    """
    Rename columns in the DataFrame to more user-friendly names.

    Args:
        df (pandas.DataFrame): The DataFrame containing the environmental data.

    Returns:
        pandas.DataFrame: The DataFrame with renamed columns.
    """
    column_mapping = {
        "YEAR": "year",
        "MO": "month",
        "DY": "day",
        "HR": "hour",
        "ALLSKY_SFC_PAR_TOT": "solar_radiation",
        "T2M": "temp",
        "RH2M": "rel_humidity",
        "GWETROOT": "rel_soil_wetness",
    }

    return df.rename(columns=column_mapping)


def convert_units(df):
    """Convert temperature from °C to K."""
    df["temp"] = df["temp"] + 273.15
    df.attrs["temperature_unit"] = "K"
    df.attrs["radiation_note"] = (
        "NASA POWER PAR field; do not assume equivalence to broadband irradiance"
    )
