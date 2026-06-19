import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import utils.data_api as api
import utils.unit_conversion as uc

def load_parameters(biochemical_params='default', 
                    environmental_params='default',
                    temp_dependence_params='default',
                    soil_params='loamy_sand',
                    hydraulic_params='default',
                    heat_params='default'):
    
    params = {}

    params.update(_load_biochemical_parameters(biochemical_params))
    params.update(_load_environmental_parameters(environmental_params))
    params.update(_load_temp_dependence_parameters(temp_dependence_params))
    params.update(_load_soil_parameters(soil_params))
    params.update(_load_hydraulic_parameters(hydraulic_params))
    params.update(_load_heat_parameters(heat_params))

    return params

def _load_biochemical_parameters(param_set='default'):
    with open('resources/biochemical_parameters.json', 'r') as file:
        param_list = json.load(file)

    # Convert list of dicts into a dict keyed by 'param_set'
    biochemical_params = {
        p["param_set"]: {k: v for k, v in p.items() if k != "param_set"} for p in param_list
    }

    if param_set in biochemical_params:
        params = _convert_units(biochemical_params[param_set])
        params['r_mes'] = _calculate_dark_respiration(params['v_cmax_mes'])
        params['r_bs'] = _calculate_dark_respiration(params['v_cmax_bs'])
        params['r_d'] = params['r_mes'] + params['r_bs']

        return params
    else:
        raise ValueError(f"Parameter set '{param_set}' not found in biochemical parameters.")

def store_api_parameters(set_name, longitude, latitude, date, time_start=5):
    """
    Store API parameters for a given latitude, longitude, and date.

    Args:
        latitude (float): Geographic latitude in decimal degrees.
        longitude (float): Geographic longitude in decimal degrees.
        date (str): The date in 'YYYYMMDD' format.

    """

    start_date, end_date = _calculate_time_frame(date)

    print(f"Fetching data from Power API for {longitude}, {latitude} from {start_date} to {end_date}")

    param_df = api.get_data_from_power_api(
        longitude=longitude,
        latitude=latitude,
        start_date=start_date,
        end_date=end_date,
        export_to_csv=False)
    
    a_air, b_air = _calculate_temp_parameters(param_df, time_start)
    
    if param_df is not None:
        # Extract year, month, day from the input date
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])

        # Filter the row for the given date at 10:00
        row = param_df[
            (param_df['year'] == year) &
            (param_df['month'] == month) &
            (param_df['day'] == day) &
            (param_df['hour'] == time_start + 5)
        ]

        if not row.empty:
            with open('resources/environmental_parameters.json', 'r') as f:
                json_data = json.load(f)
                param_dict = {
                    "param_set": set_name,
                    "solar_irradiance": row['solar_radiation'].iloc[0],
                    "a_air": a_air,
                    "b_air": b_air,
                    "rel_humidity": row['rel_humidity'].iloc[0],
                    "rel_soil_wetness": row['rel_soil_wetness'].iloc[0],
                    "co2_atm": 405,
                    "o2_atm": 210e3
                }
            
            json_data.append(param_dict)

            with open('resources/environmental_parameters.json', 'w') as f:
                json.dump(json_data, f, indent=4)

        else:
            raise ValueError(f"No data found for {year}-{month:02d}-{day:02d} at 10:00")

def _calculate_time_frame(date):
    """
    Calculate the end date for the API request based on the given date.
    
    Args:
        date (str): The start date in 'YYYYMMDD' format.
    Returns:
        str: The end date in 'YYYYMMDD' format, which is the same as the start date.
    """

    # Add 7 days to the start date
    date = np.datetime64(date[:4] + '-' + date[4:6] + '-' + date[6:], 'D')
    start_date = date - np.timedelta64(7, 'D')  
    end_date = date + np.timedelta64(7, 'D')

    start = start_date.astype('datetime64[D]').item().strftime('%Y%m%d')
    end = end_date.astype('datetime64[D]').item().strftime('%Y%m%d')
    
    # Convert back to 'YYYYMMDD' format
    return start, end

def _calculate_temp_parameters(params, time_start):
    grouped_df = (
        params.groupby('hour')
        .agg(
            solar_radiation_mean=('solar_radiation', 'mean'),
            solar_radiation_std=('solar_radiation', 'std'),
            temp_mean=('temp', 'mean'),
            temp_std=('temp', 'std')
        )
        .reset_index()
    )

    filtered_grouped = grouped_df[
        (grouped_df['hour'] >= time_start) &
        (grouped_df['hour'] <= time_start + 5)
    ]

    # Prepare data for regression
    X = filtered_grouped['solar_radiation_mean'].values.reshape(-1, 1)
    y = filtered_grouped['temp_mean'].values
    sigma_y = filtered_grouped['temp_std'].values  # standard deviation of temp

    # Fit linear regression on means
    model = LinearRegression()
    model.fit(X, y)

    # Prediction line
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    # R²
    r_squared = model.score(X, y)

    # Show plot
    _show_plot(grouped_df, X, y, sigma_y, x_line, y_line, r_squared, time_start)

    # return slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept

def _show_plot(grouped_df, X, y, sigma_y, x_line, y_line, r_squared, time_start):
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 4))

    # --- First subplot: Temperature & Radiation vs Hour ---
    ax1.set_xlabel('Hour of the Day')
    ax1.set_ylabel('Temperature (°C)', color='tab:red')
    ax1.plot(grouped_df['hour'], grouped_df['temp_mean'] - 273.15,
             color='tab:red', label='Temperature')
    ax1.axvline(x=time_start, color='gray', linestyle='--', linewidth=1)
    ax1.axvline(x=time_start + 5, color='gray', linestyle='--', linewidth=1)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Twin y-axis for radiation
    ax1_radiation = ax1.twinx()
    ax1_radiation.set_ylabel('PAR (W m$^{-2}$)', color='tab:blue')
    ax1_radiation.plot(grouped_df['hour'], grouped_df['solar_radiation_mean'],
                       color='tab:blue', label='Radiation')
    ax1_radiation.tick_params(axis='y', labelcolor='tab:blue')

    # --- Second subplot: Scatter + Linear Fit ---
    ax3.errorbar(X.flatten(), y - 273.15, yerr=sigma_y, fmt='o',
                 capsize=4, label='Hourly Means ± Std', color='black', ecolor='black', 
                 elinewidth=0.5)
    ax3.plot(x_line, y_line - 273.15, color='red', linewidth=1, label='Linear Fit')

    ax3.set_xlabel('PAR (W m$^{-2}$)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.legend()
    ax3.grid(True)

    # Add R² in the second subplot
    ax3.text(0.05, 0.95,
             f"$R^2$ = {r_squared:.4f}",
             transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, linewidth=0.5))

    fig.tight_layout()

    plt.show()
    
def _convert_units(params):
    '''
    Convert biochemical parameters to appropriate units.
    
    Args:
        params (dict): A dictionary containing biochemical parameters.
    
    Returns:
        dict: A dictionary with converted biochemical parameters.
    '''
    
    params['k_p'] = uc.ubar_to_molar_concentration(params['k_p']) 
    params['k_o'] = uc.ubar_to_molar_concentration(params['k_o'])
    params['k_c'] = uc.ubar_to_molar_concentration(params['k_c'])

    return params

def _calculate_dark_respiration(v_cmax):
    '''
    Calculate the dark respiration rate using the equation from von Caemmerer (2021).
    
    Args:
        v_cmax: Maximum rate of Rubisco (umol/m^2/s)
    
    Returns:
        r: Dark respiration rate (umol/m^2/s)
    '''

    r = 0.01 * v_cmax

    return r

def _load_environmental_parameters(param_set='default'):
    '''
    Load environmental parameters from a JSON file and convert them to appropriate units.
    The stored parameters include:

        - 'solar_irradiance'
        - 'temp'
        - 'temp_sky'
        - 'co2_atm'
        - 'o2_atm'
        - 'rel_humidity'
        - 'rel_soil_wetness'

    Args:
        param_set (str): The name of the parameter set to load. Default is 'default'.
    
    Returns:
        dict: A dictionary containing the environmental parameters with values in appropriate units.
    '''

    with open('resources/environmental_parameters.json', 'r') as f:
        param_list = json.load(f)

    # Convert list of dicts into a dict keyed by 'param_set'
    environmental_params = {
        p["param_set"]: {k: v for k, v in p.items() if k != "param_set"} for p in param_list
    }

    if param_set in environmental_params:
        params = environmental_params[param_set]

        return params
    else:
        raise ValueError(f"Parameter set '{param_set}' not found in environmental parameters.")

def _load_temp_dependence_parameters(param_set='default'):
    '''
    Load temperature dependence parameters from a JSON file.
    The stored parameters include:

        - 'v_pmax_ea'
        - 'v_pmax_hd'
        - 'v_pmax_delta_s'
        - 'v_cmax_ea'
        - 'v_cmax_hd'
        - 'v_cmax_delta_s'
        - 'k_c_ea'
        - 'k_o_ea'
        - 'k_p_ea'
        - 'r_ea'
        - 'gamma_star_ea'
        - 'a_o_ea'

    Args:
        param_set (str): The name of the parameter set to load. Default is 'default'.
    
    Returns:
        dict: A dictionary containing the temperature dependence parameters.
    '''

    with open('resources/temp_dependence_parameters.json', 'r') as f:
        param_list = json.load(f)

    # Convert list of dicts into a dict keyed by 'param_set'
    temp_dependence_params = {
        p["param_set"]: {k: v for k, v in p.items() if k != "param_set"} for p in param_list
    }

    if param_set in temp_dependence_params:
        return temp_dependence_params[param_set]
    else:
        raise ValueError(f"Parameter set '{param_set}' not found in temperature dependence \
                         parameters.")
    

def _load_soil_parameters(soil_type='loamy_sand'):
    '''
    Load soil parameters from a JSON file based on the soil type.
    The stored parameters include:

        - 'psi_sat'
        - 'b'
        - 'k_sat'

    Args:
        soil_type (str): The type of soil (e.g., 'sand', 'loamy_sand', 'sandy_loam', etc.)

    Returns:
        dict: A dictionary containing the soil parameters for the specified soil type.
    '''

    with open('resources/soil_parameters.json', 'r') as f:
        param_list = json.load(f)
    
    # Convert list of dicts into a dict keyed by 'soil_type'
    soil_params = {
        p["soil_type"]: {k: v for k, v in p.items() if k != "soil_type"} for p in param_list
    }

    if soil_type in soil_params:
        params = soil_params[soil_type]

        # Convert soil parameters to appropriate units
        params['psi_sat'] = params['psi_sat'] * 98.0665  # Convert from cm H2O to Pa
        
        return params
    else:
        raise ValueError(f"Soil type '{soil_type}' not found in soil parameters. Available types: \
                         {', '.join(soil_params.keys())}")


def _load_hydraulic_parameters(param_set='default'):
    '''
    Load hydraulic parameters from a JSON file.
    The stored parameters include:

        - 'leaf_water_potential'
        - 'k_l'

    Args:
        param_set (str): The name of the parameter set to load. Default is 'default'.
    
    Returns:
        dict: A dictionary containing the hydraulic parameters with values in appropriate units.
    '''

    with open('resources/hydraulic_parameters.json', 'r') as f:
        param_list = json.load(f)

    # Convert list of dicts into a dict keyed by 'param_set'
    hydraulic_params = {
        p["param_set"]: {k: v for k, v in p.items() if k != "param_set"} for p in param_list
    }

    if param_set in hydraulic_params:
        params = hydraulic_params[param_set]

        return params
    else:
        raise ValueError(f"Parameter set '{param_set}' not found in hydraulic parameters.")


def _load_heat_parameters(param_set='default'):
    '''
    Load heat parameters from a JSON file.
    The stored parameters include:

        - 'u_a'
        - 'g_h'
        - 'c_p'
        - 'lambda_heat'
        - 'temp_sink'
        - 'k'

    Args:
        param_set (str): The name of the parameter set to load. Default is 'default'.
    
    Returns:
        dict: A dictionary containing the heat parameters with values in appropriate units.
    '''

    with open('resources/heat_parameters.json', 'r') as f:
        param_list = json.load(f)

    # Convert list of dicts into a dict keyed by 'param_set'
    heat_params = {
        p["param_set"]: {k: v for k, v in p.items() if k != "param_set"} for p in param_list
    }

    if param_set in heat_params:
        return heat_params[param_set]
    else:
        raise ValueError(f"Parameter set '{param_set}' not found in heat parameters.")