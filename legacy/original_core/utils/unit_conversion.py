def convert_conductance(conductance, temp=298.15):
    '''
    Convert conductance from mol/m^2/s to m/s using the ideal gas law at 1014 hPa atmospheric 
    pressure and a specified temperature.

    Args:
        conductance: The conductance (mol/m^2/s)
        temp: Temperature in (K) (default: 298.15 K)

    Returns:
        conductance: The conductance in m/s
    '''

    # Constants
    P = 101400  # Pa (1014 hPa)
    R = 8.3145  # J/(mol*K)

    return conductance * R * temp / P

def convert_conductance_w_bar(conductance, temp=298.15):
    """
    Convert conductance from mol/m^2/s/bar to m/s using the ideal gas law at 1014 hPa atmospheric 
    pressure and a specified temperature.
    
    Args:
        conductance: The conductance (mol/m^2/s/bar)
        temp: Temperature in (K) (default: 298.15 K)
    
    Returns:
        conductance: The conductance in m/s
    """
    
    # Constants
    P = 101400  # Pa (1014 hPa)
    R = 8.3145  # J/(mol*K)

    conductance = conductance * 1e-5  # Convert bar to Pa

    return conductance * R * temp

def ubar_to_molar_concentration(ubar, temp=298.15):
    """
    Converts ubar to umol/m^3 at 1014 hPa atmospheric pressure and a specified temperature using 
    the ideal gas law.

    Args:
        ubar: The partial pressure of the gas (ubar)
        temp: Temperature in (K) (default: 298.15 K)

    Returns:
        molar_concentration: The concentration of the gas in umol/m^3
    """
    
    # Constants
    R = 8.3145  # J/(mol*K)
    P = 101400  # Pa (1014 hPa)

    # Convert ubar to Pa
    Pa = ubar * 0.1

    # Convert Pa to mol/m^3 using the ideal gas law
    molar_concentration = Pa / (R * temp)

    return molar_concentration * 1e6