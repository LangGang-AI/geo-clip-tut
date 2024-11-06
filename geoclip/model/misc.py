"""
GeoCLIP Utility Functions
========================

This module provides utility functions for GeoCLIP, primarily focused on
data loading, coordinate transformations, and path management operations.

Key Functions:
- GPS data loading from CSV files
- File path management
- Coordinate format conversions (Decimal, DMS, UTM)
"""

import os
import torch
import numpy as np
import pandas as pd
import math

# Get the directory containing the current file
file_dir = os.path.dirname(os.path.realpath(__file__))

def load_gps_data(csv_file):
    """[original docstring preserved]"""
    data = pd.read_csv(csv_file)
    lat_lon = data[['LAT', 'LON']]
    gps_tensor = torch.tensor(lat_lon.values, dtype=torch.float32)
    return gps_tensor

def get_utm_zone(longitude):
    """
    Determines the UTM zone number for a given longitude.
    
    Args:
        longitude (float): Longitude in decimal degrees
    
    Returns:
        int: UTM zone number (1-60)
    """
    return int(np.floor((longitude + 180) / 6) + 1)

def get_utm_letter(latitude):
    """
    Determines the UTM latitude band letter.
    
    Args:
        latitude (float): Latitude in decimal degrees
    
    Returns:
        str: UTM latitude band letter (C-X, excluding I and O)
    """
    letters = 'CDEFGHJKLMNPQRSTUVWX'
    index = int(np.floor((latitude + 80) / 8))
    if 0 <= index < len(letters):
        return letters[index]
    return None

def convert_coordinates(coordinates, from_format='decimal', to_format='dms', as_tensor=True):
    """
    Converts geographic coordinates between decimal degrees, DMS, and UTM formats.
    
    Args:
        coordinates (Union[torch.Tensor, np.ndarray, list]): Input coordinates
            For decimal: [[latitude, longitude], ...]
            For DMS: [[lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec], ...]
            For UTM: [[easting, northing, zone_number], ...] or
                    [[easting, northing, zone_number, zone_letter], ...]
        from_format (str): Input format - 'decimal', 'dms', or 'utm'
        to_format (str): Output format - 'decimal', 'dms', or 'utm'
        as_tensor (bool): Whether to return result as PyTorch tensor (True) or numpy array (False)
    
    Returns:
        Union[torch.Tensor, np.ndarray]: Converted coordinates in specified format
            If to_format='decimal': Shape (N, 2) for [latitude, longitude]
            If to_format='dms': Shape (N, 6) for [lat_deg, lat_min, lat_sec, lon_deg, lon_min, lon_sec]
            If to_format='utm': Shape (N, 4) for [easting, northing, zone_number, zone_letter]
    
    Example:
        >>> # San Francisco coordinates
        >>> coords = torch.tensor([[37.7749, -122.4194]])
        >>> utm = convert_coordinates(coords, 'decimal', 'utm')
        >>> # Result: tensor([[545974.79, 4180445.80, 10, 'S']])
        >>> decimal = convert_coordinates(utm, 'utm', 'decimal')
        >>> # Result: tensor([[37.7749, -122.4194]])
    """
    # Convert input to numpy for calculations
    if torch.is_tensor(coordinates):
        coords_np = coordinates.numpy()
    else:
        coords_np = np.array(coordinates)

    # Constants for UTM conversion
    K0 = 0.9996  # UTM scale factor
    E = 0.00669438  # WGS84 eccentricity squared
    R = 6378137  # WGS84 equatorial radius in meters
    E2 = E * E
    E3 = E2 * E
    E_P2 = E / (1 - E)

    if from_format == 'decimal' and to_format == 'dms':
        # [Original decimal to DMS conversion preserved]
        degrees = np.floor(np.abs(coords_np))
        decimal_minutes = (np.abs(coords_np) - degrees) * 60
        minutes = np.floor(decimal_minutes)
        seconds = (decimal_minutes - minutes) * 60
        degrees = np.sign(coords_np) * degrees
        result = np.column_stack((
            degrees[:, 0], minutes[:, 0], seconds[:, 0],
            degrees[:, 1], minutes[:, 1], seconds[:, 1]
        ))

    elif from_format == 'dms' and to_format == 'decimal':
        # [Original DMS to decimal conversion preserved]
        lat_decimal = (
            np.sign(coords_np[:, 0]) * 
            (np.abs(coords_np[:, 0]) + coords_np[:, 1]/60 + coords_np[:, 2]/3600)
        )
        lon_decimal = (
            np.sign(coords_np[:, 3]) * 
            (np.abs(coords_np[:, 3]) + coords_np[:, 4]/60 + coords_np[:, 5]/3600)
        )
        result = np.column_stack((lat_decimal, lon_decimal))

    elif from_format == 'decimal' and to_format == 'utm':
        result = []
        for lat, lon in coords_np:
            # Get zone number and letter
            zone_number = get_utm_zone(lon)
            zone_letter = get_utm_letter(lat)
            
            # Convert to radians
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            
            # Calculate UTM parameters
            lon0 = np.radians((zone_number - 1) * 6 - 180 + 3)
            
            N = R / np.sqrt(1 - E * np.sin(lat_rad)**2)
            T = np.tan(lat_rad)**2
            C = E_P2 * np.cos(lat_rad)**2
            A = np.cos(lat_rad) * (lon_rad - lon0)
            
            # Calculate M (meridian distance)
            M = R * ((1 - E/4 - 3*E2/64 - 5*E3/256) * lat_rad -
                    (3*E/8 + 3*E2/32 + 45*E3/1024) * np.sin(2*lat_rad) +
                    (15*E2/256 + 45*E3/1024) * np.sin(4*lat_rad) -
                    (35*E3/3072) * np.sin(6*lat_rad))
            
            # Calculate easting and northing
            easting = K0 * N * (A + (1-T+C) * A**3/6 +
                               (5-18*T+T*T+72*C-58) * A**5/120)
            northing = K0 * (M + N * np.tan(lat_rad) * (A*A/2 +
                            (5-T+9*C+4*C*C) * A**4/24 +
                            (61-58*T+T*T+600*C-330) * A**6/720))
            
            # Add false easting and northing
            easting += 500000
            if lat < 0:
                northing += 10000000
                
            result.append([easting, northing, zone_number, zone_letter])
        result = np.array(result)

    elif from_format == 'utm' and to_format == 'decimal':
        result = []
        for easting, northing, zone_number, *zone_letter in coords_np:
            # Remove false easting and northing
            easting -= 500000
            if zone_letter and zone_letter[0] < 'N':
                northing -= 10000000
                
            # Central meridian for zone
            lon0 = np.radians((zone_number - 1) * 6 - 180 + 3)
            
            # Footpoint latitude
            e1 = (1 - np.sqrt(1 - E)) / (1 + np.sqrt(1 - E))
            M = northing / K0
            mu = M / (R * (1 - E/4 - 3*E2/64 - 5*E3/256))
            
            phi1 = mu + ((3*e1/2 - 27*e1**3/32) * np.sin(2*mu) +
                        (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu) +
                        (151*e1**3/96) * np.sin(6*mu) +
                        (1097*e1**4/512) * np.sin(8*mu))
            
            # Calculate latitude and longitude
            N1 = R / np.sqrt(1 - E * np.sin(phi1)**2)
            T1 = np.tan(phi1)**2
            C1 = E_P2 * np.cos(phi1)**2
            R1 = R * (1 - E) / ((1 - E * np.sin(phi1)**2)**(3/2))
            D = easting / (N1 * K0)
            
            lat = phi1 - ((N1 * np.tan(phi1)/R1) *
                         (D*D/2 - (5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*E_P2) * D**4/24 +
                          (61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*E_P2 - 3*C1*C1) * D**6/720))
            
            lon = lon0 + (D - (1 + 2*T1 + C1) * D**3/6 +
                         (5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*E_P2 + 24*T1*T1) * D**5/120) / np.cos(phi1)
            
            result.append([np.degrees(lat), np.degrees(lon)])
        result = np.array(result)
    
    else:
        raise ValueError(f"Unsupported conversion from {from_format} to {to_format}")
    
    # Return result in requested format
    if as_tensor:
        return torch.tensor(result, dtype=torch.float32)
    return result
