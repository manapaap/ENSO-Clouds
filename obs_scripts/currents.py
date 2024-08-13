# -*- coding: utf-8 -*-
"""
Ocean currents data (eastward flow and upwelling) from data

https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import urllib
import requests
from getpass import getpass
from os import chdir


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
cz_domain = [-30, 30, 120, -80 + 360]


def url_template(year, month, day):
    """
    Uses the url template to return OSCAR surface current netcdf file urls
    """
    temp = 'https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-' +\
        'protected/OSCAR_L4_OC_FINAL_V2.0/oscar_currents_final_'

    return temp + year + month + day + '.nc'


def files_to_download():
    """
    Returns the list of OSCAR files needed to be downloaded

    to be queried one by one later and compiled into the climatological mean
    file
    """
    start_year = np.datetime64('1993-01-01')
    end_year = np.datetime64('2021-01-01')
    dates = np.arange(start_year, end_year, np.timedelta64(1,'D'))
    years = (dates.astype('datetime64[Y]').astype(int) + 1970).astype('str')
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1
    months = np.char.zfill(months.astype('str'), 2)
    days = dates - dates.astype('datetime64[M]') + 1
    # days needs to be reformatted due to numpy weirdness
    days = np.char.zfill(days.astype('int').astype('str'), 2)   
    
    links = [url_template(y, m, d) for y, m, d in zip(years, months, days)]
    names = ['obs_scripts/OSCAR_temp/' + y + m + d + '_OSCAR.nc' for\
             y, m, d in zip(years, months, days)]
    
    return links, names


def process_file(name):
    """
    Opens the downloaded OSCAR file, truncates it to the appropriate domain
    and returns the truncated file
    """
    
    


def main():
    links, names = files_to_download()

    # Your generated Earthdata token (Thanks chatgpt)
    token = ".eyJ0eXBlIjoiVXNlciIsInVpZCI6InNreV9zY2llbnRpc3QiLCJleHAiOjE3Mjg3NzEyNDYsImlhdCI6MTcyMzU4NzI0NiwiaXNzIjoiRWFydGhkYXRhIExvZ2luIn0.f9tZaBtp5-BxuJUaymc7wfPDPmBh5h31ppagKwfWdosHB3dxLHJvzwuRh00hfjAlPjtfEik10nQ9L70gHnhUKDs2KkKLgwtn9ukYYqE_VukuUw_XiIXQWXgl0wso4rJB9wDtATJijjKfa2M1B-Z3VL6lkGVPI1gyoT41c3BEFL0Z9VAL_1W0Rf2Wx8a-uKFtAaSBALAQwkiyYJP3f1SnORGJxX0FnqOX8Zc8deh4LqSLiTRlt9IhvO8qgio_zQzTPiU77macX74NlSB2GxoUsGlrV8tc1czF0SLarT7Cik7sdMgFnVt0MPmQ85-SvcMuZKfy0tmAI-k62RE-aJnDoA"

    # Authorization headers
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    for link, name in zip(links, names):
        response = requests.get(link, headers=headers)
        
        if response.status_code == 200:
            with open(name, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded: {name[23:31]}")
            
        else:
            print(f"Failed to download {link}. Status code: {response.status_code}")
            break


if __name__ == "__main__":
    main()

       
