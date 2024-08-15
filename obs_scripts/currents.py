# -*- coding: utf-8 -*-
"""
Ocean currents data (eastward flow and upwelling) from data

https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0
"""


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import requests
from os import chdir, remove
import cartopy.crs as ccrs
import threading
from tqdm import tqdm


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


def process_file(fpath, domain=cz_domain):
    """
    Opens the downloaded OSCAR file, truncates it to the appropriate domain
    and returns the truncated file
    """
    min_lat, max_lat, min_lon, max_lon = domain
    data = xr.open_dataset(fpath)
    
    # Set index
    data = data.set_index(latitude='lat')
    data = data.set_index(longitude='lon')
    
    # Since we know the day already
    data = data.drop_vars('time')
    data['u'] = data.u.squeeze()
    data['v'] = data.v.squeeze()
    
    data = data.sel(latitude=slice(min_lat, max_lat), 
                    longitude=slice(min_lon, max_lon))
    
    return data


def plot_waves(era5, every=20):
    """
    Plots wind barbs for climatological winds across the equatorial
    Pacific from ERA5 reanalysis.
    
    Plots every n'th observation as specified by "every".
    Assumes the data is already sliced to a single time step.
    """
    plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    # Remove the time dimension if it's present
    u = np.asarray(era5.u)
    v = np.asarray(era5.v)
    
    # Create meshgrid for lon and lat
    lon2d, lat2d = np.meshgrid(era5.longitude, era5.latitude)
    
    # Subset the 2D arrays
    lon2d_subset = lon2d[::every, ::every].T
    lat2d_subset = lat2d[::every, ::every].T
    u_subset = u[::every, ::every]
    v_subset = v[::every, ::every]
       
    # Calculate wind magnitude
    mag = np.hypot(u_subset, v_subset)
    
    ax.set_title('Equatorial Pacific Climatological Currents')
    con = ax.quiver(lon2d_subset, lat2d_subset, u_subset, v_subset,
                    mag, cmap='viridis', transform=ccrs.PlateCarree())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.015, pad=0.04)


def threaded_download(headers, address_list, name_list, pbar):
    """
    Threaded process for returning dataarrays that have been composited already
    """
    global composite, lock, pbar_lock
    
    data = None
    
    for link, name in zip(address_list, name_list):
        response = requests.get(link, headers=headers)
        
        if response.status_code == 200:
            with open(name, 'wb') as f:
                f.write(response.content)
                
            if data is None:
                data = process_file(name)
                data.close()
            else:
                dataset = process_file(name)
                data += dataset
                dataset.close()            
        else:
            print(f"Failed to download {link}. Status code: {response.status_code}")
            continue 
        
        # Update the progress bar
        with pbar_lock:
            pbar.update(1)
    
    # Combine the downloaded data into the composite
    with lock:
        if composite is None:
            composite = data
        else:
            composite += data


def main():
    links, names = files_to_download()

    # Your generated Earthdata token
    with open('obs_scripts/earthdata_key.txt', 'r') as file:
        token = file.read().rstrip()

    # Authorization headers
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    global composite, lock, pbar_lock
    composite = None
    lock = threading.Lock()
    pbar_lock = threading.Lock()

    nthreads = 25
    numfiles = len(links)
    
    n_per_thread = numfiles // nthreads
    extra = numfiles % nthreads
    
    threads = []
    
    # Initialize the progress bar
    with tqdm(total=numfiles, desc="Downloading and Processing Files") as pbar:
        for n in range(nthreads):
            start_idx = n_per_thread * n
            end_idx = n_per_thread * (n + 1) if n < nthreads - 1 else\
                n_per_thread * (n + 1) + extra
            
            sub_links = links[start_idx:end_idx]
            sub_names = names[start_idx:end_idx]
            
            thread = threading.Thread(target=threaded_download, 
                                      args=(headers, sub_links,
                                            sub_names, pbar))
            threads.append(thread)
            thread.start()  # Start the thread
        
        for thread in threads:
            thread.join()

    # Delete all thee downloaded files
    for name in names:
       remove(name)

    composite /= numfiles
    plot_waves(composite)  
    
    # Statistics
    print(f'Mean zonal velocity: {composite.u.mean():.3f} m/s')
    print(f'Mean meriodional velocity: {composite.v.mean():.3f} m/s')    

    # save file
    # composite.to_netcdf('misc_data/OSCAR_composite.nc')
    # We can now grab this data from the saved file easily!

if __name__ == "__main__":
    main()

       
