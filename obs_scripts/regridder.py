# -*- coding: utf-8 -*-
"""
Regridder intended to be run on WSL

I work on windows so i need this workaround so the CMIP6 data can fit 
into my existing workflow. Currently only supports equal res lat/lon grids
but that should be easy to modify if necessary.

Also merges the files together for convenience
"""

from os import chdir, listdir
import xarray as xr
import xesmf as xe

# Directory of the project
chdir('/mnt/c/Users/aakas/Documents/ENSO-Clouds/')


def load_data(input_folder):
    """
    Loads data from folder and resturls merged dataset with var
    """
    print('Loading raw files...')
    files = listdir(input_folder)
    loaded = [None for _ in files]
    for n, file in enumerate(files):
        loaded[n] = xr.load_dataset(input_folder + file)
    merged = xr.merge(loaded)
    return merged


def rename_coords(merged):
    """
    Gets user input for whether coordinates need to be merged or dropped
    """
    coords = list(merged.coords)
    # Pretty print
    coord_string = ''
    num_coords = len(coords)
    
    for n, coord in enumerate(coords):
        if n == num_coords - 1:
            coord_string += 'and ' + coord + '.'
        else:   
            coord_string += coord + ', '
    
    rename = input('Available coordinates are ' + coord_string +\
                   ' Rename (yes/no)? ')
    print('To delete a variable, enter del. To retain the old name, enter no.')
    print('The goal is to rename coords to lat/lon so xesmf can interpret it.')
    if rename == 'yes':
        rename_dict = {coord: coord for coord in coords}
        for coord in coords:
            new_name = input('Rename ' + coord + ' to: ')
            if new_name == 'no':
                continue
            elif new_name == 'del':
                del rename_dict[coord]
            else:
                rename_dict[coord] = new_name
    
        merged = merged.rename(rename_dict)
        return merged            
    else:
        return merged


def main():
    input_folder = input('Enter directory with native grid files (ex. CMIP6/IPSL-CM6A/sst_raw/): ').strip()
    output_folder = input('Enter directory to write to (ex. CMIP6/IPSL-CM6A/sst_regrid/): ').strip()
    name = input('Enter output file name: ').strip()
    res = float(input('Enter target resolution (ex. 0.25): ').strip())
    var = input('Enter variable name (ex. tos): ').strip()
    
    merged = load_data(input_folder)
    merged = rename_coords(merged)
    target_grid = xe.util.grid_global(res, res)
    source_grid = {'lon': merged['lon'], 'lat': merged['lat']}

    regridder = xe.Regridder(source_grid, target_grid, "bilinear",
                             ignore_degenerate=True)
    print('Interpolating...')
    # Clean and put into standard format
    new_data = regridder(merged[var])
    new_data = xr.Dataset({var: new_data})
    new_data = new_data.rename({'y': 'lat', 'x': 'lon'})
    new_data['lon'] = new_data.lon[0, :].to_numpy()
    new_data['lat'] = new_data.lat[:, 0].to_numpy()
    new_data = new_data.assign_coords({'lat': new_data['lat'],
                                       'lon': new_data['lon']})

    # print(new_data)
    print('Saving file...')
    new_data.to_netcdf(output_folder + name)


if __name__ == '__main__':
    main()
