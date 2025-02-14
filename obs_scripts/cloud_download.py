# -*- coding: utf-8 -*-
"""
GOAL: download mon thly ISCCP cloud data from:
https://www.ncei.noaa.gov/data/international-satellite-cloud-climate-project
-isccp-h-series-data/access/isccp/hgm/

Source info:
https://www.ncei.noaa.gov/products/international-satellite-cloud-climatology
"""


import urllib
from os import chdir, path
from bs4 import BeautifulSoup
import sys
import time


chdir('C:/Users/aakas/Documents/ENSO-Clouds/ISCCP_clouds')


def get_urls(base_url):
    """
    Returns the list of filepaths that need to be downloaded
    """
    file = urllib.request.urlopen(base_url)
    soup = BeautifulSoup(file.read(), 'html.parser')
    
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        # Filter only for the cloud files
        if href[:5] == 'ISCCP':
            links.append(href)

    return links


def progress_bar(n, max_val, cus_str=''):
    """
    I love progress bars in long loops
    """
    sys.stdout.write('\033[2K\033[1G')
    print(f'Downloading...{100 * n / max_val:.2f}% complete ' + cus_str,
          end="\r")


def download_files(links, base_url):
    """
    Downloads the nc files and saves them with simplified date formatting
    """
    num = len(links)
    for n, link in enumerate(links):
        progress_bar(n, num)
        # Download the file
        if not path.exists(link[30:37] + '.nc'):
            # DOn't redownload if aready exists
            urllib.request.urlretrieve(base_url + link, link[30:37] + '.nc')
            # Prevent excess server pings
            time.sleep(1)
    
    
def main():
    base_url = 'https://www.ncei.noaa.gov/data/international-satellite-cloud-' +\
        'climate-project-isccp-h-series-data/access/isccp-basic/hgm/'
    links = get_urls(base_url)
    
    download_files(links, base_url)


if __name__ == '__main__':
    main()
