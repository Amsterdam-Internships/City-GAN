"""
Download panoramic images in a bbox region using multithreading.
"""

import requests # pip3 install requests
import json
import os
import urllib.request
import time
from multiprocessing.pool import ThreadPool
import socket

socket.setdefaulttimeout(100)

BASE_DIR = "panoramic_images"
NUM_WORKERS = 5
BBOX_LIST = [[125000.00,490364.05,131000.00,487653.58],[109515.54,494458.21,113411.56,490364.05], [109500.00,483000.00,118000.00,478000.00], [118000.00,483000.00,118500.00,478000.00], [118000.00,483000.00,119500.00,478000.00]]
API_TAGS = "mission-2019,surface-land"

def get_pano_data(api_url, output_folder, session):
    """
    Request data from API and download panoramic image
    Return the url of the next API page, null otherwise
    """
    with session.get(api_url) as response:
        pano_data_all = json.loads(response.content)

    pano_data = pano_data_all["_embedded"]["panoramas"]

    # Append the panorama id's to the list
    for item in pano_data:
        image_url = item["_links"]["equirectangular_small"]["href"]
        try:
            urllib.request.urlretrieve(image_url, os.path.join(output_folder, item['pano_id'] + ".jpg"))
        except Exception as e:
            print("Error with image URL: " + image_url)
            print(e)

    # Check for next page
    next_page = pano_data_all["_links"]["next"]["href"]

    return next_page

def download_image(bbox):
    # Create directory with bbox name
    bbox_name = ",".join(map(str, bbox))
    output_folder = os.path.join(BASE_DIR, bbox_name)
    try:
        os.mkdir(output_folder)
    except OSError:
        print("Creation of the directory %s failed\n" % output_folder)
        return
    else:
        print("Successfully created the directory %s\n" % output_folder)

    # On Mac OS, there seem to be some bugs reading proxy settings from the operating system.
    # It sometimes causes requests to hang when using multiprocessing.
    session = requests.Session()
    session.trust_env = False  # Don't read proxy settings from OS

    # With the tags surface-land and mission-2019
    pano_url = f"https://api.data.amsterdam.nl/panorama/panoramas/?tags={API_TAGS}&bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}&srid=28992"

    # Get url of next API page
    next_page = get_pano_data(pano_url, output_folder, session)

    # Exit the while loop if there is no next page
    while next_page:
        # Get url of next API page
        next_page = get_pano_data(next_page, output_folder, session)

    print("Thread completed")

if __name__ == '__main__':
    start_time = time.time()

    # Create base directory
    if not os.path.exists(BASE_DIR):
        print("Created base directory: " + BASE_DIR)
        os.makedirs(BASE_DIR)

    p = ThreadPool(processes = NUM_WORKERS)
    p.imap_unordered(download_image, BBOX_LIST) # We define BBOX_LIST so we can iterate over it.

    p.close()
    p.join()

    print("--- %s seconds ---" % (time.time() - start_time))
