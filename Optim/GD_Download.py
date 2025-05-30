# Usage
#     python3 download_collection.py -o <collection_owner> -c <collection_name>
#
# Description
#     This script will download all models contained within a collection.
#
import sys,json,requests
import getopt
import os
import shutil
import zipfile
from pathlib import Path

def extract_first_thumbnail_from_zips(zip_dir, output_dir):
    """
    Extracts the first thumbnail image from each zip file in zip_dir
    and saves it to output_dir with the zip file's stem as the filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    for zip_path in Path(zip_dir).glob('*.zip'):
        with zipfile.ZipFile(zip_path, 'r') as z:
            thumbnail_files = [f for f in z.namelist() if f.startswith('thumbnails/') and not f.endswith('/')]
            if thumbnail_files:
                thumb_file = thumbnail_files[0]
                ext = Path(thumb_file).suffix
                out_name = zip_path.stem + ext
                out_path = Path(output_dir) / out_name
                with z.open(thumb_file) as source, open(out_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                print(f"Extracted {thumb_file} from {zip_path.name} as {out_name}")


if sys.version_info[0] < 3:
    raise Exception("Python 3 or greater is required. Try running `python3 download_collection.py`")

collection_name = ''
owner_name = ''
download_dir = ''
thumbnail_dir = ''


# Read options
optlist, args = getopt.getopt(sys.argv[1:], 'o:c:d:t:')

sensor_config_file = ''
private_token = ''
for o, v in optlist:
    if o == "-o":
        owner_name = v.replace(" ", "%20")
    if o == "-c":
        collection_name = v.replace(" ", "%20")
    if o == "-d":
            download_dir = v
    if o == "-t":
            thumbnail_dir = v

if not os.path.exists(download_dir):
    os.makedirs(download_dir)
if thumbnail_dir and not os.path.exists(thumbnail_dir):
    os.makedirs(thumbnail_dir)
    
if not owner_name:
    print('Error: missing `-o <owner_name>` option')
    quit()

if not collection_name:
    print('Error: missing `-c <collection_name>` option')
    quit()


print("Downloading models from the {}/{} collection.".format(owner_name, collection_name.replace("%20", " ")))

page = 1
count = 0

# The Fuel server URL.
base_url ='https://fuel.gazebosim.org/'

# Fuel server version.
fuel_version = '1.0'

# Path to get the models in the collection
next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)

# Path to download a single model in the collection
download_url = base_url + fuel_version + '/{}/models/'.format(owner_name)

print("Downloading models from the {}/{} collection.".format(owner_name, collection_name.replace("%20", " ")))

# Iterate over the pages
while True:
    url = base_url + fuel_version + next_url

    # Get the contents of the current page.
    r = requests.get(url)

    if not r or not r.text:
        break

    # Convert to JSON
    models = json.loads(r.text)

    # Compute the next page's URL
    page = page + 1
    next_url = '/models?page={}&per_page=100&q=collections:{}'.format(page,collection_name)
    
    

    # Download each model 
    for model in models:
        count+=1
        model_name = model['name']
        print ('Downloading (%d) %s' % (count, model_name))
        download = requests.get(download_url+model_name+'.zip', stream=True)
        file_path = os.path.join(download_dir, model_name+'.zip')
        with open(file_path, 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)
    

print('Done.')

if thumbnail_dir:
    print(f"Extracting thumbnails to {thumbnail_dir} ...")
    extract_first_thumbnail_from_zips(download_dir, thumbnail_dir)
    print("Thumbnail extraction complete.")