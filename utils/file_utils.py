import os
import gzip
import requests

os.makedirs("data", exist_ok=True)

def download_file(url, filename):
    """Download and unzip a .gz file into the data folder if it doesn't exist already."""
    filepath = os.path.join("data", filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename} ...")
        r = requests.get(url)
        gz_path = filepath + ".gz"
        with open(gz_path, "wb") as f:
            f.write(r.content)
        with gzip.open(gz_path, "rb") as f_in, open(filepath, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"{filename} downloaded and extracted to data/")
    else:
        print(f"{filename} already exists in data/, skipping download.")