import os
import gzip
import requests

os.makedirs("data/raw", exist_ok=True)

def download_file(url, filename):
    """Download and unzip a .gz file into the data folder if it doesn't exist already."""
    filepath = os.path.join("data/raw", filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename} ...")
        r = requests.get(url)
        gz_path = filepath + ".gz"
        with open(gz_path, "wb") as f:
            f.write(r.content)
        with gzip.open(gz_path, "rb") as f_in, open(filepath, "wb") as f_out:
            f_out.write(f_in.read())
        os.remove(gz_path)  # delete the .gz file after extraction
        print(f"{filename} downloaded, extracted to data/raw, and .gz file deleted.")
    else:
        print(f"{filename} already exists in data/raw, skipping download.")


os.makedirs("data/tokenized", exist_ok=True)

def save_tokenized_data(sentences, filename):
    """
    Save a list of tokenized sentences to a file.

    Args:
        sentences (list of list of str): Tokenized sentences.
        filename (str): Path to save the tokenized sentences.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for tokens in sentences:
            f.write(" ".join(tokens) + "\n")
    print(f"Tokenized data saved to {filename}")

os.makedirs("data/encoded", exist_ok=True)

def save_encoded_data(encoded_sentences, filename):
    """
    Save a list of encoded sentences (integer sequences) to a file.

    Args:
        encoded_sentences (list of list of int]): Encoded sentences.
        filename (str): Path to save the encoded sentences.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for enc in encoded_sentences:
            # save as space-separated integers
            f.write(" ".join(map(str, enc)) + "\n")
    print(f"Encoded data saved to {filename}")    