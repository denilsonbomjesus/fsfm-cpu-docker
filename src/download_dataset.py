
import argparse
import requests
import os
import zipfile
import tarfile
from tqdm import tqdm

def download_file(url, destination):
    """
    Downloads a file from a URL to a destination, showing a progress bar.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
                desc="Downloading", ascii=True
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"Successfully downloaded {os.path.basename(destination)}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def extract_file(filepath, destination_dir):
    """
    Extracts a compressed file (.zip, .tar.gz, .tgz) to a destination directory.
    """
    try:
        print(f"Extracting {os.path.basename(filepath)}...")
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(destination_dir)
        elif filepath.endswith('.tar.gz') or filepath.endswith('.tgz'):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(destination_dir)
        else:
            print(f"Unsupported file format: {os.path.basename(filepath)}. Only .zip and .tar.gz are supported.")
            return False
        
        print(f"Successfully extracted to {destination_dir}.")
        return True
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"Error extracting file: {e}")
        return False

def main():
    """
    Main function to download and extract a dataset.
    """
    parser = argparse.ArgumentParser(description="Generic Dataset Downloader and Extractor.")
    parser.add_argument("--url", type=str, required=True, help="URL of the dataset to download.")
    parser.add_argument("--dest", type=str, required=True, help="Destination directory to extract the dataset.")
    
    args = parser.parse_args()
    
    # Ensure destination directory exists
    os.makedirs(args.dest, exist_ok=True)
    
    # Define the downloaded file path
    filename = os.path.basename(args.url)
    filepath = os.path.join(args.dest, filename)
    
    # Download the file
    if not download_file(args.url, filepath):
        return

    # Extract the file if it is a compressed format
    if filepath.endswith(('.zip', '.tar.gz', '.tgz')):
        if extract_file(filepath, args.dest):
            # Clean up the compressed file after extraction
            print(f"Cleaning up {os.path.basename(filepath)}...")
            os.remove(filepath)
            print("Cleanup complete.")
    
    print("Dataset setup complete.")

if __name__ == "__main__":
    main()
