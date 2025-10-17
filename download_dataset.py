# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:39:54 2021

@author: serdarhelli
Updated with error handling and retry logic
"""

import requests
from zipfile import ZipFile
from io import BytesIO
import time
import os


def download_dataset(save_path, max_retries=3):
    """
    Download dental panoramic X-ray dataset from Mendeley Data

    Args:
        save_path: Path to save the extracted dataset
        max_retries: Number of retry attempts if download fails
    """
    url = "https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/hxt48yk462-1.zip"

    for attempt in range(max_retries):
        try:
            print(f"Downloading dataset (attempt {attempt + 1}/{max_retries})...")
            print(f"URL: {url}")

            # Download with timeout and streaming
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()  # Raise exception for bad status codes

            # Check content type
            content_type = response.headers.get('content-type', '')
            print(f"Content-Type: {content_type}")
            print(f"Content-Length: {response.headers.get('content-length', 'Unknown')} bytes")

            # Read content
            content = response.content
            print(f"Downloaded {len(content)} bytes")

            # Check if content looks like a zip file (magic number)
            if len(content) < 4:
                raise ValueError("Downloaded file is too small")

            # ZIP files start with PK (0x504B)
            if content[:2] != b'PK':
                # Try to save the error response for debugging
                with open('download_error.html', 'wb') as f:
                    f.write(content)
                raise ValueError(
                    "Downloaded file is not a ZIP file. "
                    "Response saved to 'download_error.html' for debugging. "
                    "The dataset URL might have changed."
                )

            print("Extracting files...")
            z = ZipFile(BytesIO(content))
            z.extractall(save_path)

            print(f"Completed! Dataset extracted to: {save_path}")
            print(f"Extracted files: {len(z.namelist())} files")

            return True

        except requests.exceptions.RequestException as e:
            print(f"Download error: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\nDownload failed after all retries.")
                print("Please try downloading manually from:")
                print("https://data.mendeley.com/datasets/hxt48yk462/1")
                raise

        except (ValueError, ZipFile.BadZipFile) as e:
            print(f"File validation error: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("\nDataset download/extraction failed.")
                print("\nAlternative: Download manually")
                print("1. Visit: https://data.mendeley.com/datasets/hxt48yk462/1")
                print("2. Download the ZIP file")
                print(f"3. Extract to: {os.path.abspath(save_path)}")
                raise

        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    return False


if __name__ == "__main__":
    # Test the download function
    import sys
    save_path = sys.argv[1] if len(sys.argv) > 1 else "./Data"
    download_dataset(save_path)
