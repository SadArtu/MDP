#####################################################
# ! loading zip file do raw folder from computer
#########################################33333#######

import zipfile
import os
import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def extract_zip(zip_file, extract_to):
    """
    Extracts contents of a zip file to a specified directory.

    Args:
        zip_file (str): Path to the zip file.
        extract_to (str): Directory where the contents will be extracted.
    """
    if not os.path.exists(zip_file):
        logging.error(f"Zip file does not exist: {zip_file}")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Files extracted successfully.")
    except Exception as e:
        logging.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    setup_logging()

    default_zip_path = "/home/artur_176/Downloads/archive.zip"
    default_extract_to = "/home/artur_176/CNN/CNN/datasets/raw"

    if len(sys.argv) == 3:
        zip_path = sys.argv[1]
        extract_to = sys.argv[2]

    else:
        logging.info("No arguments provided. Using default paths.")
        zip_path = default_zip_path
        extract_to = default_extract_to

    extract_zip(zip_path, extract_to)
