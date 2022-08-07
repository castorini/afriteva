import os
import sys

from dotenv import load_dotenv

from utils import authenticate_blob_client
from utils import upload_folder
from utils import download_folder

load_dotenv()


if __name__ == "__main__":
    action, *folder, container = sys.argv[1:]
    client = authenticate_blob_client(
      os.getenv("BLOB_ACCOUNT_NAME"), os.getenv("BLOB_CONN_KEY"))
    print(folder)
    if action == "upload":
      upload_folder(*folder, container, client)
    elif action == "download":
      download_folder(*folder, container, client)
    else:
      raise ValueError("`action` unrecognized")
