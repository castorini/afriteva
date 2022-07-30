import os
import sys

from dotenv import load_dotenv

from utils import authenticate_blob_client
from utils import upload_folder

load_dotenv()


if __name__ == "__main__":
    folder, container = sys.argv[1:]
    client = authenticate_blob_client(
      os.getenv("BLOB_ACCOUNT_NAME"), os.getenv("BLOB_CONN_KEY"))
    upload_folder(folder, container, client)
