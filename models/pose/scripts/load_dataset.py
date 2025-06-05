# load_dataset.py
import tarfile
from huggingface_hub import hf_hub_download
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "PocketJudge/pose"
ARCHIVE_NAME = "data.tar.gz"
TARGET_DIR = os.path.join(SCRIPT_DIR, "../")

print(f"Downloading {ARCHIVE_NAME} from {REPO_ID}...")

path = hf_hub_download(repo_id=REPO_ID, filename=ARCHIVE_NAME,repo_type="dataset")

# Clear existing folder
TARGET_PATH = os.path.join(TARGET_DIR, "data")
if os.path.exists(TARGET_PATH):
    shutil.rmtree(TARGET_PATH)

print(f"Extracting to {TARGET_DIR}... this may take a minute or two")
with tarfile.open(path, "r:gz") as tar:
    tar.extractall(TARGET_DIR)

os.remove(path)

print("✅ Dataset loaded.")

