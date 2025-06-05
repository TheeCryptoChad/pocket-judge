
# push_dataset.py
import tarfile
from huggingface_hub import HfApi
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ID = "PocketJudge/pose"
ARCHIVE_NAME = "data.tar.gz"
SOURCE_DIR = os.path.join(SCRIPT_DIR, "../data")

# Tar the dataset
print(f"Creating archive {ARCHIVE_NAME} from {SOURCE_DIR}...")
with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
    tar.add(SOURCE_DIR, arcname=os.path.basename(SOURCE_DIR))

# Upload to Hugging Face
api = HfApi()
print(f"Uploading to {REPO_ID}...")
api.upload_file(
    path_or_fileobj=ARCHIVE_NAME,
    path_in_repo=ARCHIVE_NAME,
    repo_id=REPO_ID,
    repo_type="dataset"
)

os.remove(ARCHIVE_NAME)

print("✅ Dataset pushed.")