import os
import time
import shutil
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = os.environ.get("HEIST_HF_REPO", "Shanmuk4622/heist-architect-v2")
TOKEN = os.environ.get("HF_TOKEN")
OUT_DIR = os.environ.get("HEIST_OUT_DIR", "hf_checkpoints")
MAX_PASSES = int(os.environ.get("HEIST_MAX_PASSES", "8"))
SLEEP_SECONDS = int(os.environ.get("HEIST_RETRY_SLEEP", "5"))

    
api = HfApi(token=TOKEN) if TOKEN else HfApi()
files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
if not files:
    raise SystemExit("No files found in repo")

print(f"Repo: {REPO_ID}")
print(f"Files on hub: {len(files)}")
print(f"Output dir: {os.path.abspath(OUT_DIR)}")

os.makedirs(OUT_DIR, exist_ok=True)

remaining = set(files)
for p in range(1, MAX_PASSES + 1):
    if not remaining:
        break

    print(f"\nPass {p}/{MAX_PASSES} | Remaining: {len(remaining)}")
    failed = set()

    for path in sorted(remaining):
        try:
            cache_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="model",
                filename=path,
                token=TOKEN,
            )
            target_path = os.path.join(OUT_DIR, *path.split("/"))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(cache_path, target_path)
            print(f"  OK   {path}")
        except Exception as ex:
            failed.add(path)
            print(f"  FAIL {path} :: {ex}")

    remaining = failed
    if remaining:
        time.sleep(SLEEP_SECONDS)

print("\nDownload finished")
print(f"Total files on hub: {len(files)}")
print(f"Downloaded: {len(files) - len(remaining)}")
print(f"Missing: {len(remaining)}")

if remaining:
    missing_path = os.path.join(OUT_DIR, "_missing_files.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for path in sorted(remaining):
            f.write(path + "\n")
    print(f"Missing list saved to: {missing_path}")
    raise SystemExit(2)

print("All files downloaded successfully.")
