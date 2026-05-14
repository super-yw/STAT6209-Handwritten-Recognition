import os
from huggingface_hub import snapshot_download
from config import MODEL_NAME, MODEL_DIR          



def download_model():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"Model already exists at: {MODEL_DIR}")
        return MODEL_DIR

    print(f"Start downloading: {MODEL_NAME} ...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR
    )

    print(f"Finished downloading. Model saved at: {MODEL_DIR}")
    return MODEL_DIR


if __name__ == "__main__":
    download_model()
