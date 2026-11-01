import os
import ssl
import httpx
from huggingface_hub import snapshot_download

# 1. Force the underlying HTTPX client to bypass SSL verification blocks
original_init = httpx.Client.__init__
def patched_init(self, *args, **kwargs):
    kwargs['verify'] = False
    original_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_init

# 2. Apply general global fallback contexts
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("⏳ Starting model download...")
try:
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="./local_models/all-MiniLM-L6-v2"
    )
    print("\n✅ Success! Model saved locally in ./local_models/all-MiniLM-L6-v2")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
