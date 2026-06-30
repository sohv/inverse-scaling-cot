import logging
import os

from dotenv import load_dotenv

load_dotenv()

_token = os.environ.get("HF_TOKEN", "").strip()
if _token:
    try:
        from huggingface_hub import login

        login(token=_token, add_to_git_credential=False)
        logging.getLogger(__name__).info("Logged in to Hugging Face Hub")
    except Exception as exc:
        logging.getLogger(__name__).warning("HF login failed: %s", exc)
