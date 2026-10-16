import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
UPLOAD_EVERY = int(os.getenv("UPLOAD_EVERY", "50"))
MIN_CHOSEN_LENGTH = int(os.getenv("MIN_CHOSEN_LENGTH", "200"))
MIN_REJECTED_LENGTH = int(os.getenv("MIN_REJECTED_LENGTH", "30"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "1.5"))

DATA_DIR = Path("data")
LOCAL_DATA_PATH = DATA_DIR / "dataset.parquet"
STATS_PATH = DATA_DIR / "stats.json"
PENDING_PATH = DATA_DIR / "pending.jsonl"

HELPFUL_SYSTEM = (
    "You are an expert assistant. Provide thorough, accurate, well-structured answers. "
    "Include relevant examples and explain your reasoning clearly."
)
TERSE_SYSTEM = (
    "You are a casual assistant in a hurry. Give very brief answers without explaining "
    "your reasoning or providing examples. Be superficial."
)
