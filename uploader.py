import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from huggingface_hub import HfApi
from config import (
    HF_TOKEN, HF_DATASET_REPO,
    LOCAL_DATA_PATH, PENDING_PATH, STATS_PATH, DATA_DIR,
)

api = HfApi(token=HF_TOKEN)

DEFAULT_STATS = {
    "total_generated": 0,
    "total_accepted": 0,
    "total_rejected": 0,
    "total_uploads": 0,
    "last_upload": None,
    "start_time": datetime.now(timezone.utc).isoformat(),
}


def load_stats() -> dict:
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            return json.load(f)
    return DEFAULT_STATS.copy()


def save_stats(stats: dict) -> None:
    public = {k: v for k, v in stats.items() if not k.startswith("_")}
    with open(STATS_PATH, "w") as f:
        json.dump(public, f, indent=2)


def load_seen_prompts() -> set[str]:
    if LOCAL_DATA_PATH.exists():
        df = pd.read_parquet(LOCAL_DATA_PATH, columns=["prompt"])
        return set(df["prompt"].tolist())
    return set()


def save_example(example: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(PENDING_PATH, "a") as f:
        f.write(json.dumps(example) + "\n")


def consolidate() -> int:
    if not PENDING_PATH.exists():
        return 0

    pending_rows = []
    with open(PENDING_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                pending_rows.append(json.loads(line))

    if not pending_rows:
        return 0

    pending_df = pd.DataFrame(pending_rows)

    if LOCAL_DATA_PATH.exists():
        existing_df = pd.read_parquet(LOCAL_DATA_PATH)
        combined = pd.concat([existing_df, pending_df], ignore_index=True)
    else:
        combined = pending_df

    combined.to_parquet(LOCAL_DATA_PATH, index=False)
    PENDING_PATH.unlink()

    return len(pending_rows)


def upload_to_hub(stats: dict) -> None:
    consolidated = consolidate()
    if consolidated == 0 and not LOCAL_DATA_PATH.exists():
        return

    api.upload_file(
        path_or_fileobj=str(LOCAL_DATA_PATH),
        path_in_repo="data/dataset.parquet",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Add {consolidated} examples (total: {stats.get('total_accepted', 0)})",
    )

    stats["total_uploads"] = stats.get("total_uploads", 0) + 1
    stats["last_upload"] = datetime.now(timezone.utc).isoformat()
    save_stats(stats)
