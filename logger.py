import os, json
from pathlib import Path
from datetime import datetime, timezone

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "EmranAIShaper/digital-twin-logs"

_scheduler = None

def _get_scheduler():
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    if not HF_TOKEN:
        return None
    try:
        from huggingface_hub import CommitScheduler
        Path("logs").mkdir(exist_ok=True)
        _scheduler = CommitScheduler(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            folder_path=Path("logs"),
            path_in_repo="logs",
            token=HF_TOKEN,
            private=True,
            every=5,
        )
        print(f"Logger ready → private dataset: {DATASET_REPO}")
    except Exception as e:
        print(f"Logger init failed (non-fatal): {e}")
    return _scheduler


def log_conversation(user_msg: str, ai_response: str):
    """Append one conversation turn to today's log file — non-blocking."""
    scheduler = _get_scheduler()
    if not scheduler:
        return
    try:
        log_file = Path("logs") / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_msg[:500],
            "assistant": ai_response[:3000],
        }
        with scheduler.lock:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"Log write failed (non-fatal): {e}")
