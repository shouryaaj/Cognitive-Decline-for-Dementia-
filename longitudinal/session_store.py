import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/user_sessions.csv")

def save_session(record):
    df = pd.DataFrame([record])

    if DATA_PATH.exists():
        old = pd.read_csv(DATA_PATH)
        df = pd.concat([old, df], ignore_index=True)

    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

def load_user_history(user_id):
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)
    return df[df["user_id"] == user_id].sort_values("timestamp")
