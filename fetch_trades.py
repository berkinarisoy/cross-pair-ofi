import gzip
import json
import os
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests

BASE = "https://api.massive.com"
PAIRS = ["ETH-USD", "BTC-USD", "ETH-BTC"]
HISTORY_DAYS = 30
OUT_DIR = "data/raw"


def api_key():
    """Read the Massive key from the environment. Never hardcoded."""
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MASSIVE_API_KEY is not set")
    return key


def fetch_pair(pair, start, end, key, out_path):
    """Page through every trade in [start, end) for `pair` and write them to out_path as gzip-jsonl."""
    ticker = f"X:{pair.replace('-', '')}"
    url = f"{BASE}/v3/trades/{ticker}"
    params = {"timestamp.gte": start, "timestamp.lt": end, "order": "asc",
              "sort": "timestamp", "limit": 50000, "apiKey": key}
    n = 0
    with gzip.open(out_path, "wt") as fh:
        while url:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            body = r.json()
            for rec in body.get("results", []):
                fh.write(json.dumps(rec) + "\n")
            n += len(body.get("results", []))
            print(f"{pair}: {n:,} trades")
            url = body.get("next_url")
            params = {}
            if url and "apiKey" not in url:
                url += ("&" if urlparse(url).query else "?") + f"apiKey={key}"
            if url:
                time.sleep(0.2)
    return n


def main():
    """Fetch HISTORY_DAYS of trades for every pair in PAIRS into OUT_DIR."""
    key = api_key()
    os.makedirs(OUT_DIR, exist_ok=True)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=HISTORY_DAYS)
    for pair in PAIRS:
        fetch_pair(pair, start.isoformat(), end.isoformat(), key,
                   f"{OUT_DIR}/trades_hist_{pair}.jsonl.gz")


if __name__ == "__main__":
    main()
