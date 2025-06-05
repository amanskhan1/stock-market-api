import os
import redis
import json
import pandas as pd
import requests
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

def sync_equity_data():
    redis_url = "rediss://default:AZPlAAIjcDFkYzhiOTZkNzJkMWY0MDk4YmJjOGE1ZTgzZTUyMTE0MHAxMA@master-doberman-37861.upstash.io:6379"
    if not redis_url:
        raise ValueError("REDIS_URL is not set.")

    redis_client = redis.from_url(redis_url, decode_responses=True)
    cache_key = "equity_list_cache"

    try:
        url = "https://www.nseindia.com/api/master-quote-equity?csv=true"  # Alt API (less likely to block)
        alt_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"  # Main CSV (403 fixable)

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://www.nseindia.com/"
        })

        # First request to get cookies (simulate browser visit)
        session.get("https://www.nseindia.com", timeout=10)

        # Then fetch the CSV
        response = session.get(alt_url, timeout=10)
        response.encoding = "utf-8"

        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
            data = df.to_dict(orient="records")
            redis_client.setex(cache_key, 86400, json.dumps(data))
            print("✅ Equity data synced to Redis.")
        else:
            print(f"❌ Failed to fetch: {response.status_code}")
    except Exception as e:
        print(f"❌ Error syncing data: {e}")

if __name__ == "__main__":
    sync_equity_data()
