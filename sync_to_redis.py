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
        url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com"
        }

        response = requests.get(url, headers=headers, timeout=10)
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
