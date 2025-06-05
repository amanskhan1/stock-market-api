import os
import redis
import json
import pandas as pd
import requests
from dotenv import load_dotenv
import io

load_dotenv()

def sync_equity_data():
    redis_url = "rediss://default:AZPlAAIjcDFkYzhiOTZkNzJkMWY0MDk4YmJjOGE1ZTgzZTUyMTE0MHAxMA@master-doberman-37861.upstash.io:6379"
    if not redis_url:
        raise ValueError("REDIS_URL is not set.")

    redis_client = redis.from_url(redis_url, decode_responses=True)
    cache_key = "equity_list_cache"

    try:
        nse_url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/"
        }

        with requests.Session() as session:
            session.headers.update(headers)

            # Hit homepage to establish cookies
            session.get("https://www.nseindia.com", timeout=10)

            # Fetch CSV file
            response = session.get(nse_url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(io.BytesIO(response.content), on_bad_lines='skip')
                data = df.to_dict(orient="records")
                redis_client.setex(cache_key, 86400, json.dumps(data))  # TTL 1 day
                print("✅ Equity data synced to Redis.")
            else:
                print(f"❌ Failed to fetch NSE data. Status code: {response.status_code}")

    except Exception as e:
        print(f"❌ Error syncing data: {e}")

if __name__ == "__main__":
    sync_equity_data()
