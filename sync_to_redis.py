import os
import redis
import pandas as pd
from nselib import capital_market
from dotenv import load_dotenv

load_dotenv()  # loads REDIS_URL from .env or GitHub secret

def sync_equity_data():
    redis_url = "rediss://default:AZPlAAIjcDFkYzhiOTZkNzJkMWY0MDk4YmJjOGE1ZTgzZTUyMTE0MHAxMA@master-doberman-37861.upstash.io:6379"
    if not redis_url:
        raise ValueError("REDIS_URL is not set.")

    redis_client = redis.from_url(redis_url, decode_responses=True)
    cache_key = "equity_list_cache"
    try:
        df = capital_market.equity_list()
        if isinstance(df, pd.DataFrame):
            data = df.to_dict(orient="records")
            redis_client.setex(cache_key, 86400, str(data))  # TTL 1 day
            print("✅ Data synced to Redis.")
        else:
            print("❌ Unexpected format.")
    except Exception as e:
        print(f"❌ Error syncing data: {e}")

if __name__ == "__main__":
    sync_equity_data()
