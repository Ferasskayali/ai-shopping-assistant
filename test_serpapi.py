from serpapi import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

params = {
    "engine": "amazon",
    "k": "quiet chemical metering pump",
    "amazon_domain": "amazon.com",
    "api_key": os.getenv("SERPAPI_KEY")
}

search = GoogleSearch(params)
results = search.get_dict()

products = results.get("organic_results", [])

for p in products[:5]:
    print("TITLE:", p.get("title"))
    print("PRICE:", p.get("price"))
    print("LINK:", p.get("link"))
    print("------")