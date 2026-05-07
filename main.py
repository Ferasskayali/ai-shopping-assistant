import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv(dotenv_path=".env")

app = FastAPI(title="AI Shopping Assistant")
client = OpenAI()

STATE_FILE = "session_state.json"


class RequestBody(BaseModel):
    command: str


def default_state():
    return {
        "current_order": {
            "product": None,
            "noise_preference": None,
            "use_case": None,
            "chemical_type": None,
            "power_type": None,
            "max_price": None,
            "market_range_shown": False
        }
    }


def load_state():
    if not os.path.exists(STATE_FILE):
        return default_state()
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def extract_info(message, order):
    prompt = f"""
Extract shopping information from the user's message.

Current order:
{json.dumps(order, indent=2)}

User message:
{message}

Return ONLY valid JSON with these fields:
product, noise_preference, use_case, chemical_type, power_type, max_price.

If unknown, use null.
"""

    response = client.responses.create(
        model="gpt-4.1-nano",
        input=prompt
    )

    text = response.output_text.strip().replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        return {}


def merge_order(order, extracted):
    for key, value in extracted.items():
        if value not in [None, "", [], {}]:
            order[key] = value
    return order


def ask_missing(order):
    if not order.get("product"):
        return "What product are you looking for?"

    if "pump" in str(order.get("product")).lower():
        if not order.get("noise_preference"):
            return "Do you prefer quiet operation or is noise not important?"

        if not order.get("use_case"):
            return "Will it be intermittent, continuous, home, or industrial-like use?"

        if not order.get("chemical_type"):
            return "What chemical type will it handle? Example: bleach, acid, solvent, cleaner, or unknown hazardous chemical."

        if not order.get("power_type"):
            return "Do you prefer electric, air-operated, or no preference?"

    return None


def estimate_market(order):
    quiet = str(order.get("noise_preference", "")).lower()
    use_case = str(order.get("use_case", "")).lower()

    if "quiet" in quiet and "industrial" in use_case:
        return """
Based on your specifications:

Quiet + chemical-resistant + industrial-like use

Typical market ranges:

Low range: $180–$300
Small electric dosing pumps, lighter duty.

Recommended range: $300–$600
Better electric diaphragm or metering pumps.

Professional range: $600–$1,500+
Industrial-grade pumps.

For your case, I recommend around $500–$600.

What maximum budget do you want me to use?
"""

    return """
Typical market ranges:

Basic: $100–$250
Mid-range: $250–$500
Professional: $500–$1,500+

What maximum budget do you want me to use?
"""


def search_amazon_with_serpapi(order):
    query = f"{order.get('noise_preference')} {order.get('chemical_type')} {order.get('power_type')} {order.get('product')} under {order.get('max_price')}"

    params = {
        "engine": "amazon",
        "k": query,
        "amazon_domain": "amazon.com",
        "api_key": os.getenv("SERPAPI_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    products = results.get("organic_results", [])

    if not products:
        return "No Amazon products found. Try a different keyword or higher budget."

    reply = f"Amazon search used this query:\n{query}\n\nTop Amazon results:\n\n"

    for i, product in enumerate(products[:5], start=1):
        title = product.get("title", "No title")
        price = product.get("price", "No price shown")
        rating = product.get("rating", "No rating shown")

        link = (
            product.get("product_link")
            or product.get("link")
            or product.get("serpapi_product_api")
            or "No link available"
        )

        asin = product.get("asin")
        if asin and link == "No link available":
            link = f"https://www.amazon.com/dp/{asin}"

        reply += f"""
{i}. {title}

Price: {price}

Rating: {rating}

Open Product:
{link}

------------------------

"""

    reply += "Approval required before purchase."
    return reply


@app.get("/")
def home():
    return {"message": "AI Shopping Assistant is running"}


@app.post("/reset")
def reset():
    state = default_state()
    save_state(state)
    return {"reply": "Conversation reset."}


@app.post("/chat")
def chat(body: RequestBody):
    state = load_state()
    order = state["current_order"]

    extracted = extract_info(body.command, order)
    order = merge_order(order, extracted)

    missing = ask_missing(order)
    if missing:
        state["current_order"] = order
        save_state(state)
        return {"reply": missing, "current_order": order}

    if not order.get("market_range_shown"):
        order["market_range_shown"] = True
        state["current_order"] = order
        save_state(state)
        return {"reply": estimate_market(order), "current_order": order}

    if not order.get("max_price"):
        state["current_order"] = order
        save_state(state)
        return {"reply": "What maximum budget do you want me to use?", "current_order": order}

    state["current_order"] = order
    save_state(state)

    return {
        "reply": search_amazon_with_serpapi(order),
        "current_order": order
    }