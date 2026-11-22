import json
import os
import time
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv(dotenv_path=".env")

logger = logging.getLogger("shopping_assistant")

app = FastAPI(title="AI Shopping Assistant")

# Network settings: fail fast instead of hanging, retry transient errors.
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
SERPAPI_TIMEOUT_SECONDS = float(os.getenv("SERPAPI_TIMEOUT_SECONDS", "20"))
SERPAPI_MAX_RETRIES = int(os.getenv("SERPAPI_MAX_RETRIES", "2"))

_client = None


def get_openai_client():
    """Create the OpenAI client lazily so the app can start without a key."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(timeout=OPENAI_TIMEOUT_SECONDS, max_retries=OPENAI_MAX_RETRIES)
    return _client

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

    try:
        response = get_openai_client().responses.create(
            model="gpt-4.1-nano",
            input=prompt
        )
    except (OpenAIError, RuntimeError) as e:
        logger.warning("OpenAI extraction failed: %s", e)
        return None

    text = (response.output_text or "").strip().replace("```json", "").replace("```", "").strip()

    # The model sometimes wraps JSON in extra prose; keep only the object.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse extraction output: %r", text[:200])
        return {}

    return data if isinstance(data, dict) else {}


ORDER_FIELDS = {"product", "noise_preference", "use_case", "chemical_type", "power_type", "max_price"}


def merge_order(order, extracted):
    for key, value in extracted.items():
        if key in ORDER_FIELDS and value not in [None, "", [], {}]:
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
    parts = [order.get(k) for k in ("noise_preference", "chemical_type", "power_type", "product")]
    query = " ".join(str(p) for p in parts if p)
    if order.get("max_price"):
        query += f" under {order.get('max_price')}"

    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return "Product search is unavailable: SERPAPI_KEY is not set on the server."

    params = {
        "engine": "amazon",
        "k": query,
        "amazon_domain": "amazon.com",
        "api_key": api_key
    }

    results = None
    last_error = None
    for attempt in range(1, SERPAPI_MAX_RETRIES + 2):
        try:
            search = GoogleSearch(params)
            search.timeout = SERPAPI_TIMEOUT_SECONDS
            results = search.get_dict()
            break
        except Exception as e:  # network errors, timeouts, bad JSON
            last_error = e
            logger.warning("SerpAPI attempt %d failed: %s", attempt, e)
            if attempt <= SERPAPI_MAX_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))

    if results is None:
        return f"Product search failed due to a network problem ({type(last_error).__name__}). Please try again."

    if results.get("error"):
        logger.warning("SerpAPI returned an error: %s", results["error"])
        return f"Product search failed: {results['error']}"

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
    if extracted is None:
        return {
            "reply": "I couldn't reach the AI service just now. Please try again in a moment.",
            "current_order": order
        }
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