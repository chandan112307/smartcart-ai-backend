"""Query Understanding Agent.

Entry point of the intelligence pipeline.
Converts natural language → structured StructuredQuery JSON.

README
------
All downstream agents depend on this agent's output.
It ALWAYS returns a StructuredQuery (never raw text).
If the LLM is unavailable, a rule-based fallback ensures the pipeline continues.
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Tuple

from app.core.exceptions import AgentException
from app.data.models import (
    ItemAttributes,
    QueryConstraints,
    QueryFilters,
    QueryIntent,
    QueryMetadata,
    StructuredItem,
    StructuredQuery,
)
from app.llm.manager import LLMManager

_SCHEMA_EXAMPLE = """{
  "normalized_query": "packaged milk under 60",
  "product": "packaged milk",
  "items": [
    {
      "name": "packaged milk",
      "category": "dairy",
      "attributes": {
        "quantity": null,
        "unit": null,
        "preferences": ["cheap"]
      }
    }
  ],
  "constraints": {
    "budget": {"operator": "under", "amount": 60.0, "currency": "INR"},
    "servings": null,
    "preferences": ["cheap"]
  },
  "metadata": {
    "confidence": 0.88,
    "notes": "normalized from raw query"
  },
  "filters": {
    "max_price": 60,
    "min_price": null,
    "category": "dairy",
    "quantity": "500ml",
    "brand": null
  },
  "intent": "product_search"
}"""

_PROMPT_TEMPLATE = """
Extract structured information from the grocery query below.

Query: "{query}"

Return JSON with fields:
- normalized_query (string): cleaned/standardized query in English
- product (string): primary product name
- items (list): normalized grocery entities with category and attributes
- constraints.budget (object|null): budget constraint (operator, amount, currency)
- constraints.servings (number|null): serving size if present
- constraints.preferences (list): user preference keywords
- metadata.confidence (number): confidence in [0,1]
- metadata.notes (string): short processing notes
- filters.max_price (number|null): maximum price if mentioned
- filters.min_price (number|null): minimum price if mentioned
- filters.category (string|null): category if mentioned
- filters.quantity (string|null): quantity/size if mentioned
- filters.brand (string|null): brand if mentioned
- intent (string): one of "product_search", "recipe", "cart_optimization", "exploratory", "unsupported"

Intent rules:
- "unsupported" if not grocery/food related.
- "recipe" if query mentions recipe, cook, make, prepare, or dish names
- "cart_optimization" if query mentions optimize, cart, total, basket
- "exploratory" if grocery-related but too vague to pinpoint exact item
- otherwise "product_search"
"""

# Simple keyword rules for rule-based fallback
_PRICE_PATTERN = re.compile(r"(?:under|below|less than|<|max|upto)\s*(?:rs\.?|₹)?\s*(\d+(?:\.\d+)?)", re.I)
_MIN_PRICE_PATTERN = re.compile(r"(?:above|over|more than|at least|>=?)\s*(?:rs\.?|₹)?\s*(\d+(?:\.\d+)?)", re.I)
_QUANTITY_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(kg|g|gram|grams|l|liter|litre|ml|pack|packs|pcs|pc|piece|pieces)\b", re.I)
_SERVINGS_PATTERN = re.compile(r"(?:for|serves?)\s*(\d+)\s*(?:people|persons|servings?)?", re.I)
_RECIPE_KEYWORDS = {"recipe", "cook", "prepare", "make", "dish", "biryani", "curry", "pasta", "soup", "salad"}
_CART_KEYWORDS = {"optimize", "cart", "basket", "total cost", "split"}
_EXPLORATORY_KEYWORDS = {"something", "anything", "some", "options", "ideas"}
_PREFERENCE_KEYWORDS = {
    "cheap": "cheap",
    "budget": "cheap",
    "organic": "organic",
    "healthy": "healthy",
    "premium": "premium",
    "fresh": "fresh",
}
_NON_GROCERY_KEYWORDS = {
    "laptop", "phone", "mobile", "charger", "headphones", "movie", "flight", "hotel",
    "stocks", "crypto", "insurance", "car", "bike", "shoes", "shirt",
}
_NOISE_TOKENS = {"pls", "plz", "please", "hey", "hi", "hello", "bhai", "yaar"}
_TOKEN_CLEAN = re.compile(r"[^\w₹\s\.]", re.UNICODE)
_STOPWORDS = {"find", "show", "get", "buy", "need", "want", "for", "me", "the", "a", "an", "under", "above"}

_KNOWN_PRODUCTS = {
    "milk", "bread", "eggs", "rice", "tomato", "onion", "oil", "butter",
    "pasta", "sugar", "atta", "flour", "salt", "tea", "coffee", "biscuits",
    "chips", "noodles", "cheese", "yogurt", "curd", "chicken", "ghee", "paneer",
    "snacks", "salad", "ginger", "lentils", "cumin seeds", "red chili powder",
    "wheat flour",
}

_NORMALIZATION_MAP = {
    "atta": "wheat flour",
    "jeera": "cumin seeds",
    "mirchi powder": "red chili powder",
    "dahi": "curd",
    "yoghurt": "curd",
    "yogurt": "curd",
    "milk": "packaged milk",
    "dal": "lentils",
    "ginger piece": "ginger",
}
# Used for product phrase detection after text normalization.
_MULTI_WORD_NORMALIZATION_PHRASES = sorted(
    [v for v in _NORMALIZATION_MAP.values() if " " in v],
    key=len,
    reverse=True,
)
_HIGH_CONFIDENCE_UNSUPPORTED = 0.95
_DEFAULT_PARSE_CONFIDENCE = 0.72
_GROCERY_CONTEXT_KEYWORDS = _KNOWN_PRODUCTS.union(
    {"grocery", "groceries", "food", "ingredient", "ingredients"}
)

logger = logging.getLogger(__name__)


def _normalize_text(query: str) -> str:
    text = unicodedata.normalize("NFKC", query)
    text = text.lower().strip()
    text = _TOKEN_CLEAN.sub(" ", text)
    tokens = [t for t in text.split() if t and t not in _NOISE_TOKENS]
    text = " ".join(tokens)
    for source, target in sorted(_NORMALIZATION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, text, flags=re.UNICODE)
    return " ".join(text.split())


def _extract_preferences(q: str) -> List[str]:
    words = set(q.split())
    return list(dict.fromkeys(v for k, v in _PREFERENCE_KEYWORDS.items() if k in words))


def _extract_product_and_items(q: str, intent: QueryIntent, preferences: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    if intent == QueryIntent.unsupported:
        return "", []
    if intent == QueryIntent.exploratory:
        return "grocery items", [{
            "name": "grocery items",
            "category": "general",
            "attributes": {"quantity": None, "unit": None, "preferences": preferences},
        }]

    tokens = [t.strip(".,!?") for t in q.split()]
    product = ""
    for phrase in _MULTI_WORD_NORMALIZATION_PHRASES:
        if phrase in q:
            product = phrase
            break
    if not product:
        for word in tokens:
            if word in _KNOWN_PRODUCTS:
                product = word
                break
    if not product:
        candidates = [t for t in tokens if t not in _STOPWORDS and not t.isdigit()]
        product = candidates[0] if candidates else q
    category = None
    if any(k in product for k in ("milk", "curd", "paneer", "ghee")):
        category = "dairy"
    elif any(k in product for k in ("chicken", "eggs")):
        category = "poultry"
    elif any(k in product for k in ("wheat flour", "rice", "lentils")):
        category = "staples"
    elif any(k in product for k in ("salad", "tomato", "onion", "ginger")):
        category = "vegetable"
    elif "snack" in product:
        category = "snacks"

    return product, [{
        "name": product,
        "category": category or "general",
        "attributes": {"quantity": None, "unit": None, "preferences": preferences},
    }]


def _rule_based_parse(query: str) -> Dict[str, Any]:
    """Simple rule-based fallback when LLM is unavailable."""
    q = _normalize_text(query)
    preferences = _extract_preferences(q)

    # Detect intent
    q_words = set(q.split())
    has_non_grocery_signal = bool(q_words.intersection(_NON_GROCERY_KEYWORDS))
    has_grocery_signal = bool(q_words.intersection(_GROCERY_CONTEXT_KEYWORDS))
    if has_non_grocery_signal and not has_grocery_signal:
        intent = QueryIntent.unsupported
    elif any(k in q for k in _RECIPE_KEYWORDS):
        intent = QueryIntent.recipe
    elif any(k in q for k in _CART_KEYWORDS):
        intent = QueryIntent.cart_optimization
    elif any(k in q.split() for k in _EXPLORATORY_KEYWORDS):
        intent = QueryIntent.exploratory
    else:
        intent = QueryIntent.product_search

    product, items = _extract_product_and_items(q, intent, preferences)

    # Extract constraints
    match = _PRICE_PATTERN.search(q)
    max_price = float(match.group(1)) if match else None
    min_match = _MIN_PRICE_PATTERN.search(q)
    min_price = float(min_match.group(1)) if min_match else None
    qty_match = _QUANTITY_PATTERN.search(q)
    quantity = qty_match.group(1) if qty_match else None
    unit = qty_match.group(2).lower() if qty_match else None
    servings_match = _SERVINGS_PATTERN.search(q)
    servings = int(servings_match.group(1)) if servings_match else None

    if items and quantity is not None:
        try:
            parsed_quantity = float(quantity)
        except ValueError:
            parsed_quantity = None
        items[0]["attributes"]["quantity"] = parsed_quantity
        items[0]["attributes"]["unit"] = unit

    return {
        "normalized_query": q,
        "product": product,
        "items": items,
        "constraints": {
            "budget": {"operator": "under", "amount": max_price, "currency": "INR"} if max_price is not None else None,
            "servings": servings,
            "preferences": preferences,
        },
        "metadata": {
            "confidence": (
                _DEFAULT_PARSE_CONFIDENCE
                if intent != QueryIntent.unsupported
                else _HIGH_CONFIDENCE_UNSUPPORTED
            ),
            "notes": "rule-based fallback parse",
        },
        "filters": {
            "max_price": max_price,
            "min_price": min_price,
            "category": items[0]["category"] if items else None,
            "quantity": f"{quantity}{unit}" if quantity and unit else None,
            "brand": None,
        },
        "intent": intent.value,
    }


class QueryUnderstandingAgent:
    """Converts raw user query into a StructuredQuery.

    Tries LLM first; falls back to rule-based parsing on failure.
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        self._llm = llm_manager

    async def run(self, raw_query: str) -> StructuredQuery:
        if not raw_query or not raw_query.strip():
            raise AgentException("QueryUnderstandingAgent", "Empty query received")

        # Rule-based first: use deterministic parsing as the primary path.
        # Only fall back to LLM for genuinely ambiguous queries where the
        # rule-based parser fails to identify a known product.
        rule_parsed = _rule_based_parse(raw_query)
        llm_called = False

        product = rule_parsed.get("product", "")
        intent = rule_parsed.get("intent", "product_search")
        confidence = rule_parsed.get("metadata", {}).get("confidence", 0.0)

        # If the rule-based parser found a known product with adequate
        # confidence, skip the LLM entirely.
        # Check the extracted product and also its source tokens against
        # the known-products list (rule-based normalization may have
        # rewritten the term, e.g. "milk" → "packaged milk").
        raw_tokens = set(_normalize_text(raw_query).split())
        product_is_known = (
            product in _KNOWN_PRODUCTS
            or bool(raw_tokens & _KNOWN_PRODUCTS)
            or product in _NORMALIZATION_MAP.values()
        )
        needs_llm = (
            intent == "product_search"
            and not product_is_known
            and confidence < 0.8
        )

        if needs_llm:
            prompt = _PROMPT_TEMPLATE.format(query=raw_query.strip())
            try:
                llm_called = True
                parsed = await self._llm.call(prompt, schema_example=_SCHEMA_EXAMPLE)
                logger.debug("[QUERY_AGENT] llm_called=%s raw_output=%s", llm_called, parsed)
                parsed = self._validate_parsed(parsed, raw_query)
                logger.debug("[PARSE_CHECK] success=true reason=")
            except Exception:
                logger.debug("[PARSE_CHECK] success=false reason=llm_or_validation_failed")
                parsed = rule_parsed
        else:
            parsed = rule_parsed

        logger.debug("[QUERY_AGENT] parsed_query=%s llm_called=%s", parsed, llm_called)

        return StructuredQuery(
            product=parsed.get("product", raw_query).strip(),
            filters=QueryFilters(**{
                k: parsed.get("filters", {}).get(k)
                for k in QueryFilters.model_fields
            }),
            intent=QueryIntent(parsed.get("intent", QueryIntent.product_search.value)),
            normalized_query=str(parsed.get("normalized_query") or _normalize_text(raw_query)),
            items=[
                StructuredItem(
                    name=str(item.get("name") or parsed.get("product") or raw_query).strip(),
                    category=str(item.get("category") or "general").strip().lower(),
                    attributes=ItemAttributes(
                        quantity=(
                            float(item.get("attributes", {}).get("quantity"))
                            if item.get("attributes", {}).get("quantity") is not None
                            else None
                        ),
                        unit=(
                            str(item.get("attributes", {}).get("unit")).strip().lower()
                            if item.get("attributes", {}).get("unit") is not None
                            else None
                        ),
                        preferences=[
                            str(p).strip().lower()
                            for p in item.get("attributes", {}).get("preferences", [])
                            if str(p).strip()
                        ],
                    ),
                )
                for item in (parsed.get("items") or [])
            ],
            constraints=QueryConstraints(
                budget=parsed.get("constraints", {}).get("budget"),
                servings=parsed.get("constraints", {}).get("servings"),
                preferences=[
                    str(p).strip().lower()
                    for p in parsed.get("constraints", {}).get("preferences", [])
                    if str(p).strip()
                ],
            ),
            metadata=QueryMetadata(
                confidence=max(0.0, min(1.0, float(parsed.get("metadata", {}).get("confidence", 0.0)))),
                notes=str(parsed.get("metadata", {}).get("notes", "")).strip(),
            ),
            raw_query=raw_query,
        )

    @staticmethod
    def _validate_parsed(parsed: Dict[str, Any], raw_query: str) -> Dict[str, Any]:
        """Ensure required fields exist and are valid."""
        if not isinstance(parsed.get("product"), str) or not parsed["product"]:
            parsed["product"] = raw_query.split()[0] if raw_query.split() else raw_query
        valid_intents = {i.value for i in QueryIntent}
        if parsed.get("intent") not in valid_intents:
            parsed["intent"] = QueryIntent.product_search.value
        if "filters" not in parsed or not isinstance(parsed["filters"], dict):
            parsed["filters"] = {}
        if "normalized_query" not in parsed or not isinstance(parsed["normalized_query"], str):
            parsed["normalized_query"] = _normalize_text(raw_query)
        if "items" not in parsed or not isinstance(parsed["items"], list):
            parsed["items"] = [{"name": parsed["product"], "category": "general", "attributes": {"quantity": None, "unit": None, "preferences": []}}]
        if not parsed["items"] and parsed.get("product"):
            parsed["items"] = [{"name": parsed["product"], "category": "general", "attributes": {"quantity": None, "unit": None, "preferences": []}}]
        if "constraints" not in parsed or not isinstance(parsed["constraints"], dict):
            parsed["constraints"] = {"budget": None, "servings": None, "preferences": []}
        if "metadata" not in parsed or not isinstance(parsed["metadata"], dict):
            parsed["metadata"] = {"confidence": 0.0, "notes": ""}
        return parsed
