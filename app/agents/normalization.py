"""Normalization Agent.

Converts raw/extracted user terms into canonical grocery intents and variants
using a rule-based-first strategy, with LLM fallback only for unknown terms.

An in-memory cache (``_normalization_cache``) prevents repeated LLM calls
for the same term within the process lifetime.
"""

import logging
from typing import Any, Dict, List, Mapping

from app.agents.base_execution import BaseExecutionAgent
from app.agents.synonym_memory import SynonymMemoryAgent
from app.data.models import NormalizedEntities, NormalizedEntity, NormalizedItem, RawEntities
from app.llm.manager import LLMManager

logger = logging.getLogger(__name__)

# In-memory cache: term → NormalizedItem (avoids repeated LLM calls)
_normalization_cache: Dict[str, NormalizedItem] = {}

_SCHEMA_EXAMPLE = """{
  "canonical_name": "paneer",
  "possible_variants": ["paneer cubes", "fresh paneer"],
  "category": "dairy",
  "attributes": ["fresh"]
}"""

_PROMPT_TEMPLATE = """
Normalize this grocery search intent into a canonical searchable item.

Input: "{term}"

Return JSON with:
- canonical_name: standardized grocery term
- possible_variants: list of search variants/synonyms/regional names
- category: grocery category (vegetable, dairy, staples, snacks, etc.)
- attributes: optional descriptors (fresh, frozen, branded, etc.)

Rules:
- Capture meaning, not exact words.
- Handle vague intent by choosing practical purchasable grocery target.
- Keep canonical_name concise and searchable.
"""

_MAYONNAISE_VARIANTS = ["mayo", "mayonnaise", "veg mayo", "eggless mayo"]

_SAFE_FALLBACKS: Dict[str, Dict[str, Any]] = {
    "milk": {
        "canonical_name": "packaged milk",
        "possible_variants": ["milk", "packaged milk", "toned milk", "full cream milk"],
        "category": "dairy",
        "attributes": [],
    },
    "capsicum": {
        "canonical_name": "capsicum",
        "possible_variants": ["green capsicum", "shimla mirch"],
        "category": "vegetable",
        "attributes": ["fresh"],
    },
    "atta": {
        "canonical_name": "wheat flour",
        "possible_variants": ["atta", "whole wheat atta"],
        "category": "staples",
        "attributes": [],
    },
    "jeera": {
        "canonical_name": "cumin seeds",
        "possible_variants": ["jeera", "cumin", "cumin seeds"],
        "category": "staples",
        "attributes": [],
    },
    "mirchi powder": {
        "canonical_name": "red chili powder",
        "possible_variants": ["mirchi powder", "red chilli powder", "red chili powder"],
        "category": "staples",
        "attributes": [],
    },
    "dal": {
        "canonical_name": "lentils",
        "possible_variants": ["dal", "lentils"],
        "category": "staples",
        "attributes": [],
    },
    "ginger piece": {
        "canonical_name": "ginger",
        "possible_variants": ["ginger piece", "ginger"],
        "category": "vegetable",
        "attributes": ["fresh"],
    },
    "paneer cubes": {
        "canonical_name": "paneer",
        "possible_variants": ["paneer cubes", "fresh paneer"],
        "category": "dairy",
        "attributes": ["fresh"],
    },
    "salad leaves": {
        "canonical_name": "salad",
        "possible_variants": ["salad leaves", "lettuce", "spinach", "cucumber", "tomato"],
        "category": "vegetable",
        "attributes": ["fresh"],
    },
    "something for evening snacks": {
        "canonical_name": "snacks",
        "possible_variants": ["chips", "biscuits", "namkeen"],
        "category": "snacks",
        "attributes": [],
    },
    "mayo": {
        "canonical_name": "mayonnaise",
        "possible_variants": _MAYONNAISE_VARIANTS,
        "category": "condiments",
        "attributes": [],
    },
    "mayonnaise": {
        "canonical_name": "mayonnaise",
        "possible_variants": _MAYONNAISE_VARIANTS,
        "category": "condiments",
        "attributes": [],
    },
}

_KEYWORD_FALLBACKS: Dict[str, Dict[str, Any]] = {
    "milk": {"canonical_name": "milk", "possible_variants": ["milk"], "category": "dairy", "attributes": []},
    "bread": {"canonical_name": "bread", "possible_variants": ["bread"], "category": "snacks", "attributes": []},
    "eggs": {"canonical_name": "eggs", "possible_variants": ["eggs"], "category": "poultry", "attributes": []},
    "rice": {"canonical_name": "rice", "possible_variants": ["rice"], "category": "staples", "attributes": []},
    "tomato": {"canonical_name": "tomato", "possible_variants": ["tomato"], "category": "vegetable", "attributes": ["fresh"]},
    "onion": {"canonical_name": "onion", "possible_variants": ["onion"], "category": "vegetable", "attributes": ["fresh"]},
    "paneer": {"canonical_name": "paneer", "possible_variants": ["paneer", "paneer cubes"], "category": "dairy", "attributes": ["fresh"]},
    "capsicum": {"canonical_name": "capsicum", "possible_variants": ["capsicum", "shimla mirch"], "category": "vegetable", "attributes": ["fresh"]},
    "snack": {"canonical_name": "snacks", "possible_variants": ["chips", "biscuits", "namkeen"], "category": "snacks", "attributes": []},
    "salad": {"canonical_name": "salad", "possible_variants": ["salad leaves", "lettuce"], "category": "vegetable", "attributes": ["fresh"]},
    "cucumber": {"canonical_name": "cucumber", "possible_variants": ["cucumber", "kheera"], "category": "vegetable", "attributes": ["fresh"]},
    "garlic": {"canonical_name": "garlic", "possible_variants": ["garlic", "lahsun"], "category": "vegetable", "attributes": ["fresh"]},
    "mayo": {"canonical_name": "mayonnaise", "possible_variants": _MAYONNAISE_VARIANTS, "category": "condiments", "attributes": []},
    "mayonnaise": {"canonical_name": "mayonnaise", "possible_variants": _MAYONNAISE_VARIANTS, "category": "condiments", "attributes": []},
}
_HIGH_NORMALIZATION_CONFIDENCE = 0.9
_LOW_NORMALIZATION_CONFIDENCE = 0.65
_UNRESOLVED_CONFIDENCE_THRESHOLD = 0.7


def _fallback_normalization(term: str) -> Dict[str, Any]:
    key = term.strip().lower()
    if key in _SAFE_FALLBACKS:
        return _SAFE_FALLBACKS[key]
    for keyword, mapped in _KEYWORD_FALLBACKS.items():
        if keyword in key:
            return mapped
    return {
        "canonical_name": key,
        "possible_variants": [key],
        "category": "general",
        "attributes": [],
    }


class NormalizationAgent(BaseExecutionAgent):
    """LLM-first normalizer for open-ended grocery terms."""

    def __init__(self, llm_manager: LLMManager, synonym_memory: SynonymMemoryAgent | None = None) -> None:
        self._llm = llm_manager
        self._synonym_memory = synonym_memory or SynonymMemoryAgent()

    async def run(self, term: str) -> NormalizedItem:
        cache_key = term.strip().lower()
        if cache_key in _normalization_cache:
            logger.debug("[NORMALIZATION] input=%s cache=hit", term)
            return _normalization_cache[cache_key]

        prompt = _PROMPT_TEMPLATE.format(term=term.strip())
        remembered = await self._synonym_memory.lookup(term)
        if remembered:
            raw_output = _fallback_normalization(remembered)
        else:
            # Rule-based first: check safe fallbacks before calling LLM
            rule_result = _fallback_normalization(term)
            if rule_result["category"] != "general":
                # Known term — skip LLM entirely
                raw_output = rule_result
            else:
                # Unknown term — try LLM, fall back to rule result
                raw_output: Dict[str, Any]
                try:
                    raw_output = await self._llm.call(prompt, schema_example=_SCHEMA_EXAMPLE)
                    logger.debug("[NORMALIZATION] input=%s llm_output=%s", term, raw_output)
                except Exception:
                    logger.debug("[NORMALIZATION] input=%s error=llm_failed_using_fallback", term)
                    raw_output = rule_result

        canonical = str(raw_output.get("canonical_name") or term).strip().lower()
        variants = raw_output.get("possible_variants") or []
        if not isinstance(variants, list):
            variants = [str(variants)]
        variants = [str(v).strip().lower() for v in variants if str(v).strip()]
        if canonical not in variants:
            variants.insert(0, canonical)

        category = raw_output.get("category")
        attrs = raw_output.get("attributes") or []
        if not isinstance(attrs, list):
            attrs = [str(attrs)]
        attrs = [str(a).strip().lower() for a in attrs if str(a).strip()]

        item = NormalizedItem(
            canonical_name=canonical,
            possible_variants=variants,
            category=str(category).strip().lower() if category else None,
            attributes=attrs,
        )
        logger.debug(
            '[NORMALIZATION] input="%s" canonical="%s" variants=%s category="%s"',
            term,
            item.canonical_name,
            item.possible_variants,
            item.category or "",
        )
        await self._synonym_memory.remember(term, item.canonical_name)
        _normalization_cache[cache_key] = item
        return item

    async def run_entities(self, raw_entities: RawEntities) -> NormalizedEntities:
        normalized: List[NormalizedEntity] = []
        unresolved: List[str] = []
        candidate_terms = [e.text for e in raw_entities.entities] + list(raw_entities.candidate_entities)
        deduped_terms = list(dict.fromkeys(t.strip() for t in candidate_terms if t and t.strip()))
        for term in deduped_terms:
            entity_text = term
            item = await self.run(entity_text)
            aliases = await self._synonym_memory.aliases_for(item.canonical_name)
            variants = list(dict.fromkeys(item.possible_variants + aliases))
            confidence = (
                _HIGH_NORMALIZATION_CONFIDENCE
                if item.category and item.category != "general"
                else _LOW_NORMALIZATION_CONFIDENCE
            )
            normalized.append(
                NormalizedEntity(
                    raw_text=entity_text,
                    canonical_name=item.canonical_name,
                    category=item.category,
                    possible_variants=variants,
                    confidence=confidence,
                )
            )
            if confidence < _UNRESOLVED_CONFIDENCE_THRESHOLD:
                unresolved.append(entity_text)
        return NormalizedEntities(entities=normalized, unresolved_entities=unresolved)

    async def act(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        structured_query = state["structured_query"]
        candidates = list(state.get("candidate_entities") or [structured_query.product])
        current_index = min(int(state.get("current_path_index", 0)), max(len(candidates) - 1, 0))
        current_entity = candidates[current_index] if candidates else structured_query.product
        normalized_item = await self.run(current_entity)
        return {
            "current_step": "normalization_node",
            "candidate_entities": candidates,
            "current_path_index": current_index,
            "current_entity": current_entity,
            "selected_path": f"path-{current_index}",
            "normalized_item": normalized_item,
            "unified_product": None,
            "ranking_result": None,
            "ranked_products": None,
            "deal_result": None,
            "deals": None,
            "match_quality": "",
            "tool_request": None,
            "tool_result": None,
            "preliminary_products": [],
            "last_observation": {
                "normalized_entity": normalized_item.canonical_name,
                "candidate_index": current_index,
            },
        }
