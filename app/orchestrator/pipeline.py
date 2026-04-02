"""LangGraph-based Orchestrator.

Controls the full multi-agent pipeline:
  User Query → QueryUnderstandingAgent → (recipe | product pipeline)
            → ProductMatchingAgent → RankingAgent → DealDetectionAgent
            → Response

README
------
Core pipeline engine. Uses a directed graph (LangGraph-style StateGraph) to
manage agent execution flow. Falls back to a simple sequential runner if
LangGraph is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import app.data.layer as data_layer
from app.agents.constraint_extraction import ConstraintExtractionAgent
from app.agents.constraint_optimizer import ConstraintOptimizerAgent
from app.agents.deal_detection import DealDetectionAgent
from app.agents.domain_guard import DomainGuardAgent
from app.agents.ambiguity_reasoning import AmbiguityReasoningAgent
from app.agents.evaluation import EvaluationAgent
from app.agents.entity_extraction import EntityExtractionAgent
from app.agents.execution_planner import ExecutionPlannerAgent
from app.agents.intent_detection import IntentDetectionAgent
from app.agents.language_processing import LanguageProcessingAgent
from app.agents.fallback import FallbackAgent
from app.agents.normalization import NormalizationAgent
from app.agents.output_formatter import OutputFormatterAgent
from app.agents.product_matching import ProductMatchingAgent
from app.agents.query_logging import QueryLoggingAgent
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.ranking import RankingAgent
from app.agents.recipe import RecipeAgent
from app.agents.synonym_memory import SynonymMemoryAgent
from app.agents.user_context import UserContextAgent
from app.data.layer import get_products_for_entity
from app.data.models import (
    CartItem,
    CartOptimizationResult,
    CartPlatformGroup,
    FailurePolicy,
    FinalStructuredQuery,
    FinalResponse,
    NormalizedItem,
    LearningSignals,
    PlatformEvent,
    PlatformEventType,
    QueryFilters,
    QueryIntent,
    QueryMetadata,
    QueryConstraints,
    RecipeResult,
    StructuredItem,
    StructuredQuery,
    UnifiedProduct,
)
from app.coordination.network import get_coordination_network
from app.events.platform_events import get_platform_event_intelligence
from app.memory.shared import get_shared_memory
from app.learning.feedback import LearningLoop
from app.llm.manager import LLMManager, get_llm_manager
from app.orchestrator.search_graph import build_search_execution_graph
from app.orchestrator.state import SearchGraphState
from app.response.builder import ResponseBuilder

logger = logging.getLogger(__name__)
# The LangGraph enrichment loop is intentionally capped at two retries:
# one expansion pass for close variants and one broader fallback pass.
# This replaces the older broader loop budget of three because controller-led
# execution now separates tool enrichment from candidate enrichment, so an
# extra candidate retry no longer improves coverage meaningfully.
_MAX_ENRICHMENT_RETRY_ATTEMPTS = 2
_MAX_CANDIDATE_ENTITIES = 3
_MIN_OPTIMIZATION_SCORE = 0.2
_GLOBAL_OPTIMIZATION_DELIVERY_WEIGHT = 0.2
_DEFAULT_DELIVERY_MINUTES = 30.0
_MAX_DELIVERY_MINUTES = 90.0
_MATCHING_QUALITY_THRESHOLD = 0.45
_MIN_HIGH_QUALITY_RESULTS = 2


class AgentPipeline:
    """Sequential multi-agent pipeline (LangGraph-compatible design).

    Each step receives the accumulated state dict and returns updated state.
    This mirrors LangGraph's StateGraph pattern and can be trivially wrapped
    in a real LangGraph graph if the package is available.
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None, use_collaborative_llm: Optional[bool] = None) -> None:
        llm = llm_manager or get_llm_manager()
        self._llm_manager = llm
        self._synonym_memory = SynonymMemoryAgent()
        self._language_agent = LanguageProcessingAgent()
        self._intent_agent = IntentDetectionAgent()
        self._entity_agent = EntityExtractionAgent()
        self._constraint_agent = ConstraintExtractionAgent()
        self._domain_guard_agent = DomainGuardAgent()
        self._ambiguity_agent = AmbiguityReasoningAgent()
        self._execution_planner_agent = ExecutionPlannerAgent()
        self._evaluation_agent = EvaluationAgent()
        self._user_context_agent = UserContextAgent()
        self._constraint_optimizer = ConstraintOptimizerAgent()
        self._fallback_agent = FallbackAgent()
        self._formatter_agent = OutputFormatterAgent()
        self._query_logger = QueryLoggingAgent()
        self._query_agent = QueryUnderstandingAgent(llm)
        self._product_agent = ProductMatchingAgent()
        self._normalization_agent = NormalizationAgent(llm, synonym_memory=self._synonym_memory)
        self._ranking_agent = RankingAgent()
        self._deal_agent = DealDetectionAgent()
        self._recipe_agent = RecipeAgent(llm)
        self._builder = ResponseBuilder()
        self._learning_loop = LearningLoop(self._synonym_memory)
        self._platform_events = get_platform_event_intelligence()
        self._shared_memory = get_shared_memory()
        self._coordination = get_coordination_network()
        self._matching_quality_threshold = _MATCHING_QUALITY_THRESHOLD
        self._min_high_quality_results = _MIN_HIGH_QUALITY_RESULTS
        self._max_enrichment_retry_attempts = _MAX_ENRICHMENT_RETRY_ATTEMPTS
        from app.core.config import get_settings
        self._use_collaborative_llm = (
            use_collaborative_llm
            if use_collaborative_llm is not None
            else get_settings().llm_collaborative_controller
        )
        self._search_graph = build_search_execution_graph(self)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def parse_query(self, query: str) -> FinalStructuredQuery:
        """Build fully structured query before execution layer."""
        state: Dict[str, Any] = {"raw_query": query}
        clean_query = await self._language_agent.run(query)
        await self._query_logger.run("language_processing", clean_query.model_dump())
        intent_result = await self._intent_agent.run(clean_query)
        await self._query_logger.run("intent_detection", intent_result.model_dump())
        raw_entities = await self._entity_agent.run(clean_query, intent_result)
        await self._query_logger.run("entity_extraction", raw_entities.model_dump())
        normalized_entities = await self._normalization_agent.run_entities(raw_entities)
        await self._query_logger.run("normalization", normalized_entities.model_dump())
        constraints = await self._constraint_agent.run(clean_query)
        await self._query_logger.run("constraint_extraction", constraints.model_dump())
        domain_guard = await self._domain_guard_agent.run(clean_query, intent_result)
        await self._query_logger.run("domain_guard", domain_guard.model_dump())
        ambiguity = await self._ambiguity_agent.run(intent_result, raw_entities, normalized_entities)
        await self._query_logger.run("ambiguity_reasoning", ambiguity.model_dump())
        fallback = await self._fallback_agent.run(normalized_entities, intent_result.intent)
        await self._query_logger.run("fallback", fallback.model_dump())
        user_context = await self._user_context_agent.run(clean_query)
        await self._query_logger.run("user_context", user_context.model_dump())
        execution_plan, execution_graph, candidate_paths = await self._execution_planner_agent.run(
            intent_result=intent_result,
            constraints=constraints,
            user_context=user_context,
            candidate_entities=ambiguity.candidate_entities or raw_entities.candidate_entities,
        )
        await self._query_logger.run("execution_planning", execution_plan.model_dump())
        ranking_adjustments = self._constraint_optimizer.derive_weights(
            constraints.ranking_preference_weights,
            constraints.preferences,
            user_context.preferences,
        )
        constraints.ranking_preference_weights = ranking_adjustments
        failure_policies = [
            FailurePolicy(failure_type="vague_query", action="branch_candidates", retries_allowed=2),
            FailurePolicy(failure_type="constraint_violation", action="rebalance_budget", retries_allowed=2),
            FailurePolicy(failure_type="missing_entities", action="fallback_search", retries_allowed=1),
            FailurePolicy(failure_type="external_failure", action="graceful_degrade", retries_allowed=1),
        ]
        learning_signals = LearningSignals(
            normalization_reinforced=[e.canonical_name for e in normalized_entities.entities if e.canonical_name],
            failed_matches=list(normalized_entities.unresolved_entities),
            ranking_adjustments=ranking_adjustments,
            constraint_violations=list(constraints.conflict_notes),
            evaluation_notes=[],
            retry_count=0,
        )

        primary_normalized_entity = normalized_entities.entities[0] if normalized_entities.entities else None
        primary_entity = primary_normalized_entity.canonical_name if primary_normalized_entity else ""
        category = primary_normalized_entity.category if primary_normalized_entity else "general"
        structured_query = StructuredQuery(
            product=primary_entity or clean_query.normalized_text or query,
            filters=QueryFilters(
                max_price=(constraints.budget or {}).get("amount") if constraints.budget else None,
                category=category,
            ),
            intent=intent_result.intent,
            normalized_query=clean_query.normalized_text,
            items=[
                StructuredItem(
                    name=primary_entity or "grocery items",
                    category=category or "general",
                    attributes={"preferences": constraints.preferences},
                )
            ],
            constraints=QueryConstraints(
                budget=constraints.budget,
                servings=constraints.servings,
                preferences=constraints.preferences,
            ),
            metadata=QueryMetadata(confidence=intent_result.confidence, notes=intent_result.notes),
            raw_query=query,
        )
        final_structured = await self._formatter_agent.run(
            clean_query=clean_query,
            intent_result=intent_result,
            raw_entities=raw_entities,
            normalized_entities=normalized_entities,
            constraints=constraints,
            domain_guard=domain_guard,
            ambiguity=ambiguity,
            fallback=fallback,
            execution_plan=execution_plan,
            execution_graph=execution_graph,
            candidate_paths=candidate_paths,
            user_context=user_context,
            learning_signals=learning_signals,
            evaluation_history=[],
            failure_policies=failure_policies,
            structured_query=structured_query,
        )
        strategy = await self._shared_memory.get_strategy("recommendation_signals")
        final_structured.platform_signals = {
            "recommendation_signals": strategy.get("recommendation_signals", {}),
            "analytics_signals": strategy.get("analytics_signals", {}),
            "forecast_signals": strategy.get("forecast_signals", {}),
        }
        if strategy.get("recommendation_signals", {}).get("preferences"):
            for pref in strategy["recommendation_signals"]["preferences"]:
                if pref not in final_structured.user_context.preferences:
                    final_structured.user_context.preferences.append(pref)
        if strategy.get("forecast_signals", {}).get("predicted_needs"):
            for need in strategy["forecast_signals"]["predicted_needs"]:
                if need not in final_structured.user_context.predicted_needs:
                    final_structured.user_context.predicted_needs.append(need)
        self._coordination.share("planner", "ranking", "ranking_preferences", constraints.ranking_preference_weights)
        final_structured.coordination_trace = self._coordination.trace()
        await self._query_logger.run("output_formatter", final_structured.model_dump())
        await self._learning_loop.learn_from_success(final_structured)
        state["final_structured_query"] = final_structured
        return final_structured

    async def run_search(self, final_structured: FinalStructuredQuery) -> FinalResponse:
        """Execute search only from finalized structured intelligence."""
        logger.debug("[ENTRY] endpoint=/search type=search_execution_structured")
        state: SearchGraphState = {
            "user_query": final_structured.clean_query.text,
            "final_structured_query": final_structured,
            "structured_query": final_structured.structured_query,
        }

        if not final_structured.domain_guard.allowed:
            return self._builder.build_domain_guard_response(
                {
                    "raw_query": final_structured.clean_query.text,
                    "user_query": state["user_query"],
                    "final_structured_query": final_structured,
                    "structured_query": final_structured.structured_query,
                }
            )

        sq: StructuredQuery = state["structured_query"]
        if sq.intent == QueryIntent.unsupported:
            return self._builder.build_unsupported_response(
                {
                    "raw_query": final_structured.clean_query.text,
                    "user_query": state["user_query"],
                    "structured_query": final_structured.structured_query,
                }
            )
        if any(n.operation == "recipe_generation" for n in final_structured.execution_graph.nodes):
            response = await self.run_recipe(final_structured.clean_query.text)
            if any(n.operation == "cart_optimization" for n in final_structured.execution_graph.nodes):
                response.metadata["secondary_intents"] = [i.value for i in final_structured.intent_result.secondary_intents]
                response.metadata["execution_plan"] = final_structured.execution_plan.model_dump()
            return response
        policy = await self._learning_loop.load_policy(final_structured.clean_query.normalized_text)
        market_signals = await self._shared_memory.get_strategy("market_signals")
        self._coordination.share("memory", "matching", "market_signals", market_signals)
        ranking_boost = self._coordination.request("ranking", "ranking_preferences", {})
        if policy.get("ranking_adjustments"):
            final_structured.constraints.ranking_preference_weights = self._constraint_optimizer.derive_weights(
                policy.get("ranking_adjustments", {}),
                final_structured.constraints.preferences,
                final_structured.user_context.preferences,
            )
        if ranking_boost:
            merged = dict(final_structured.constraints.ranking_preference_weights)
            merged.update({k: float(v) for k, v in ranking_boost.items()})
            final_structured.constraints.ranking_preference_weights = merged
        state.update(
            {
                "current_step": "run_search_entry",
                "next_action": "controller_node",
                "candidate_entities": self._build_candidate_entities(final_structured),
                "current_path_index": 0,
                "retry_count": 0,
                "path_history": [],
                "decision_trace": [],
                "tool_trace": [],
                "collaborative_proposals": [],
                "collaborative_critiques": [],
                "synthesis_trace": {},
                "tool_request": None,
                "tool_result": None,
                "preliminary_products": [],
                "market_signals": market_signals if isinstance(market_signals, dict) else {},
                "ranking_preferences": dict(final_structured.constraints.ranking_preference_weights),
                "budget_limit": (final_structured.constraints.budget or {}).get("amount"),
                "max_retries": self._max_enrichment_retry_attempts,
            }
        )
        result_state = await self._search_graph.ainvoke(state)
        updated_final_structured = result_state.get("final_structured_query")
        if updated_final_structured is not None:
            for field_name in updated_final_structured.__class__.model_fields:
                setattr(final_structured, field_name, getattr(updated_final_structured, field_name))
        response = result_state.get("response") or self._builder.build_search_response(
            {
                "raw_query": final_structured.clean_query.text,
                "structured_query": final_structured.structured_query,
                "final_structured_query": final_structured,
            }
        )
        logger.debug(
            "[FINAL_OUTPUT] result_count=%s total_price=%s deals=%s",
            len(response.results),
            response.total_price,
            len(response.deals),
        )
        return response

    async def run_recipe(self, query: str, servings: int = 2) -> FinalResponse:
        """Execute the recipe pipeline."""
        logger.debug("[ENTRY] endpoint=/ai/recipe query=%s type=recipe", query)
        state: Dict[str, Any] = {"raw_query": query}
        state["recipe_result"] = await self._recipe_agent.run(query, servings)
        response = self._builder.build_recipe_response(state)
        logger.debug(
            "[FINAL_OUTPUT] result_count=%s total_price=%s deals=%s",
            len(response.results),
            response.total_price,
            len(response.deals),
        )
        return response

    async def run_cart_optimize(self, items: List[CartItem]) -> FinalResponse:
        """Find the optimal platform split for a list of cart items."""
        logger.debug(
            "[ENTRY] endpoint=/ai/cart-optimize query=%s type=cart_optimize",
            ",".join(item.name for item in items),
        )
        state: Dict[str, Any] = {"cart_items": items}

        # For each item, find globally optimized option across platforms (cost + delivery)
        item_best: Dict[str, Any] = {}
        for item in items:
            normalized_item = await self._normalization_agent.run(item.name)
            products = get_products_for_entity(normalized_item.canonical_name)
            if not products:
                continue
            best = min((p for p in products if p.in_stock), key=self._global_product_objective, default=None)
            if best:
                item_best[item.name] = best

        # Group cheapest items by platform
        platform_groups: Dict[str, CartPlatformGroup] = {}
        total_cost = 0.0
        for item_name, product in item_best.items():
            plat = product.platform
            if plat not in platform_groups:
                platform_groups[plat] = CartPlatformGroup(platform=plat, items=[], subtotal=0.0)
            platform_groups[plat].items.append(product)
            platform_groups[plat].subtotal = round(
                platform_groups[plat].subtotal + product.price, 2
            )
            total_cost += product.price

        # Compare with single-platform cost
        single_platform_costs: Dict[str, float] = {}
        for item in items:
            normalized_item = await self._normalization_agent.run(item.name)
            products = get_products_for_entity(normalized_item.canonical_name)
            for p in products:
                plat = p.platform.value
                single_platform_costs[plat] = single_platform_costs.get(plat, 0) + p.price
        max_single = max(single_platform_costs.values()) if single_platform_costs else total_cost
        savings = round(max_single - total_cost, 2) if max_single > total_cost else 0.0

        result = CartOptimizationResult(
            original_items=items,
            platform_groups=list(platform_groups.values()),
            total_optimized_cost=round(total_cost, 2),
            savings=savings,
        )
        state["cart_result"] = result
        response = self._builder.build_cart_response(state)
        response.metadata["global_optimization"] = {
            "objective": "cost_delivery_availability",
            "delivery_weight": _GLOBAL_OPTIMIZATION_DELIVERY_WEIGHT,
            "stores_considered": list({group.platform.value for group in result.platform_groups}),
        }
        logger.debug(
            "[FINAL_OUTPUT] result_count=%s total_price=%s deals=%s",
            len(response.results),
            response.total_price,
            len(response.deals),
        )
        return response

    @staticmethod
    def _sanitize_candidate_entities(entities: List[str]) -> List[str]:
        return list(dict.fromkeys([e for e in entities if isinstance(e, str) and e.strip()]))[:_MAX_CANDIDATE_ENTITIES]

    def _build_candidate_entities(self, final_structured: FinalStructuredQuery) -> List[str]:
        candidate_entities = [c.entity_candidate for c in final_structured.candidate_paths] or [
            final_structured.structured_query.product
        ]
        if final_structured.normalized_entities.entities:
            candidate_entities = [
                final_structured.normalized_entities.entities[0].canonical_name,
                *candidate_entities,
            ]
        return self._sanitize_candidate_entities(candidate_entities)

    @staticmethod
    def _category_candidates(category: str) -> List[str]:
        return list(data_layer._CATEGORY_TO_ENTITIES.get(category or "", []))

    @staticmethod
    def _apply_market_signal_adjustments(
        unified_product: UnifiedProduct,
        normalized_item: NormalizedItem,
        market_signals: Dict[str, Any],
    ) -> UnifiedProduct:
        live_entity = market_signals.get(normalized_item.canonical_name, {}) if isinstance(market_signals, dict) else {}
        if not live_entity:
            return unified_product
        adjusted = unified_product.model_copy(deep=True)
        for product in adjusted.platforms:
            if product.normalized_name == normalized_item.canonical_name.lower().strip():
                if live_entity.get("in_stock") is False:
                    product.in_stock = False
                if live_entity.get("price") is not None:
                    product.price = min(product.price, float(live_entity["price"]))
        return adjusted

    def _decorate_search_response(
        self,
        response: FinalResponse,
        final_structured: FinalStructuredQuery,
        retry_count: int,
        selected_path: str,
        match_quality: str,
        tool_trace: List[Dict[str, Any]],
        path_history: List[Dict[str, Any]],
        decision_trace: List[Dict[str, Any]],
        collaborative_proposals: List[Dict[str, Any]],
        collaborative_critiques: List[Dict[str, Any]],
        synthesis_trace: Dict[str, Any],
    ) -> None:
        if final_structured.user_context.predicted_needs:
            response.metadata["predicted_needs"] = final_structured.user_context.predicted_needs
        response.metadata["coordination_trace"] = self._coordination.trace()
        response.metadata["platform_signals"] = final_structured.platform_signals
        response.metadata["search_graph"] = {
            "match_quality": match_quality,
            "retry_count": retry_count,
            "selected_path": selected_path,
            "tool_trace": tool_trace,
            "path_history": path_history,
            "decision_trace": decision_trace,
            "collaborative_proposals": collaborative_proposals,
            "collaborative_critiques": collaborative_critiques,
            "synthesis_trace": synthesis_trace,
        }
        budget_limit = (final_structured.constraints.budget or {}).get("amount")
        if budget_limit:
            response.results = [r for r in response.results if float(r.get("price", 0)) <= float(budget_limit)]
            if response.best_option and float(response.best_option.get("price", 0)) > float(budget_limit):
                response.best_option = {}
            if not response.best_option and response.results:
                response.best_option = response.results[0]
            response.total_price = response.best_option.get("price", 0.0) if response.best_option else 0.0
            if not response.results:
                response.metadata["no_results_message"] = "No products found within budget"

    async def _emit_search_event(
        self,
        final_structured: FinalStructuredQuery,
        response: FinalResponse,
        selected_path: str,
    ) -> None:
        await self._platform_events.ingest(
            event=PlatformEvent(
                event_type=PlatformEventType.user_behavior,
                user_id=final_structured.user_context.user_id or "anonymous",
                payload={
                    "action": "search_execute",
                    "query": final_structured.clean_query.normalized_text,
                    "best_option": response.best_option.get("name") if isinstance(response.best_option, dict) else None,
                    "selected_path": selected_path,
                },
            )
        )

    def _apply_budget_optimization(self, ranking_result: Any, budget_limit: Any) -> Any:
        if not budget_limit:
            return ranking_result
        hard_budget_ranked = [
            item for item in ranking_result.ranked_list
            if item.product and item.product.price <= float(budget_limit)
        ]
        ranked = []
        source_ranked = hard_budget_ranked
        source_ranked_keys = {(item.product.platform.value, item.product.product_id) for item in source_ranked}
        for item in ranking_result.ranked_list:
            item_key = (item.product.platform.value, item.product.product_id)
            if item_key not in source_ranked_keys:
                continue
            optimization_score = self._constraint_optimizer.score_candidate(
                item.product,
                float(budget_limit),
            )
            if optimization_score > _MIN_OPTIMIZATION_SCORE and (not hard_budget_ranked or item.product.price <= float(budget_limit)):
                ranked.append(item)
        ranking_result.ranked_list = ranked
        ranking_result.best_option = ranked[0] if ranked else None
        return ranking_result

    @staticmethod
    def _global_product_objective(product: Any) -> float:
        normalized_delivery = (
            min(float(product.delivery_time_minutes or _DEFAULT_DELIVERY_MINUTES), _MAX_DELIVERY_MINUTES)
            / _MAX_DELIVERY_MINUTES
        )
        return product.price + (_GLOBAL_OPTIMIZATION_DELIVERY_WEIGHT * normalized_delivery * product.price)


# Module-level singleton
_pipeline: Optional[AgentPipeline] = None


def get_pipeline() -> AgentPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgentPipeline()
    return _pipeline
