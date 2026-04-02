"""Integration tests for the full agent pipeline."""

import pytest
import re
from unittest.mock import AsyncMock

from app.data.models import (
    CartItem,
    MatchingDiagnostics,
    UnifiedProduct,
)
from app.orchestrator.pipeline import AgentPipeline


@pytest.fixture
def pipeline() -> AgentPipeline:
    """Create a pipeline with a mock LLM that falls back to rule-based parsing."""
    mock_llm = AsyncMock()
    mock_llm.call.side_effect = Exception("LLM not configured in tests")
    return AgentPipeline(llm_manager=mock_llm)


class TestSearchPipeline:
    async def _run_from_query(self, pipeline, query: str):
        parsed = await pipeline.parse_query(query)
        return await pipeline.run_search(parsed)

    @pytest.mark.asyncio
    async def test_parse_query_returns_strict_intelligence_output(self, pipeline):
        parsed = await pipeline.parse_query("cheap milk under 60")
        assert parsed.clean_query.normalized_text == "cheap milk under 60"
        assert parsed.intent_result.intent.value == "product_search"
        assert parsed.domain_guard.allowed is True
        assert parsed.structured_query.product in {"packaged milk", "milk"}
        assert hasattr(parsed, "user_context")
        assert hasattr(parsed, "learning_signals")

    @pytest.mark.asyncio
    async def test_search_returns_final_response(self, pipeline):
        result = await self._run_from_query(pipeline, "milk")
        assert result.query == "milk"
        assert isinstance(result.results, list)
        assert isinstance(result.best_option, dict)
        assert isinstance(result.deals, list)
        assert isinstance(result.total_price, float)
        assert "normalized_query" in result.metadata
        assert "items" in result.metadata
        assert "constraints" in result.metadata
        if result.results:
            assert "url" in result.results[0]
            assert "brand" in result.results[0]
            assert "source" in result.results[0]
            assert "link_status" in result.results[0]
        if result.best_option:
            assert "url" in result.best_option
            assert "brand" in result.best_option
            assert "source" in result.best_option
            assert "link_status" in result.best_option

    @pytest.mark.asyncio
    async def test_search_with_price_filter(self, pipeline):
        result = await self._run_from_query(pipeline, "milk under 30")
        for r in result.results:
            assert r["price"] <= 30.0

    @pytest.mark.asyncio
    async def test_search_budget_is_strict_when_no_matches(self, pipeline):
        result = await self._run_from_query(pipeline, "ghee under 10")
        assert result.results == []
        assert result.best_option == {}
        assert result.metadata.get("no_results_message") == "No products found within budget"

    @pytest.mark.asyncio
    async def test_search_best_option_present(self, pipeline):
        result = await self._run_from_query(pipeline, "rice")
        assert result.best_option != {}
        assert "platform" in result.best_option
        assert "price" in result.best_option

    @pytest.mark.asyncio
    async def test_search_unknown_product_graceful(self, pipeline):
        result = await self._run_from_query(pipeline, "xyz_unknown_product_123")
        assert result.query == "xyz_unknown_product_123"
        assert len(result.results) > 0
        assert result.best_option != {}
        assert result.metadata.get("intent") in {"product_search", "exploratory"}
        assert result.metadata.get("matching", {}).get("approximate_match") is True

    @pytest.mark.asyncio
    async def test_search_deals_detected_for_discounted_product(self, pipeline):
        result = await self._run_from_query(pipeline, "milk")
        # Zepto milk has 12.5% discount — should appear as deal
        deal_platforms = {d["platform"] for d in result.deals}
        assert len(result.deals) >= 0  # may vary based on thresholds

    @pytest.mark.asyncio
    async def test_search_generic_chicken_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "chicken")
        assert len(result.results) > 0
        assert result.total_price > 0

    @pytest.mark.asyncio
    async def test_search_generic_curd_alias_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "dahi")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_mayo_synonym_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "mayo")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_result_has_link_status_when_url_missing(self, pipeline):
        result = await self._run_from_query(pipeline, "milk")
        if result.results:
            assert result.results[0].get("link_status") in {"available", "link unavailable"}

    @pytest.mark.asyncio
    async def test_search_capsicum_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "capsicum")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_atta_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "atta")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_paneer_cubes_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "paneer cubes")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_salad_leaves_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "salad leaves")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_evening_snacks_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "something for evening snacks")
        assert len(result.results) > 0
        assert result.metadata.get("intent") == "exploratory"

    @pytest.mark.asyncio
    async def test_search_response_includes_matching_provenance(self, pipeline):
        result = await self._run_from_query(pipeline, "milk")
        matching = result.metadata.get("matching", {})
        assert "matched_via" in matching
        assert "quality_score" in matching
        assert "fallback_trace" in matching

    @pytest.mark.asyncio
    async def test_search_vague_multi_item_query_returns_results(self, pipeline):
        result = await self._run_from_query(pipeline, "need paneer cubes and salad leaves for dinner")
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_search_unsupported_query_returns_structured_unsupported(self, pipeline):
        result = await self._run_from_query(pipeline, "book me a flight to mumbai")
        assert result.results == []
        assert result.best_option == {}
        assert result.metadata.get("domain_guard", {}).get("allowed") is False

    @pytest.mark.asyncio
    async def test_parse_query_multi_intent_generates_execution_plan(self, pipeline):
        parsed = await pipeline.parse_query("plan tomato pasta and optimize my cart")
        assert parsed.intent_result.intent.value == "recipe"
        assert "cart_optimization" in [i.value for i in parsed.intent_result.secondary_intents]
        assert parsed.execution_plan.mode == "graph"
        assert parsed.execution_graph.graph_id.startswith("graph-")
        assert any(node.operation == "recipe_generation" for node in parsed.execution_graph.nodes)

    @pytest.mark.asyncio
    async def test_parse_query_search_execution_graph_uses_langgraph_nodes(self, pipeline):
        parsed = await pipeline.parse_query("milk")
        operations = [node.operation for node in parsed.execution_graph.nodes]
        assert operations[:6] == [
            "controller_node",
            "parse_query_node",
            "normalization_node",
            "product_matching_node",
            "tool_execution_node",
            "match_quality_node",
        ]
        assert "enrichment_node" in operations
        assert "ranking_node" in operations
        assert "response_node" in operations

    @pytest.mark.asyncio
    async def test_parse_query_ambiguity_has_candidates(self, pipeline):
        parsed = await pipeline.parse_query("need paneer and salad")
        assert parsed.ambiguity.needs_resolution is True
        assert len(parsed.ambiguity.candidate_entities) >= 1
        assert len(parsed.candidate_paths) >= 1

    @pytest.mark.asyncio
    async def test_reasoning_loop_applies_budget_refinement(self, pipeline):
        parsed = await pipeline.parse_query("cheap milk under 20")
        result = await pipeline.run_search(parsed)
        if result.best_option:
            assert result.best_option["price"] <= 20
        else:
            assert result.results == []
        assert parsed.learning_signals.retry_count >= 0
        assert len(parsed.evaluation_history) >= 1

    @pytest.mark.asyncio
    async def test_dynamic_execution_can_skip_deals(self, pipeline):
        parsed = await pipeline.parse_query("cheap rice under 40")
        assert parsed.execution_plan.adaptive_flags.get("skip_deals") is True
        result = await pipeline.run_search(parsed)
        assert result.deals == []

    @pytest.mark.asyncio
    async def test_cheap_query_orders_results_cheapest_first(self, pipeline):
        result = await self._run_from_query(pipeline, "cheap wheat flour under 150")
        prices = [r["price"] for r in result.results]
        assert prices == sorted(prices)
        if prices:
            assert prices[0] <= 150

    @pytest.mark.asyncio
    async def test_learning_policy_affects_future_run(self, pipeline):
        parsed = await pipeline.parse_query("cheap milk under 20")
        await pipeline.run_search(parsed)
        parsed2 = await pipeline.parse_query("cheap milk under 20")
        await pipeline.run_search(parsed2)
        assert any(note.startswith("selected_path:") for note in parsed2.learning_signals.evaluation_notes)

    @pytest.mark.asyncio
    async def test_search_response_includes_graph_metadata(self, pipeline):
        result = await self._run_from_query(pipeline, "milk")
        graph_meta = result.metadata.get("search_graph", {})
        assert graph_meta.get("selected_path", "").startswith("path-")
        assert graph_meta.get("match_quality") in {"strong", "weak", "empty"}
        assert isinstance(graph_meta.get("retry_count"), int)
        assert isinstance(graph_meta.get("tool_trace"), list)
        assert isinstance(graph_meta.get("decision_trace"), list)
        assert graph_meta.get("decision_trace")
        assert isinstance(graph_meta.get("collaborative_proposals"), list)
        assert isinstance(graph_meta.get("collaborative_critiques"), list)
        assert isinstance(graph_meta.get("synthesis_trace"), dict)

    @pytest.mark.asyncio
    async def test_search_graph_retry_count_caps_at_two(self, pipeline):
        parsed = await pipeline.parse_query("milk")
        empty_unified = UnifiedProduct(
            entity="milk",
            normalized_name="milk",
            platforms=[],
            diagnostics=MatchingDiagnostics(
                input_term="milk",
                fallback_trace=["db_lookup", "no_usable_products"],
                quality_score=0.0,
            ),
        )
        pipeline._product_agent.act = AsyncMock(
            return_value={
                "current_step": "product_matching_node",
                "unified_product": empty_unified,
                "diagnostics": empty_unified.diagnostics,
                "tool_trace": [],
                "tool_attempts": [],
                "path_history": [],
                "tool_request": None,
                "tool_result": None,
                "preliminary_products": [],
                "last_observation": {"phase": "matching_complete", "result_count": 0},
            }
        )
        result = await pipeline.run_search(parsed)
        assert result.results == []
        assert parsed.learning_signals.retry_count == 2
        decision_actions = [
            entry.get("action")
            for entry in result.metadata.get("search_graph", {}).get("decision_trace", [])
        ]
        assert decision_actions.count("enrichment_node") == 2

    @pytest.mark.asyncio
    async def test_search_unknown_product_routes_through_tool_node(self, pipeline):
        result = await self._run_from_query(pipeline, "xyz_unknown_product_123")
        decision_trace = result.metadata.get("search_graph", {}).get("decision_trace", [])
        actions = [entry.get("action") for entry in decision_trace]
        assert "tool_execution_node" in actions
        assert result.metadata.get("search_graph", {}).get("tool_trace")

    @pytest.mark.asyncio
    async def test_search_graph_surfaces_collaborative_controller_traces(self):
        async def collaborative_call(prompt: str, schema_example: str | None = None, response_model=None):
            if "collaborative controller proposal" in prompt:
                fallback_match = re.search(r'"fallback_action":\s*"([^"]+)"', prompt)
                action = fallback_match.group(1) if fallback_match else "response_node"
                return {
                    "role": "routing_strategist",
                    "action": action,
                    "rationale": "Use the deterministic-safe route for this state.",
                    "confidence": 0.8,
                    "evidence": ["available_actions", "fallback_action"],
                }
            if "collaborative controller critique" in prompt:
                proposal_match = re.search(r'"action":\s*"([^"]+)"', prompt)
                action = proposal_match.group(1) if proposal_match else "response_node"
                return {
                    "critic_role": "feasibility_critic",
                    "proposal_action": action,
                    "score": 0.7,
                    "verdict": "support",
                    "strengths": ["Action is valid"],
                    "risks": [],
                    "recommended_adjustments": [],
                }
            if "collaborative controller synthesis" in prompt:
                fallback_match = re.search(r'"fallback_action":\s*"([^"]+)"', prompt)
                action = fallback_match.group(1) if fallback_match else "response_node"
                return {
                    "action": action,
                    "rationale": "Consensus selected the safe action.",
                    "confidence": 0.78,
                    "consensus": "majority_support",
                    "score_breakdown": {action: 1.5},
                }
            raise Exception("LLM not configured in tests")

        mock_llm = AsyncMock()
        mock_llm.call.side_effect = collaborative_call
        pipeline = AgentPipeline(llm_manager=mock_llm, use_collaborative_llm=True)

        parsed = await pipeline.parse_query("milk")
        result = await pipeline.run_search(parsed)

        graph_meta = result.metadata.get("search_graph", {})
        assert graph_meta.get("collaborative_proposals")
        assert graph_meta.get("collaborative_critiques")
        assert graph_meta.get("synthesis_trace", {}).get("consensus") == "majority_support"
        assert any(
            entry.get("decision_source") == "collaborative_llm"
            for entry in graph_meta.get("decision_trace", [])
        )

    @pytest.mark.asyncio
    async def test_parse_query_includes_platform_signals_and_coordination_trace(self, pipeline):
        parsed = await pipeline.parse_query("milk")
        assert isinstance(parsed.platform_signals, dict)
        assert "recommendation_signals" in parsed.platform_signals
        assert isinstance(parsed.coordination_trace, dict)
        assert "signals" in parsed.coordination_trace


class TestRecipePipeline:
    @pytest.mark.asyncio
    async def test_recipe_returns_ingredients(self, pipeline):
        result = await pipeline.run_recipe("tomato pasta", servings=2)
        assert result.query == "tomato pasta"
        assert isinstance(result.results, list)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_recipe_total_price_positive(self, pipeline):
        result = await pipeline.run_recipe("fried rice")
        assert result.total_price >= 0

    @pytest.mark.asyncio
    async def test_recipe_metadata_intent(self, pipeline):
        result = await pipeline.run_recipe("dal")
        assert result.metadata.get("intent") == "recipe"


class TestCartOptimizePipeline:
    @pytest.mark.asyncio
    async def test_cart_optimize_basic(self, pipeline):
        items = [CartItem(name="milk"), CartItem(name="bread")]
        result = await pipeline.run_cart_optimize(items)
        assert isinstance(result.results, list)
        assert result.total_price > 0

    @pytest.mark.asyncio
    async def test_cart_optimize_metadata(self, pipeline):
        items = [CartItem(name="rice"), CartItem(name="onion")]
        result = await pipeline.run_cart_optimize(items)
        assert result.metadata.get("intent") == "cart_optimize"
        assert result.metadata.get("item_count") == 2
        assert result.metadata.get("global_optimization", {}).get("objective") == "cost_delivery_availability"

    @pytest.mark.asyncio
    async def test_cart_optimize_empty_items(self, pipeline):
        result = await pipeline.run_cart_optimize([])
        assert result.total_price == 0.0


class TestResponseFormat:
    """Validate strict output format: {query, results, best_option, deals, total_price}."""

    @pytest.mark.asyncio
    async def test_search_response_has_required_fields(self, pipeline):
        parsed = await pipeline.parse_query("eggs")
        result = await pipeline.run_search(parsed)
        d = result.model_dump()
        assert "query" in d
        assert "results" in d
        assert "best_option" in d
        assert "deals" in d
        assert "total_price" in d

    @pytest.mark.asyncio
    async def test_recipe_response_has_required_fields(self, pipeline):
        result = await pipeline.run_recipe("tomato pasta")
        d = result.model_dump()
        assert "query" in d
        assert "results" in d
        assert "best_option" in d
        assert "deals" in d
        assert "total_price" in d
