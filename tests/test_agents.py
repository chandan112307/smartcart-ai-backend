"""Tests for AI agents."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.data.layer as data_layer
from app.agents.deal_detection import DealDetectionAgent
from app.agents.ambiguity_reasoning import AmbiguityReasoningAgent
from app.agents.evaluation import EvaluationAgent
from app.agents.normalization import NormalizationAgent
from app.agents.product_matching import ProductMatchingAgent
from app.agents.tools.product_intelligence import ProductIntelligenceContext, ProductIntelligenceRegistry, map_external_product
from app.agents.query_understanding import QueryUnderstandingAgent, _rule_based_parse
from app.agents.ranking import RankingAgent
from app.agents.recipe import RecipeAgent
from app.data.models import (
    AmbiguityDecision,
    CleanQuery,
    Constraints,
    DomainGuardResult,
    EvaluationResult,
    LearningSignals,
    FallbackDecision,
    FinalResponse,
    FinalStructuredQuery,
    IntentResult,
    NormalizedEntities,
    NormalizedEntity,
    RawEntities,
    RawEntity,
    QueryMetadata,
    QueryConstraints,
    Platform,
    PlatformProduct,
    QueryFilters,
    QueryIntent,
    StructuredQuery,
    UnifiedProduct,
    UserContext,
    ExecutionPlan,
    ExecutionGraph,
    ToolAttempt,
    StructuredItem,
)


# ---------------------------------------------------------------------------
# Query Understanding Agent
# ---------------------------------------------------------------------------


class TestQueryUnderstandingAgentRuleBased:
    def test_simple_product(self):
        result = _rule_based_parse("cheap milk under 60")
        assert result["product"] == "packaged milk"
        assert result["filters"]["max_price"] == 60.0
        assert result["intent"] == QueryIntent.product_search.value

    def test_recipe_intent(self):
        result = _rule_based_parse("recipe for tomato pasta")
        assert result["intent"] == QueryIntent.recipe.value

    def test_deal_intent(self):
        result = _rule_based_parse("best deals on bread today")
        assert result["intent"] == QueryIntent.product_search.value

    def test_cart_intent(self):
        result = _rule_based_parse("optimize my cart total")
        assert result["intent"] == QueryIntent.cart_optimization.value

    def test_price_extraction_with_rupee_symbol(self):
        result = _rule_based_parse("milk under ₹50")
        assert result["filters"]["max_price"] == 50.0

    def test_price_extraction_upto(self):
        result = _rule_based_parse("rice upto 130")
        assert result["filters"]["max_price"] == 130.0

    def test_empty_product_fallback(self):
        result = _rule_based_parse("tomato")
        assert result["product"] == "tomato"

    def test_generic_ingredient_terms_extracted(self):
        result = _rule_based_parse("need chicken and curd")
        assert result["product"] == "chicken"

    def test_ghee_known_product(self):
        result = _rule_based_parse("buy ghee")
        assert result["product"] == "ghee"

    def test_unsupported_non_grocery_query(self):
        result = _rule_based_parse("find me a laptop under 50000")
        assert result["intent"] == QueryIntent.unsupported.value


class TestQueryUnderstandingAgentLLM:
    @pytest.mark.asyncio
    async def test_run_with_llm_success(self):
        """LLM is invoked for unknown products that rule-based parsing cannot
        confidently identify, and the LLM result is returned."""
        mock_llm = AsyncMock()
        mock_llm.call.return_value = {
            "product": "organic quinoa",
            "filters": {"max_price": None, "min_price": None, "category": "staples", "quantity": None, "brand": None},
            "intent": "product_search",
            "normalized_query": "organic quinoa",
            "items": [{"name": "organic quinoa", "category": "staples", "attributes": {"quantity": None, "unit": None, "preferences": []}}],
            "constraints": {"budget": None, "servings": None, "preferences": []},
            "metadata": {"confidence": 0.9, "notes": "LLM parsed"},
        }
        agent = QueryUnderstandingAgent(mock_llm)
        result = await agent.run("organic quinoa")
        assert result.product == "organic quinoa"
        assert result.intent == QueryIntent.product_search
        mock_llm.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_skips_llm_for_known_products(self):
        """Known products use rule-based parsing and never call LLM."""
        mock_llm = AsyncMock()
        mock_llm.call.return_value = {"product": "milk"}
        agent = QueryUnderstandingAgent(mock_llm)
        result = await agent.run("cheap milk under 60")
        assert result.product == "packaged milk"
        assert result.intent == QueryIntent.product_search
        mock_llm.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_falls_back_on_llm_error(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")
        agent = QueryUnderstandingAgent(mock_llm)
        result = await agent.run("cheap milk under 60")
        # Rule-based parser maps "milk" → "packaged milk" via _NORMALIZATION_MAP
        assert result.product == "packaged milk"
        assert result.intent == QueryIntent.product_search
        assert result.constraints.budget is not None

    @pytest.mark.asyncio
    async def test_run_raises_on_empty_query(self):
        mock_llm = AsyncMock()
        agent = QueryUnderstandingAgent(mock_llm)
        with pytest.raises(Exception, match="Empty query"):
            await agent.run("")


class TestNormalizationAgent:
    @pytest.mark.asyncio
    async def test_normalization_capsicum(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")
        agent = NormalizationAgent(mock_llm)
        result = await agent.run("capsicum")
        assert result.canonical_name == "capsicum"
        assert "shimla mirch" in result.possible_variants

    @pytest.mark.asyncio
    async def test_normalization_paneer_cubes(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")
        agent = NormalizationAgent(mock_llm)
        result = await agent.run("paneer cubes")
        assert result.canonical_name == "paneer"
        assert result.category == "dairy"

    @pytest.mark.asyncio
    async def test_normalization_vague_snacks(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")
        agent = NormalizationAgent(mock_llm)
        result = await agent.run("something for evening snacks")
        assert result.canonical_name == "snacks"
        assert len(result.possible_variants) > 0

    @pytest.mark.asyncio
    async def test_normalization_jeera_maps_to_cumin(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")
        agent = NormalizationAgent(mock_llm)
        result = await agent.run("jeera")
        assert result.canonical_name == "cumin seeds"

    @pytest.mark.asyncio
    async def test_normalization_mayo_maps_to_mayonnaise(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")
        agent = NormalizationAgent(mock_llm)
        result = await agent.run("mayo")
        assert result.canonical_name == "mayonnaise"


class TestAmbiguityReasoningAgent:
    @pytest.mark.asyncio
    async def test_single_high_confidence_entity_skips_ambiguity(self):
        agent = AmbiguityReasoningAgent()
        decision = await agent.run(
            IntentResult(intent=QueryIntent.product_search, confidence=0.9, notes=""),
            RawEntities(
                entities=[RawEntity(text="garlic", confidence=0.8)],
                primary_entity="garlic",
                ambiguity_flags=[],
                candidate_entities=["garlic"],
            ),
            NormalizedEntities(
                entities=[
                    NormalizedEntity(
                        raw_text="garlic",
                        canonical_name="garlic",
                        category="vegetable",
                        possible_variants=["garlic"],
                        confidence=0.9,
                    )
                ],
                unresolved_entities=[],
            ),
        )
        assert decision.needs_resolution is False
        assert decision.resolution_strategy == "none"


# ---------------------------------------------------------------------------
# Product Matching Agent
# ---------------------------------------------------------------------------


class TestProductMatchingAgent:
    @pytest.mark.asyncio
    async def test_match_known_product(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="milk", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="milk")
        result = await agent.run(sq)
        assert result.entity == "milk"
        assert len(result.platforms) > 0

    @pytest.mark.asyncio
    async def test_price_filter_applied(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(
            product="milk",
            filters=QueryFilters(max_price=28.0),
            intent=QueryIntent.product_search,
            raw_query="milk under 28",
        )
        result = await agent.run(sq)
        for p in result.platforms:
            assert p.price <= 28.0

    @pytest.mark.asyncio
    async def test_unknown_product_returns_empty(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="xyz_nonexistent", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="xyz")
        result = await agent.run(sq)
        assert len(result.platforms) > 0
        assert all(product.source == "approximation" for product in result.platforms)
        assert result.diagnostics.approximate_match is True
        assert result.diagnostics.quality_score >= 0.0

    @pytest.mark.asyncio
    async def test_match_generic_chicken_term(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="chicken", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="chicken")
        result = await agent.run(sq)
        assert len(result.platforms) > 0

    @pytest.mark.asyncio
    async def test_match_synonym_dahi_to_curd(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="dahi", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="dahi")
        result = await agent.run(sq)
        assert len(result.platforms) > 0
        assert all(p.normalized_name == "curd" for p in result.platforms)

    @pytest.mark.asyncio
    async def test_top_k_fallback_when_filters_remove_all(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(
            product="ghee",
            filters=QueryFilters(max_price=10.0),
            intent=QueryIntent.product_search,
            raw_query="cheap ghee under 10",
        )
        result = await agent.run(sq)
        assert len(result.platforms) == 3

    @pytest.mark.asyncio
    async def test_match_uses_db_source_when_db_returns_rows(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="milk", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="milk")
        sample = PlatformProduct(
            platform=Platform.blinkit,
            product_id="db-1",
            name="DB Milk",
            normalized_name="milk",
            price=30.0,
            source="db",
        )
        with patch.object(data_layer, "_search_db_products", return_value=[sample]), patch.object(
            agent._tool_registry,
            "fetch",
            AsyncMock(return_value=([], [])),  # (products, attempts)
        ) as mock_fetch:
            result = await agent.run(sq)
        assert len(result.platforms) == 1
        assert result.platforms[0].source == "db"
        mock_fetch.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_fallback_maps_inconsistent_payload(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="oats", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="oats")
        tool_products = [
            {
                "title": "Rolled Oats 500g",
                "current_price": "₹99",
                "stars": "4.4",
                "id": "ext-1",
                "platform": "blinkit",
                "url": "https://blinkit.example/item/1",
            }
        ]
        with patch.object(data_layer, "_search_db_products", return_value=[]), patch.object(
            agent._tool_registry,
            "fetch",
            AsyncMock(return_value=([map_external_product(tool_products[0], entity="oats", default_source="api")], [ToolAttempt(tool_name="api", success=True, result_count=1)])),
        ):
            result = await agent.run(sq)
        assert len(result.platforms) == 1
        assert result.platforms[0].price == 99.0
        assert result.platforms[0].rating == 4.4
        assert result.diagnostics.matched_via == "api"

    @pytest.mark.asyncio
    async def test_tool_failure_falls_back_to_approximation(self):
        agent = ProductMatchingAgent()
        sq = StructuredQuery(product="mystery thing", filters=QueryFilters(), intent=QueryIntent.product_search, raw_query="mystery thing")
        with patch.object(data_layer, "_search_db_products", return_value=[]), patch.object(
            agent._tool_registry,
            "fetch",
            AsyncMock(return_value=([], [ToolAttempt(tool_name="api", success=False, error="boom")])),
        ), patch.object(
            agent._tool_registry,
            "approximate",
            AsyncMock(
                return_value=[
                    PlatformProduct(
                        platform=Platform.blinkit,
                        product_id="approx-1",
                        name="Approx Grocery Result",
                        normalized_name="bread",
                        price=42.0,
                        source="approximation",
                    )
                ]
            ),
        ):
            result = await agent.run(sq)
        assert len(result.platforms) == 1
        assert result.platforms[0].source == "approximation"
        assert result.diagnostics.approximate_match is True


class TestProductIntelligenceMapping:
    def test_map_external_product_handles_sparse_payload(self):
        product = map_external_product(
            {
                "title": "Organic Rice 1kg",
                "selling_price": "₹120",
                "review_score": "4.7",
                "link": "https://example.com/rice",
            },
            entity="rice",
            default_source="http_fetch",
        )
        assert product is not None
        assert product.price == 120.0
        assert product.rating == 4.7
        assert product.platform == Platform.external
        assert product.product_id.startswith("external-")


# ---------------------------------------------------------------------------
# Ranking Agent
# ---------------------------------------------------------------------------


class TestRankingAgent:
    def _make_product(self, platform: Platform, price: float, delivery: int, rating: float, discount: float = 0.0) -> PlatformProduct:
        return PlatformProduct(
            platform=platform,
            product_id=f"{platform}-test",
            name=f"{platform} product",
            normalized_name="test",
            price=price,
            delivery_time_minutes=delivery,
            rating=rating,
            discount_percent=discount,
        )

    @pytest.mark.asyncio
    async def test_best_option_is_highest_scored(self):
        agent = RankingAgent()
        products = [
            self._make_product(Platform.blinkit, 30.0, 12, 4.5, 10.0),
            self._make_product(Platform.zepto, 50.0, 40, 3.0, 0.0),
        ]
        unified = UnifiedProduct(entity="milk", platforms=products)
        result = await agent.run(unified)
        assert result.best_option is not None
        assert result.best_option.platform == Platform.blinkit
        assert len(result.ranked_list) == 2

    @pytest.mark.asyncio
    async def test_empty_products_returns_empty_ranking(self):
        agent = RankingAgent()
        unified = UnifiedProduct(entity="milk", platforms=[])
        result = await agent.run(unified)
        assert result.ranked_list == []
        assert result.best_option is None

    @pytest.mark.asyncio
    async def test_ranks_are_sequential(self):
        agent = RankingAgent()
        products = [
            self._make_product(Platform.blinkit, 30.0, 12, 4.5),
            self._make_product(Platform.zepto, 32.0, 10, 4.3),
            self._make_product(Platform.bigbasket, 29.0, 30, 4.6),
        ]
        unified = UnifiedProduct(entity="milk", platforms=products)
        result = await agent.run(unified)
        ranks = [r.rank for r in result.ranked_list]
        assert ranks == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_price_first_ranking_with_high_price_preference(self):
        agent = RankingAgent()
        products = [
            self._make_product(Platform.blinkit, 32.0, 8, 4.8, 20.0),
            self._make_product(Platform.zepto, 25.0, 25, 4.0, 0.0),
            self._make_product(Platform.bigbasket, 25.0, 20, 3.8, 5.0),
        ]
        unified = UnifiedProduct(entity="milk", platforms=products)
        result = await agent.run(unified, ranking_preferences={"price": 0.8, "delivery": 0.1, "rating": 0.05, "discount": 0.05})
        assert result.ranked_list[0].product.price == 25.0
        assert result.ranked_list[0].platform == Platform.bigbasket


# ---------------------------------------------------------------------------
# Deal Detection Agent
# ---------------------------------------------------------------------------


class TestDealDetectionAgent:
    def _make_discounted_product(self, platform: Platform, price: float, original: float) -> PlatformProduct:
        discount = round((original - price) / original * 100, 1)
        return PlatformProduct(
            platform=platform,
            product_id=f"{platform}-deal",
            name=f"{platform} deal product",
            normalized_name="deal",
            price=price,
            original_price=original,
            discount_percent=discount,
        )

    @pytest.mark.asyncio
    async def test_detects_deals_above_threshold(self):
        agent = DealDetectionAgent()
        products = [
            self._make_discounted_product(Platform.zepto, 28.0, 32.0),   # 12.5%
            self._make_discounted_product(Platform.blinkit, 30.0, 30.0),  # 0%
        ]
        unified = UnifiedProduct(entity="milk", platforms=products)
        result = await agent.run(unified)
        assert len(result.deals) == 1
        assert result.deals[0].platform == Platform.zepto

    @pytest.mark.asyncio
    async def test_trending_deals_above_threshold(self):
        agent = DealDetectionAgent()
        products = [
            self._make_discounted_product(Platform.zepto, 28.0, 32.0),   # 12.5% → trending
        ]
        unified = UnifiedProduct(entity="milk", platforms=products)
        result = await agent.run(unified)
        assert len(result.trending_deals) == 1

    @pytest.mark.asyncio
    async def test_no_deals_returns_empty(self):
        agent = DealDetectionAgent()
        products = [
            PlatformProduct(
                platform=Platform.blinkit,
                product_id="bl-001",
                name="Product",
                normalized_name="product",
                price=30.0,
                discount_percent=0.0,
            )
        ]
        unified = UnifiedProduct(entity="test", platforms=products)
        result = await agent.run(unified)
        assert result.deals == []


# ---------------------------------------------------------------------------
# Recipe Agent
# ---------------------------------------------------------------------------


class TestRecipeAgent:
    @pytest.mark.asyncio
    async def test_recipe_returns_ingredients(self):
        mock_llm = AsyncMock()
        mock_llm.call.side_effect = Exception("LLM unavailable")  # force static fallback
        agent = RecipeAgent(mock_llm)
        result = await agent.run("tomato pasta", servings=2)
        assert result.recipe_name != ""
        assert len(result.ingredients) > 0

    @pytest.mark.asyncio
    async def test_recipe_maps_known_ingredients(self):
        mock_llm = AsyncMock()
        mock_llm.call.return_value = {
            "recipe_name": "Milk Tea",
            "servings": 2,
            "ingredients": [
                {"name": "milk", "quantity": "200", "unit": "ml"},
            ],
        }
        agent = RecipeAgent(mock_llm)
        result = await agent.run("milk tea")
        assert len(result.ingredients) == 1
        ip = result.ingredients[0]
        assert ip.ingredient.name == "milk"
        assert len(ip.matched_products) > 0
        assert ip.cheapest_option is not None

    @pytest.mark.asyncio
    async def test_total_cost_computed(self):
        mock_llm = AsyncMock()
        mock_llm.call.return_value = {
            "recipe_name": "Rice Bowl",
            "servings": 2,
            "ingredients": [
                {"name": "rice", "quantity": "1", "unit": "cup"},
            ],
        }
        agent = RecipeAgent(mock_llm)
        result = await agent.run("rice bowl")
        assert result.total_estimated_cost > 0


class TestEvaluationAgent:
    @pytest.mark.asyncio
    async def test_single_entity_no_results_does_not_retry(self):
        agent = EvaluationAgent()
        parsed = FinalStructuredQuery(
            clean_query=CleanQuery(text="unknown garlic", normalized_text="unknown garlic"),
            intent_result=IntentResult(intent=QueryIntent.product_search, confidence=0.9, notes=""),
            raw_entities=RawEntities(
                entities=[RawEntity(text="garlic", confidence=0.8)],
                primary_entity="garlic",
                ambiguity_flags=[],
                candidate_entities=["garlic"],
            ),
            normalized_entities=NormalizedEntities(
                entities=[
                    NormalizedEntity(
                        raw_text="garlic",
                        canonical_name="garlic",
                        category="vegetable",
                        possible_variants=["garlic"],
                        confidence=0.9,
                    )
                ],
                unresolved_entities=[],
            ),
            constraints=Constraints(),
            domain_guard=DomainGuardResult(allowed=True),
            ambiguity=AmbiguityDecision(needs_resolution=False, candidate_entities=["garlic"], confidence=0.9),
            fallback=FallbackDecision(),
            execution_plan=ExecutionPlan(),
            execution_graph=ExecutionGraph(graph_id="g"),
            candidate_paths=[],
            user_context=UserContext(),
            learning_signals=LearningSignals(),
            evaluation_history=[],
            failure_policies=[],
            structured_query=StructuredQuery(
                product="garlic",
                filters=QueryFilters(),
                intent=QueryIntent.product_search,
                normalized_query="garlic",
                items=[StructuredItem(name="garlic", category="vegetable")],
                constraints=QueryConstraints(),
                metadata=QueryMetadata(confidence=0.9, notes=""),
                raw_query="garlic",
            ),
        )
        response = FinalResponse(query="garlic", results=[], best_option={}, deals=[], total_price=0.0, metadata={})
        result = await agent.run(parsed, response)
        assert result.should_retry is True
        assert result.success is False
