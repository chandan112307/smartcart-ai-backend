"""Compiled LangGraph runtime for controller-driven search execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from app.agents.controller import ControllerAgent
from app.data.models import DealResult, EvaluationFrame, MatchingDiagnostics
from app.orchestrator.state import SearchGraphState

if TYPE_CHECKING:
    from app.orchestrator.pipeline import AgentPipeline


def build_search_execution_graph(pipeline: "AgentPipeline"):
    """Build the compiled controller-driven search execution graph."""

    controller_agent = ControllerAgent(
        max_retries=pipeline._max_enrichment_retry_attempts,
        llm_manager=pipeline._llm_manager,
        use_collaborative_llm=pipeline._use_collaborative_llm,
    )

    def clone_diagnostics(state: SearchGraphState) -> MatchingDiagnostics:
        return (state.get("diagnostics") or MatchingDiagnostics()).model_copy(deep=True)

    async def controller_node(state: SearchGraphState) -> SearchGraphState:
        return await controller_agent.act(state)

    async def parse_query_node(state: SearchGraphState) -> SearchGraphState:
        final_structured = await pipeline.parse_query(state.get("user_query", ""))
        return {
            "current_step": "parse_query_node",
            "final_structured_query": final_structured,
            "structured_query": final_structured.structured_query,
            "candidate_entities": pipeline._build_candidate_entities(final_structured),
            "ranking_preferences": dict(final_structured.constraints.ranking_preference_weights),
            "budget_limit": (final_structured.constraints.budget or {}).get("amount"),
            "last_observation": {
                "parsed_query": final_structured.clean_query.normalized_text,
                "candidate_count": len(final_structured.candidate_paths),
            },
        }

    async def normalization_node(state: SearchGraphState) -> SearchGraphState:
        return await pipeline._normalization_agent.act(state)

    async def product_matching_node(state: SearchGraphState) -> SearchGraphState:
        update = await pipeline._product_agent.act(state)
        if update.get("unified_product") is not None and state.get("normalized_item") is not None:
            adjusted_product = pipeline._apply_market_signal_adjustments(
                update["unified_product"],
                state["normalized_item"],
                state.get("market_signals", {}),
            )
            update["unified_product"] = adjusted_product
            update["diagnostics"] = adjusted_product.diagnostics
            update["tool_trace"] = [attempt.model_dump() for attempt in adjusted_product.diagnostics.tool_attempts]
        return update

    async def tool_execution_node(state: SearchGraphState) -> SearchGraphState:
        tool_request = state.get("tool_request") or {}
        tool_result = await pipeline._product_agent.execute_tool_request(tool_request)
        tool_trace = list(state.get("tool_trace", []))
        tool_trace.extend(attempt.model_dump() for attempt in tool_result.get("attempts", []))
        return {
            "current_step": "tool_execution_node",
            "tool_result": tool_result,
            "tool_trace": tool_trace,
            "last_observation": {
                "request_type": tool_request.get("request_type", ""),
                "tool_results": len(tool_result.get("products", [])),
            },
        }

    async def match_quality_node(state: SearchGraphState) -> SearchGraphState:
        diagnostics = state.get("diagnostics") or MatchingDiagnostics()
        unified_product = state.get("unified_product")
        products = unified_product.platforms if unified_product else []
        if not products:
            match_quality = "empty"
        elif (
            diagnostics.approximate_match
            or diagnostics.quality_score < pipeline._matching_quality_threshold
            or len(products) < pipeline._min_high_quality_results
        ):
            match_quality = "weak"
        else:
            match_quality = "strong"
        return {
            "current_step": "match_quality_node",
            "match_quality": match_quality,
            "last_observation": {
                "match_quality": match_quality,
                "quality_score": diagnostics.quality_score,
            },
        }

    async def enrichment_node(state: SearchGraphState) -> SearchGraphState:
        """Expand candidate entities from variants, ambiguity candidates, and category fallbacks."""
        final_structured = state["final_structured_query"]
        normalized_item = state.get("normalized_item")
        diagnostics = clone_diagnostics(state)
        diagnostics.fallback_trace.append("controller_enrichment")
        extra_candidates = []
        if normalized_item:
            extra_candidates.extend(normalized_item.possible_variants)
            if normalized_item.category:
                extra_candidates.extend(pipeline._category_candidates(normalized_item.category))
        extra_candidates.extend(final_structured.ambiguity.candidate_entities or [])
        extra_candidates.extend(final_structured.fallback.alternatives or [])
        extra_candidates.extend(
            variant
            for entity in final_structured.normalized_entities.entities
            for variant in entity.possible_variants
        )
        updated_candidates = pipeline._sanitize_candidate_entities(
            [*state.get("candidate_entities", []), *extra_candidates]
        )
        next_index = min(
            state.get("current_path_index", 0) + 1,
            max(len(updated_candidates) - 1, 0),
        )
        return {
            "current_step": "enrichment_node",
            "candidate_entities": updated_candidates,
            "current_path_index": next_index,
            "retry_count": state.get("retry_count", 0) + 1,
            "diagnostics": diagnostics,
            "normalized_item": None,
            "unified_product": None,
            "match_quality": "",
            "ranking_result": None,
            "ranked_products": None,
            "deal_result": None,
            "deals": None,
            "tool_request": None,
            "tool_result": None,
            "preliminary_products": [],
            "last_observation": {
                "enriched_candidates": updated_candidates,
                "retry_count": state.get("retry_count", 0) + 1,
            },
        }

    async def ranking_node(state: SearchGraphState) -> SearchGraphState:
        update = await pipeline._ranking_agent.act(state)
        ranking_result = pipeline._apply_budget_optimization(
            update["ranking_result"].model_copy(deep=True),
            state.get("budget_limit"),
        )
        update["ranking_result"] = ranking_result
        update["ranked_products"] = ranking_result
        return update

    async def deal_detection_node(state: SearchGraphState) -> SearchGraphState:
        final_structured = state["final_structured_query"]
        update = await pipeline._deal_agent.act(state)
        if "deal_detection_node" not in final_structured.execution_plan.steps:
            update["deal_result"] = DealResult()
            update["deals"] = update["deal_result"]
        return update

    async def response_node(state: SearchGraphState) -> SearchGraphState:
        final_structured = state["final_structured_query"].model_copy(deep=True)
        response = pipeline._builder.build_search_response(
            {
                **dict(state),
                "final_structured_query": final_structured,
            }
        )
        evaluation_update = await pipeline._evaluation_agent.act(
            {
                **dict(state),
                "final_structured_query": final_structured,
                "response": response,
            }
        )
        evaluation_result = evaluation_update["evaluation_result"]
        selected_path = state.get("selected_path", "path-0")
        retry_count = state.get("retry_count", 0)

        final_structured.learning_signals.retry_count = retry_count
        final_structured.learning_signals.evaluation_notes.extend(evaluation_result.correction_suggestions)
        final_structured.learning_signals.evaluation_notes.append(f"selected_path:{selected_path}")
        final_structured.evaluation_history.append(
            EvaluationFrame(
                iteration=retry_count,
                path_id=selected_path,
                quality_score=evaluation_result.quality_score,
                failures=evaluation_result.failure_signals,
                corrections=evaluation_result.correction_suggestions,
            )
        )
        for policy_item in final_structured.failure_policies:
            if policy_item.failure_type in (evaluation_result.failure_signals or []):
                policy_item.applied = True
        seen_paths = {entry["path_id"] for entry in state.get("path_history", [])}
        for path in final_structured.candidate_paths:
            if path.path_id == selected_path:
                path.selected = True
                path.status = "selected"
                path.quality_score = evaluation_result.quality_score
            elif path.path_id in seen_paths:
                path.selected = False
                path.status = "evaluated"

        await pipeline._query_logger.run("evaluation", evaluation_result.model_dump())
        pipeline._decorate_search_response(
            response=response,
            final_structured=final_structured,
            retry_count=retry_count,
            selected_path=selected_path,
            match_quality=state.get("match_quality", "empty"),
            tool_trace=state.get("tool_trace", []),
            path_history=state.get("path_history", []),
            decision_trace=state.get("decision_trace", []),
            collaborative_proposals=state.get("collaborative_proposals", []),
            collaborative_critiques=state.get("collaborative_critiques", []),
            synthesis_trace=state.get("synthesis_trace", {}),
        )
        if evaluation_result.success:
            await pipeline._learning_loop.learn_from_success(final_structured)
        else:
            await pipeline._learning_loop.learn_from_outcome(final_structured, success=False)
        await pipeline._emit_search_event(final_structured, response, selected_path)
        return {
            "current_step": "response_node",
            "response": response,
            "evaluation_result": evaluation_result,
            "final_structured_query": final_structured,
            "last_observation": {
                "response_results": len(response.results),
                "response_deals": len(response.deals),
            },
        }

    def route_from_controller(state: SearchGraphState) -> str:
        """Return the next node selected by the controller from graph state."""
        return state.get("next_action", "response_node")

    graph = StateGraph(SearchGraphState)
    graph.add_node("controller_node", controller_node)
    graph.add_node("parse_query_node", parse_query_node)
    graph.add_node("normalization_node", normalization_node)
    graph.add_node("product_matching_node", product_matching_node)
    graph.add_node("tool_execution_node", tool_execution_node)
    graph.add_node("match_quality_node", match_quality_node)
    graph.add_node("enrichment_node", enrichment_node)
    graph.add_node("ranking_node", ranking_node)
    graph.add_node("deal_detection_node", deal_detection_node)
    graph.add_node("response_node", response_node)

    graph.add_edge(START, "controller_node")
    graph.add_conditional_edges(
        "controller_node",
        route_from_controller,
        {
            "parse_query_node": "parse_query_node",
            "normalization_node": "normalization_node",
            "product_matching_node": "product_matching_node",
            "tool_execution_node": "tool_execution_node",
            "match_quality_node": "match_quality_node",
            "enrichment_node": "enrichment_node",
            "ranking_node": "ranking_node",
            "deal_detection_node": "deal_detection_node",
            "response_node": "response_node",
        },
    )
    graph.add_edge("parse_query_node", "controller_node")
    graph.add_edge("normalization_node", "controller_node")
    graph.add_edge("product_matching_node", "controller_node")
    graph.add_edge("tool_execution_node", "controller_node")
    graph.add_edge("match_quality_node", "controller_node")
    graph.add_edge("enrichment_node", "controller_node")
    graph.add_edge("ranking_node", "controller_node")
    graph.add_edge("deal_detection_node", "controller_node")
    graph.add_edge("response_node", END)
    return graph.compile()
