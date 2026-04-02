"""Controller agent for dynamic LangGraph search routing."""

from __future__ import annotations

from typing import Any, Mapping

from app.agents.base_execution import BaseExecutionAgent
from app.agents.collaborative_reasoning import (
    ControllerProposal,
    ControllerSynthesis,
    CritiqueAgent,
    ProposalAgent,
    ProposalCritique,
    SynthesisAgent,
    score_actions,
)


class ControllerAgent(BaseExecutionAgent):
    """Decides the next graph action from accumulated runtime state.

    By default uses deterministic rule-based routing (zero LLM calls).
    LLM-based collaborative reasoning is only used when *both*
    ``llm_manager`` is supplied *and* ``use_collaborative_llm`` is True.
    """

    def __init__(
        self,
        *,
        max_retries: int,
        llm_manager: Any | None = None,
        use_collaborative_llm: bool = False,
    ) -> None:
        self._max_retries = max_retries
        self._llm_manager = llm_manager
        self._use_collaborative_llm = use_collaborative_llm and llm_manager is not None
        self._proposal_agents = (
            ProposalAgent(
                llm_manager=llm_manager,
                role="routing_strategist",
                focus="Pick the next state transition that unlocks the strongest downstream progress.",
            ),
            ProposalAgent(
                llm_manager=llm_manager,
                role="tool_retry_specialist",
                focus="Prefer tools, retries, and enrichment only when current evidence is insufficient.",
            ),
            ProposalAgent(
                llm_manager=llm_manager,
                role="termination_guardian",
                focus="Protect against unnecessary loops and prefer safe termination when quality is sufficient.",
            ),
        )
        self._critique_agents = (
            CritiqueAgent(
                llm_manager=llm_manager,
                role="feasibility_critic",
                focus="Check if the proposal is feasible with the current graph state and action set.",
            ),
            CritiqueAgent(
                llm_manager=llm_manager,
                role="quality_critic",
                focus="Check if the proposal maximizes result quality and avoids premature response building.",
            ),
        )
        self._synthesis_agent = SynthesisAgent(llm_manager=llm_manager) if llm_manager is not None else None

    async def act(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Read runtime state and emit the controller's next action decision."""
        fallback_action = self._decide_next_action(state)
        collaborative_proposals: list[dict[str, Any]] = []
        collaborative_critiques: list[dict[str, Any]] = []
        synthesis_trace: dict[str, Any] = {
            "action": fallback_action,
            "rationale": "Deterministic fallback path selected.",
            "confidence": 0.0,
            "consensus": "fallback",
            "score_breakdown": {fallback_action: 1.0},
        }
        decision_source = "deterministic_fallback"
        available_actions = self._available_actions(state, fallback_action)
        next_action = fallback_action
        if self._use_collaborative_llm:
            (
                next_action,
                collaborative_proposals,
                collaborative_critiques,
                synthesis_trace,
                decision_source,
            ) = await self._run_collaborative_reasoning(
                state=state,
                available_actions=available_actions,
                fallback_action=fallback_action,
            )
        decision_trace = list(state.get("decision_trace", []))
        decision_trace.append(
            {
                "step": "controller_node",
                "action": next_action,
                "retry_count": state.get("retry_count", 0),
                "match_quality": state.get("match_quality", ""),
                "current_entity": state.get("current_entity", ""),
                "available_actions": available_actions,
                "decision_source": decision_source,
                "rationale": synthesis_trace.get("rationale", ""),
                "confidence": synthesis_trace.get("confidence", 0.0),
            }
        )
        return {
            "current_step": "controller_node",
            "next_action": next_action,
            "decision_trace": decision_trace,
            "collaborative_proposals": collaborative_proposals,
            "collaborative_critiques": collaborative_critiques,
            "synthesis_trace": synthesis_trace,
            "last_observation": {
                "controller_action": next_action,
                "retry_count": state.get("retry_count", 0),
                "match_quality": state.get("match_quality", ""),
                "decision_source": decision_source,
            },
        }

    async def _run_collaborative_reasoning(
        self,
        *,
        state: Mapping[str, Any],
        available_actions: list[str],
        fallback_action: str,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
        if not available_actions:
            return (
                fallback_action,
                [],
                [],
                {
                    "action": fallback_action,
                    "rationale": "No available actions were supplied.",
                    "confidence": 0.0,
                    "consensus": "fallback",
                    "score_breakdown": {fallback_action: 1.0},
                },
                "deterministic_fallback",
            )
        try:
            collaborative_state = self._build_collaboration_state(state, available_actions, fallback_action)
            proposals = await self._generate_proposals(collaborative_state)
            critiques = await self._generate_critiques(collaborative_state, proposals)
            score_breakdown = score_actions(proposals, critiques, available_actions, fallback_action)
            synthesis = await self._synthesize(collaborative_state, proposals, critiques, score_breakdown)
            if not synthesis.action:
                synthesis.action = max(score_breakdown, key=score_breakdown.get)
            if synthesis.action not in available_actions:
                raise ValueError(f"invalid action {synthesis.action!r}")
            synthesis.score_breakdown = synthesis.score_breakdown or score_breakdown
            return (
                synthesis.action,
                [proposal.model_dump() for proposal in proposals],
                [critique.model_dump() for critique in critiques],
                synthesis.model_dump(),
                "collaborative_llm",
            )
        except Exception:
            return (
                fallback_action,
                [],
                [],
                {
                    "action": fallback_action,
                    "rationale": "Collaborative reasoning unavailable; deterministic fallback selected.",
                    "confidence": 0.0,
                    "consensus": "fallback",
                    "score_breakdown": {fallback_action: 1.0},
                },
                "deterministic_fallback",
            )

    def _build_collaboration_state(
        self,
        state: Mapping[str, Any],
        available_actions: list[str],
        fallback_action: str,
    ) -> dict[str, Any]:
        diagnostics = self._serialize_diagnostics(state.get("diagnostics"))
        current_entity = state.get("current_entity")
        if not current_entity:
            normalized_item = state.get("normalized_item")
            if hasattr(normalized_item, "canonical_name"):
                current_entity = normalized_item.canonical_name
            elif isinstance(normalized_item, dict):
                current_entity = normalized_item.get("canonical_name", "")
        if not current_entity:
            candidate_entities = state.get("candidate_entities", [])
            current_path_index = min(state.get("current_path_index", 0), max(len(candidate_entities) - 1, 0))
            if candidate_entities:
                current_entity = candidate_entities[current_path_index]
        if not current_entity:
            structured_query = state.get("structured_query")
            if isinstance(structured_query, dict):
                current_entity = structured_query.get("product", "")
            elif structured_query:
                current_entity = getattr(structured_query, "product", "")
        return {
            "user_query": state.get("user_query", ""),
            "current_entity": current_entity or "",
            "match_quality": state.get("match_quality", ""),
            "retry_count": state.get("retry_count", 0),
            "diagnostics": diagnostics,
            "tool_trace": state.get("tool_trace", []),
            "available_actions": available_actions,
            "fallback_action": fallback_action,
            "last_observation": state.get("last_observation", {}),
            "has_tool_request": bool(state.get("tool_request")),
            "has_tool_result": bool(state.get("tool_result")),
            "has_ranking_result": state.get("ranking_result") is not None,
            "has_deal_result": state.get("deal_result") is not None,
        }

    @staticmethod
    def _serialize_diagnostics(diagnostics: Any) -> dict[str, Any]:
        if diagnostics is None:
            return {}
        if isinstance(diagnostics, dict):
            return diagnostics
        if hasattr(diagnostics, "model_dump"):
            return diagnostics.model_dump()
        return {"value": str(diagnostics)}

    async def _generate_proposals(self, collaborative_state: Mapping[str, Any]) -> list[ControllerProposal]:
        proposals: list[ControllerProposal] = []
        for agent in self._proposal_agents:
            proposal = await agent.act(collaborative_state)
            if proposal.action in collaborative_state["available_actions"]:
                proposals.append(proposal)
        if not proposals:
            proposals.append(
                ControllerProposal(
                    role="fallback_router",
                    action=collaborative_state["fallback_action"],
                    rationale="Fallback action used because no valid proposal was returned.",
                    confidence=1.0,
                    evidence=["deterministic controller logic"],
                )
            )
        return proposals

    async def _generate_critiques(
        self,
        collaborative_state: Mapping[str, Any],
        proposals: list[ControllerProposal],
    ) -> list[ProposalCritique]:
        critiques: list[ProposalCritique] = []
        for proposal in proposals:
            critique_state = {**dict(collaborative_state), "proposal": proposal.model_dump()}
            for agent in self._critique_agents:
                critique = await agent.act(critique_state)
                if critique.proposal_action in collaborative_state["available_actions"]:
                    critiques.append(critique)
        return critiques

    async def _synthesize(
        self,
        collaborative_state: Mapping[str, Any],
        proposals: list[ControllerProposal],
        critiques: list[ProposalCritique],
        score_breakdown: dict[str, float],
    ) -> ControllerSynthesis:
        if self._synthesis_agent is None:
            return ControllerSynthesis(
                action=max(score_breakdown, key=score_breakdown.get),
                rationale="Deterministic score selection was used because no synthesis agent is configured.",
                confidence=score_breakdown[max(score_breakdown, key=score_breakdown.get)],
                consensus="score_only",
                score_breakdown=score_breakdown,
            )
        synthesis_state = {
            **dict(collaborative_state),
            "proposals": [proposal.model_dump() for proposal in proposals],
            "critiques": [critique.model_dump() for critique in critiques],
            "score_breakdown": score_breakdown,
        }
        synthesis = await self._synthesis_agent.act(synthesis_state)
        if not synthesis.score_breakdown:
            synthesis.score_breakdown = score_breakdown
        return synthesis

    def _available_actions(self, state: Mapping[str, Any], fallback_action: str) -> list[str]:
        final_structured = state.get("final_structured_query")
        plan_steps = set(final_structured.execution_plan.steps) if final_structured else set()
        if not state.get("structured_query"):
            return ["parse_query_node"]
        if state.get("tool_request") and not state.get("tool_result"):
            return ["tool_execution_node"]
        if state.get("tool_result"):
            return ["product_matching_node"]
        if state.get("response") is not None:
            return ["response_node"]
        if not state.get("normalized_item"):
            return ["normalization_node"]
        if not state.get("unified_product"):
            return ["product_matching_node"]
        if not state.get("match_quality"):
            return ["match_quality_node"]
        if state.get("match_quality") == "strong":
            actions: list[str] = []
            if not state.get("ranking_result"):
                actions.append("ranking_node")
            if "deal_detection_node" in plan_steps and state.get("deal_result") is None:
                actions.append("deal_detection_node")
            actions.append("response_node")
            return list(dict.fromkeys(actions))
        actions = []
        if state.get("retry_count", 0) < int(state.get("max_retries", self._max_retries)):
            actions.append("enrichment_node")
        unified_product = state.get("unified_product")
        platforms = getattr(unified_product, "platforms", None)
        if platforms:
            if not state.get("ranking_result"):
                actions.append("ranking_node")
            if "deal_detection_node" in plan_steps and state.get("deal_result") is None:
                actions.append("deal_detection_node")
        actions.append("response_node")
        if fallback_action not in actions:
            actions.insert(0, fallback_action)
        return list(dict.fromkeys(actions))

    def _decide_next_action(self, state: Mapping[str, Any]) -> str:
        final_structured = state.get("final_structured_query")
        plan_steps = set(final_structured.execution_plan.steps) if final_structured else set()
        if not state.get("structured_query"):
            return "parse_query_node"
        if state.get("tool_request") and not state.get("tool_result"):
            return "tool_execution_node"
        if state.get("tool_result"):
            return "product_matching_node"
        if state.get("response") is not None:
            return "response_node"
        if not state.get("normalized_item"):
            return "normalization_node"
        if not state.get("unified_product"):
            return "product_matching_node"
        if not state.get("match_quality"):
            return "match_quality_node"
        if state.get("match_quality") == "strong":
            if not state.get("ranking_result"):
                return "ranking_node"
            if "deal_detection_node" in plan_steps and state.get("deal_result") is None:
                return "deal_detection_node"
            return "response_node"
        if state.get("retry_count", 0) < int(state.get("max_retries", self._max_retries)):
            return "enrichment_node"
        unified_product = state.get("unified_product")
        if unified_product and getattr(unified_product, "platforms", []):
            if not state.get("ranking_result"):
                return "ranking_node"
            if "deal_detection_node" in plan_steps and state.get("deal_result") is None:
                return "deal_detection_node"
            return "response_node"
        return "response_node"
