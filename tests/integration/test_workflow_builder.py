from __future__ import annotations

from app.agents.portfolio_orchestrator.workflow_builder import WorkflowBuilder


class DummyNodes:
    async def run_investment_allocation(self, state): return {}
    async def run_research(self, state): return {}
    async def run_fundamental_analysis(self, state): return {}
    async def run_macro_analysis(self, state): return {}
    async def run_technical_analysis(self, state): return {}
    async def run_volatility_liquidity(self, state): return {}
    async def run_behavioral_psychology(self, state): return {}
    async def join_parallel_analyses(self, state): return {}
    async def run_risk_management(self, state): return {}


def test_workflow_builder_compiles_and_has_expected_edges():
    builder = WorkflowBuilder(DummyNodes())
    workflow = builder.build()
    graph = workflow.get_graph()
    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
    assert ("run_investment_allocation", "run_research") in edge_pairs
    assert ("run_research", "run_fundamental_analysis") in edge_pairs
    assert ("parallel_join", "run_risk_management") in edge_pairs
