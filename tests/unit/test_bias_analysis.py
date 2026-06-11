from app.agents.behavioral_agent.agents.bias_analysis import BiasAnalysisAgent


def test_bias_thresholds_detect_expected_biases():
    agent = BiasAnalysisAgent()
    state = {
        "selected_stocks": [{"volatility": 0.4}, {"volatility": 0.35}],
        "technical_analysis": {"confidence_score": 0.9, "momentum_score": 0.8},
        "messages": [],
    }

    import asyncio

    result = asyncio.run(agent.execute(state))
    bias_types = {bias["type"].value for bias in result["behavioral_biases"]}
    assert "overconfidence" in bias_types
    assert "loss_aversion" in bias_types
    assert "fomo" in bias_types
