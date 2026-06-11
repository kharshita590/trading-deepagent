import asyncio
import os

from app.config.validate_env import validate_required_env
from app.constants import DISCLAIMER_TEXT
from app.core.logging import configure_logging


async def main():
    validate_required_env()
    configure_logging()
    api_url = os.getenv("PORTFOLIO_API_URL", "").strip()
    orchestrator = None
    conversation_history = []
    result = None
    print("Tell me about your investment plans, and I'll help you create a portfolio.")
    print(DISCLAIMER_TEXT)
    print("Type 'exit' to quit.\n")

    async def analyze_query(query: str, history):
        if not api_url:
            nonlocal orchestrator
            if orchestrator is None:
                from .portfolio_orchestrator.main import PortfolioOrchestrator

                orchestrator = PortfolioOrchestrator(api_key=os.getenv("GOOGLE_API_KEY", ""))
            return await orchestrator.run_from_query(query, history)
        try:
            import requests
        except ModuleNotFoundError:  # pragma: no cover - sandbox fallback
            raise RuntimeError("requests is required when PORTFOLIO_API_URL is set")
        response = await asyncio.to_thread(
            requests.post,
            f"{api_url.rstrip('/')}/portfolio/analyze",
            json={"query": query, "conversation_history": history},
            headers={"X-API-Key": os.getenv("API_KEY_SECRET", "")} if os.getenv("API_KEY_SECRET") else {},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "needs_clarification":
            return {"status": "incomplete", "message": payload.get("message"), "conversation_history": payload.get("conversation_history", [])}
        job_id = payload["job_id"]
        while True:
            status_response = await asyncio.to_thread(
                requests.get,
                f"{api_url.rstrip('/')}/portfolio/analyze/{job_id}",
                headers={"X-API-Key": os.getenv("API_KEY_SECRET", "")} if os.getenv("API_KEY_SECRET") else {},
                timeout=120,
            )
            status_response.raise_for_status()
            job = status_response.json()
            if job.get("status") in {"completed", "failed"}:
                return job.get("result", job)
            await asyncio.sleep(1.0)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using Portfolio Investment Assistant!")
            break
        if not user_input:
            continue
        result = await analyze_query(user_input, conversation_history)
        if result.get('status') == 'incomplete':
            print(f"\nAssistant: {result.get('message')}\n")
            conversation_history = result.get('conversation_history', [])
        elif result.get('status') == 'complete':
            print(" PORTFOLIO ANALYSIS COMPLETED")
            print(f"\nInvestment Amount: ₹{result.get('investment_amount'):,.2f}")
            print(f"Risk Tolerance: {result.get('risk_tolerance')}")
            print(f"Investment Horizon: {result.get('investment_horizon')}")
            print(f"Recommendations: {len(result.get('recommendations', []))} stocks") 
            print(f"\nDisclaimer: {result.get('disclaimer', DISCLAIMER_TEXT)}")
            if result.get('recommendations'):
                print("\nRecommended Stocks:")
                for i, rec in enumerate(result.get('recommendations'), 1):
                    print(f"\n{i}. {rec['company_name']} ({rec['ticker']})")
                    print(f"   Sector: {rec['sector']}")
                    print(f"   Price: ₹{rec['price']:,.2f}")
                    print(f"   Allocation: {rec['allocation_percentage']:.1f}% (₹{rec['allocation_amount']:,.2f})")
                        
            print("\nWould you like to create another portfolio? (yes/no)")
            response = input("You: ").strip().lower()
            
            if response in ['no', 'n', 'exit', 'quit']:
                print("\nThank you for using Portfolio Investment Assistant!")
                break
            else:
                conversation_history = []
                print("Let's start a new portfolio analysis.")
    
    return result
if __name__ == "__main__":
    asyncio.run(main())
