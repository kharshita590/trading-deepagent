import asyncio
import logging
import os

from app.config.validate_env import validate_required_env

from .portfolio_orchestrator.main import PortfolioOrchestrator


async def main():
    validate_required_env()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    orchestrator = PortfolioOrchestrator(api_key=os.getenv("GOOGLE_API_KEY", ""))
    conversation_history = []
    result = None
    print("Tell me about your investment plans, and I'll help you create a portfolio.")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nThank you for using Portfolio Investment Assistant!")
            break
        if not user_input:
            continue
        result = await orchestrator.run_from_query(user_input, conversation_history)
        if result.get('status') == 'incomplete':
            print(f"\nAssistant: {result.get('message')}\n")
            conversation_history = result.get('conversation_history', [])
        elif result.get('status') == 'complete':
            print(" PORTFOLIO ANALYSIS COMPLETED")
            print(f"\nInvestment Amount: ₹{result.get('investment_amount'):,.2f}")
            print(f"Risk Tolerance: {result.get('risk_tolerance')}")
            print(f"Investment Horizon: {result.get('investment_horizon')}")
            print(f"Recommendations: {len(result.get('recommendations', []))} stocks") 
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
