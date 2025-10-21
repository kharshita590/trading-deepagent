import logging
import asyncio
import json
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from ..models.types import QueryState
logger = logging.getLogger(__name__)
class QueryProcessor:    
    SYSTEM_PROMPT = """You are a financial assistant helping users plan their investments. 
    Extract the following parameters from the user's query:
    
    Required parameters:
    1. investment_amount (float): Amount in rupees (₹). Accept formats like "50k", "1 lakh", "50000"
    2. risk_tolerance (string): Must be exactly 'low', 'moderate', or 'high'
    3. investment_horizon (string): Must be exactly 'short' (< 1 year), 'medium' (1-5 years), or 'long' (> 5 years)
    
    Optional parameters:
    4. preferred_sectors (list): Any sectors mentioned (e.g., Technology, Healthcare, Finance)
    5. exclude_sectors (list): Any sectors to avoid
    
    Return ONLY a valid JSON object with this exact structure:
    {
        "extracted_parameters": {
            "investment_amount": <float or null>,
            "risk_tolerance": <"low"/"moderate"/"high" or null>,
            "investment_horizon": <"short"/"medium"/"long" or null>,
            "preferred_sectors": <list or null>,
            "exclude_sectors": <list or null>
        },
        "missing_parameters": [<list of missing required parameter names>],
        "is_complete": <true/false>,
        "clarification_needed": "<friendly question asking for missing parameters>"
    }
    
    Inference rules:
    - "50k" = 50000, "1 lakh" = 100000, "1L" = 100000, "2 lakhs" = 200000
    - "cautious", "safe", "conservative" = low risk
    - "balanced", "moderate" = moderate risk
    - "aggressive", "high growth" = high risk
    - "few months" = short, "couple years", "2-3 years" = medium, "retirement", "long term" = long
    
    If all required parameters are present, set is_complete to true and clarification_needed to empty string.
    Only include parameters that are explicitly mentioned or can be clearly inferred.
    Do not include null values in extracted_parameters, omit them entirely if not found."""
    
    def __init__(self, gemini_model: ChatGoogleGenerativeAI):
        self.gemini_model = gemini_model
    
    async def extract_parameters(
        self, 
        query: str, 
        conversation_history: List[Dict] = None
    ) -> QueryState:
        conversation_history = conversation_history or []
        full_prompt = self._build_prompt(query, conversation_history)
        response = await asyncio.to_thread(
            self.gemini_model.invoke,
            full_prompt
        )
        result = self._parse_response(response)
        new_history = conversation_history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": result.get("clarification_needed", "")}
        ]
        return QueryState(
            user_query=query,
            extracted_parameters=result.get("extracted_parameters", {}),
            missing_parameters=result.get("missing_parameters", []),
            is_complete=result.get("is_complete", False),
            conversation_history=new_history
        )
    async def generate_clarification_message(
        self, 
        query_state: QueryState
    ) -> str:
        prompt = f"""Based on this conversation, the user has provided: {json.dumps(query_state.extracted_parameters)}
        Missing required parameters: {', '.join(query_state.missing_parameters)}
        Generate a friendly, conversational message asking for the missing information. 
        Be specific about what's needed but keep it natural and helpful.
        Return only the message text, no JSON."""
        
        response = await asyncio.to_thread(
            self.gemini_model.invoke,
            prompt
        )
        
        return response.content.strip() if hasattr(response, 'content') else str(response).strip()
    def _build_prompt(self, query: str, conversation_history: List[Dict]) -> str:
        full_prompt = self.SYSTEM_PROMPT + "\n\nConversation history:\n"
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
        
        full_prompt += f"\nUser: {query}\n\nExtract parameters and return JSON:"
        return full_prompt
    def _parse_response(self, response) -> Dict:
        try:
            response_text = response.content if hasattr(response, 'content') else str(response)
            response_text = response_text.strip()            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
        
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response}")
            raise
