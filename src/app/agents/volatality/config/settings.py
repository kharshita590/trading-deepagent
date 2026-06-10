import logging

from app.config.env import getenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = getenv("GOOGLE_API_KEY", "") or ""
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0
