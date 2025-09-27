import redis
from app.config import settings

r = redis.Redis.from_url(settings.redis_url, decode_responses=True)
