import sqlalchemy
from sqlalchemy import create_engine
from app.config import settings

engine = create_engine(settings.postgres_url, echo=True)

def get_connection():
    return engine.connect()
