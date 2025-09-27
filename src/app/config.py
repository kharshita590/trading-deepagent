from pydantic import BaseSettings

class Setting(BaseSettings):
    neo4j_uri:str
    neo4j_user:str
    neo4j_password:str

    redis_url:str
    postgress_url:str

    kite_api_key: str
    kite_api_secret: str
    paper_mode: bool = True

    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
setting=Setting()