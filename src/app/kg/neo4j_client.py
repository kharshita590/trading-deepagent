from neo4j import GraphDatabase
from app.config import settings

driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

def run_cypher(query, parameters=None):
    with driver.session() as session:
        return session.run(query, parameters or {})
