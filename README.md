# trading-deepagent

Multi-agent portfolio advisor for NSE/Indian stocks built with LangGraph. The system coordinates research, fundamental, macro, technical, volatility, behavioral, and risk workflows to produce portfolio suggestions.

## Prerequisites

- Python 3.11+
- Poetry
- Redis for caching and async job storage
- Optional: Postgres if you want to override the default SQLite persistence
- API access for live analysis:
  - `GOOGLE_API_KEY`
  - `TWELVEDATA_API_KEY` if you plan to use the TwelveData fallback

## Setup

1. Clone the repo and enter it:

```bash
git clone <repo-url>
cd trading-deepagent
```

2. Install dependencies:

```bash
poetry install
```

3. Create your environment file:

```bash
cp .env.example .env
```

4. Fill in the required values in `.env`:

- `GOOGLE_API_KEY`
- `TWELVEDATA_API_KEY`
- `LOG_LEVEL`
- `REDIS_URL`
- `DATABASE_URL`
- `API_KEY_SECRET` for API access control if you use the FastAPI service

The application validates required values on startup and will fail fast if a required key is missing.

## Run the CLI

The CLI is the simplest way to use the project locally:

```bash
poetry run python -m app.agents.main
PYTHONPATH=src poetry run python -m app.agents.main
```

Type a portfolio question, for example:

```text
I have 250000 INR, moderate risk, 3 year horizon, diversified across pharma and banking.
```

The CLI prints a disclaimer with every recommendation. It is algorithmic/educational output only and is not SEBI-registered investment advice.

If you want the CLI to talk to a local API server instead of running the orchestrator directly, set:

```bash
export PORTFOLIO_API_URL=http://localhost:8000
```

## Run the API

Start the FastAPI service:

```bash
poetry run uvicorn app.api.main:app --reload
```

Useful endpoints:

- `POST /portfolio/analyze`
- `GET /portfolio/analyze/{job_id}`
- `GET /portfolio/history`
- `GET /health`
- `GET /docs`

Requests are protected with `X-API-Key` when `API_KEY_SECRET` is set.

Example request:

```bash
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "query": "I want a moderately risky portfolio for 3 years with 250000 INR",
    "conversation_history": []
  }'
```

## Backtesting

The technical signal logic can be validated against historical data:

```bash
poetry run python -m app.backtest.run --ticker RELIANCE --period 2y
```

This is reporting only and does not alter live recommendations.

## NSE Universe Refresh

The research agent uses a bundled NSE universe CSV. To refresh it manually:

```bash
poetry run python scripts/refresh_nse_universe.py
```

## Docker

Build and run the stack with Redis:

```bash
docker compose up --build
```

The compose file includes:

- `app` for the FastAPI service
- `redis` for caching and job storage
- optional `postgres` behind a profile

## Testing

Run the main test suite with:

```bash
poetry run pytest
```

The repository also includes unit and integration tests for strategy logic, risk math, technical indicators, bias detection, and graph wiring.

## Notes

- The code keeps a compatibility bridge for the legacy `behavorial_agent` import path, but new code should use `behavioral_agent`.
- SQLite is the default database. Set `DATABASE_URL` to point at Postgres if needed.
- `KNOWN_LIMITATIONS.md` tracks gaps that are intentionally deferred.
