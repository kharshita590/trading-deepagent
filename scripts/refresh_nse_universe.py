from __future__ import annotations

import csv
from pathlib import Path


NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def refresh(output_path: str = "src/app/data/nse_universe.csv") -> None:
    try:
        import requests
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("requests is required to refresh the NSE universe") from exc
    response = requests.get(NSE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    content = response.text.splitlines()
    reader = csv.DictReader(content)
    rows = []
    for row in reader:
        symbol = (row.get("SYMBOL") or row.get("symbol") or "").strip().upper()
        company = (row.get("NAME OF COMPANY") or row.get("company_name") or symbol).strip()
        if not symbol:
            continue
        rows.append({"symbol": symbol, "company_name": company, "sector": row.get("SECTOR", "Unknown")})

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "company_name", "sector"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    refresh()
