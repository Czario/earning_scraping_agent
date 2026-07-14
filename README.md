uv run earnings --ticker MSFT --dry-run
uv run earnings --ticker AAPL MSFT GOOGL --dry-run

uv run earnings --ticker AAPL MSFT GOOGL NVDA          # 4 parallel workers (default)
uv run earnings --ticker AAPL MSFT GOOGL --max-workers 2

uv run earnings --ticker MSFT -v

