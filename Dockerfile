FROM python:3.12-slim

# System deps for playwright (Chromium) and lxml/beautifulsoup4
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    # Chromium runtime libs required by playwright
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 libpangocairo-1.0-0 libx11-6 libx11-xcb1 libxcb1 \
    libxext6 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies only (project source not copied yet — maximises layer cache)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY src/ ./src/

# Install the project itself (registers earnings-8k-worker entry point)
RUN uv sync --frozen --no-dev

# Install playwright browser (Chromium only — smallest footprint)
RUN uv run playwright install chromium

CMD ["uv", "run", "earnings-8k-worker"]
