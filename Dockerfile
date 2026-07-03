# syntax=docker/dockerfile:1
FROM python:3.12-slim

# ── System packages ─────────────────────────────────────────────────────────
# Cache mount keeps downloaded .deb files between rebuilds so apt-get is a
# near-no-op when nothing changed.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    # Chromium runtime libs required by playwright
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
    libcairo2 libpangocairo-1.0-0 libx11-6 libx11-xcb1 libxcb1 \
    libxext6 fonts-liberation

# ── uv (pinned minor — avoids surprise cache invalidation from upstream) ─────
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /usr/local/bin/uv

WORKDIR /app

# ── Dependency layer (rebuilt only when pyproject.toml / uv.lock change) ─────
COPY pyproject.toml uv.lock ./

# Install all third-party packages WITHOUT the project source.
# BuildKit cache mount stores downloaded wheels so re-installs after a lockfile
# change only fetch packages that actually changed.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ── Playwright Chromium (~200 MB) ────────────────────────────────────────────
# CRITICAL: placed BEFORE "COPY src/" so a code-only edit never re-downloads
# the browser.  This layer is only invalidated when the playwright dep version
# changes in uv.lock.
RUN uv run playwright install chromium

# ── Source layer (fast rebuild — all heavy layers above are cached) ───────────
COPY src/ ./src/

# Register the project entry-point.  Wheels are already installed; this is
# near-instant because uv only needs to link the package into the venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

CMD ["uv", "run", "earnings-8k-worker"]
