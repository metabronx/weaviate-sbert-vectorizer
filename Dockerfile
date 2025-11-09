FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev


FROM python:3.13-slim-trixie

# create nonroot user
RUN groupadd --system --gid 999 nonroot && \
    useradd --system --gid 999 --uid 999 --create-home nonroot

ENV PATH="/venv/.venv/bin:$PATH"
COPY --from=builder --chown=nonroot:nonroot /venv /venv

USER nonroot
WORKDIR /wstv

COPY ./wstv/model.py model.py

# download and cache model in image
ENV PYTHONPATH="/wstv"
RUN python model.py

COPY --chown=nonroot:nonroot wstv /wstv

CMD [ "granian", "--host", "0.0.0.0", "--port", "8080", "--no-ws", "--interface", "asgi",  "main:app" ]
