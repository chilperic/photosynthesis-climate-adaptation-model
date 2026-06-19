FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY docs ./docs

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[dev]"

CMD ["bash", "-lc", "pytest -q && python scripts/00_list_parameter_presets.py && python scripts/01_run_model_smoke.py"]
