###############################################
# Base image
###############################################
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} as python_base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PIPENV_VENV_IN_PROJECT=1\
    PYSETUP_PATH="/app" \
    VENV_PATH="/app/.venv"

ENV PATH="$VENV_PATH/bin:$PATH"

###############################################
# Builder image
###############################################
FROM python_base as builder_base
# Installing host dependencies
RUN apt-get update
RUN apt-get --no-install-recommends -y install \
    libcairo2-dev \
    pkg-config \
    python3-dev
# Clean up repositories
RUN rm -rf /var/lib/apt/lists/*

WORKDIR $PYSETUP_PATH
COPY Pipfile Pipfile.lock ./
RUN pip install --no-cache-dir pipenv \
&& pipenv sync --dev

###############################################
# Production Image
###############################################
FROM python_base as production
COPY --from=builder_base $PYSETUP_PATH $PYSETUP_PATH

WORKDIR $PYSETUP_PATH
COPY . $PYSETUP_PATH

ENTRYPOINT ["python", "run_model.py"]