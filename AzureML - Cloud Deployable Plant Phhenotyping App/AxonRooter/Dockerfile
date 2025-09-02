FROM python:3.9-slim

WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root

# install the streamlit dependency


COPY src ./src
COPY static ./static

EXPOSE 80

# STARTS THE API SERVER
ENTRYPOINT ["python", "-m", "axonrooter.start_api"]
