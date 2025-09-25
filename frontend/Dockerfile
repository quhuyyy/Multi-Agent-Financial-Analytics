FROM base AS frontend

RUN apt-get update && apt-get install -y \
    libmagic-dev \
 && rm -rf /var/lib/apt/lists/*

RUN pip install streamlit


WORKDIR /app

COPY . /app/

CMD ["streamlit", "run", "app.py", "--server.port=8501"]