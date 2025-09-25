FROM base AS backend

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libmagic-dev \
    tesseract-ocr \
    libgl1-mesa-glx \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install vnstock

RUN pip install DateTime

RUN pip install shap

RUN pip install selenium==4.32.0

RUN pip install imblearn==0.0

RUN pip install ta==0.11.0

RUN pip install tensorflow==2.19.0

RUN pip install tf-keras

RUN pip install keras-self-attention==0.51.0

RUN pip install keras-tuner==1.4.7

# Cài đặt Google Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update \
    && apt-get install -y ./google-chrome-stable_current_amd64.deb \
    && rm google-chrome-stable_current_amd64.deb

# Cài đặt ChromeDriver
RUN wget https://storage.googleapis.com/chrome-for-testing-public/137.0.7151.103/linux64/chromedriver-linux64.zip \
    && unzip chromedriver-linux64.zip \
    && mv chromedriver-linux64/chromedriver /usr/bin/chromedriver \
    && chmod +x /usr/bin/chromedriver \
    && rm chromedriver-linux64.zip

WORKDIR /app

COPY . /app/

CMD ["sh", "-c", "sleep 20 && uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120 --reload"]