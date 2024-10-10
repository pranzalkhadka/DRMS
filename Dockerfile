FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY nep-fuse-2.traineddata /usr/share/tesseract-ocr/5/tessdata/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/home.py"]