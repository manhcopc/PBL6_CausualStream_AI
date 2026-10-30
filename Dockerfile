FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY PBL6_Full075_Test_077/demo_video_077.py ./PBL6_Full075_Test_077/demo_video_077.py
COPY PBL6_Full075_Test_077/__init__.py ./PBL6_Full075_Test_077/__init__.py
RUN pip install --no-cache-dir .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

