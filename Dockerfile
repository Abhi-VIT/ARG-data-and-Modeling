FROM python:3.12.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home studio && mkdir -p /app/media /app/staticfiles && chown -R studio:studio /app
USER studio
EXPOSE 8000
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
