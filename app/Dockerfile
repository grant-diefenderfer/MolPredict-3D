FROM python:3.11-slim

WORKDIR /app

# Install backend dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pull all generated Random Forest models into the container
COPY *.pkl ./

# Copy the backend API code
COPY app/ ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]