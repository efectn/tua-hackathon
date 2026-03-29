FROM python:3.10-slim

# Set working directory to the project root to accommodate parent directory dependencies
WORKDIR /app

# Install dependencies first for better caching
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the project files
COPY . .

# Switch to the backend directory
WORKDIR /app/backend

# Expose the Flask port
EXPOSE 5000

# Run the backend application
CMD ["python", "app.py"]
