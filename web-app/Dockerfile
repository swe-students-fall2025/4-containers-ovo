FROM python:3.10-slim

WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --deploy --ignore-pipfile

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["pipenv", "run", "python", "app.py"]

