![Lint](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![Web App CI](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/web-app.yml/badge.svg)
![ML Client Tests](https://github.com/swe-students-fall2025/4-containers-ovo/actions/workflows/ml-app.yml/badge.svg)


## 🎵 OvO - Music Genre Classification System

A containerized system that classifies audio files as either "rock" or "hiphop" using machine learning. The system consists of three Docker containers:

- **Machine Learning Client**: Processes audio files and performs genre classification
- **Web App**: Provides a dashboard to view classification results with audio recording and file upload capabilities
- **MongoDB Database**: Stores audio files, classification tasks, and results

## Team Members

- [Yilin Wu](https://github.com/YilinWu1028)
- [Lily Luo](https://github.com/lilyluo7412)
- [Jingyao Fu](https://github.com/Sophiaaa430)
- [Mojin Yuan](https://github.com/Mojin-Yuan)
- [Christine Jin](https://github.com/Christine-Jin)

## Quick Start

### Prerequisites

- Docker (20.10+) and Docker Compose (2.0+)

### Run with Docker Compose

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 4-containers-ovo
   ```

2. Set up environment variables:
   ```bash
   cd web-app
   cp env.example .env
   ```
   For docker-compose, edit `.env` and set `MONGO_URI=mongodb://mongodb:27017/`

3. Start all services:
   ```bash
   docker-compose up --build
   ```

4. Access the dashboard at `http://localhost:8000`

### Run Individual Services

**MongoDB:**
```bash
docker run --name mongodb -d -p 27017:27017 mongo:6
```

**Web App:**
```bash
cd web-app
cp env.example .env
# Edit .env: MONGO_URI=mongodb://localhost:27017/
docker build -t ovo-web .
docker run -p 5000:5000 -e MONGO_URI=mongodb://localhost:27017/ ovo-web
```

**Machine Learning Client:**
```bash
cd machine-learning-client
docker build -t ovo-ml-client .
docker run --network host -e MONGO_URI=mongodb://localhost:27017 ovo-ml-client
```

### Local Development

**Web App:**
```bash
cd web-app
pipenv install --dev
cp env.example .env
# Edit .env: MONGO_URI=mongodb://localhost:27017/
pipenv run python app.py
```

**Machine Learning Client:**
```bash
cd machine-learning-client
pipenv install --dev
pipenv run python -m app.worker
```

## Environment Variables

### Web App (`.env` file)

Create `web-app/.env` from `env.example`:

```bash
MONGO_URI=mongodb://localhost:27017/          # localhost for local dev, mongodb://mongodb:27017/ for docker-compose
MONGO_DB_NAME=ml_system
SECRET_KEY=your-secret-key-here
API_LIMIT=100
```

### Machine Learning Client

- `MONGO_URI`: MongoDB connection string (default: `mongodb://mongodb:27017`)
- Database name: `ml_audio` (hardcoded)

## Starter Data

The system uses these MongoDB collections:

- `tasks`: Audio classification tasks
- `results`: Classification results
- `reference_tracks`: Reference audio fingerprints for classification
- `classifications`: Final results (displayed in web app)

**To seed reference data:**
```bash
cd machine-learning-client
pipenv install --dev
pipenv run python -m app.seed_reference
```

Or manually insert documents into MongoDB's `reference_tracks` collection in the `ml_audio` database.

## Configuration Files

The `.env` file is not included in version control. Create it from `env.example`:

```bash
cd web-app
cp env.example .env
```

**Important:** Never commit `.env` files with real credentials. The `env.example` file provides a template with dummy values.
