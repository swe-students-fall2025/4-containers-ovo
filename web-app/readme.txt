Web App Container
=================

Code related to the web app goes in this folder.

Build the web app image (from the repository root):

    docker build -t ovo-web -f web-app/Dockerfile web-app

Run it against your MongoDB instance (replace the URI if needed):

    docker run -p 5000:5000 --env MONGO_URI="mongodb://mongodb:27017/" ovo-web

For local development:

    cd web-app
    pipenv install --dev
    cp env.example .env
    # Edit .env: MONGO_URI=mongodb://localhost:27017/
    pipenv run python app.py

The container/application provides a web dashboard at http://localhost:5000 (or port specified by PORT env var) that displays music classification results from the ML client. Users can record audio or upload audio files, which are stored in MongoDB GridFS and processed by the ML client.
