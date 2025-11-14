![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Containerized App Exercise

Build a containerized app that uses machine learning. See [instructions](./instructions.md) for details.


## Project Structure

```
4-containers-ovo/
├─ .githooks/
│  └─ commit-msg
├─ .github/
│  └─ workflows/
│     ├─ event-logger.yml
│     ├─ lint.yml
│     ├─ Pipfile
│     ├─ Pipfile.lock
│     └─ requirements.txt
├─ machine-learning-client/
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ features.py
│  │  ├─ seed_reference.py
│  │  └─ worker.py
│  ├─ Pipfile
│  ├─ Pipfile.lock
│  └─ readme.txt
├─ web-app/
│  ├─ templates/
│  │  └─ index.html
│  ├─ test_web/
│  │  ├─ conftest.py
│  │  ├─ test_app.py
│  │  └─ test_database.py
│  ├─ .pylintrc
│  ├─ app.py
│  ├─ config.py
│  ├─ database.py
│  ├─ Dockerfile
│  ├─ env.example
│  ├─ Pipfile
│  ├─ Pipfile.lock
│  ├─ pytest.ini
│  └─ readme.txt
├─ .dockerignore
├─ .gitignore
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
├─ instructions.md
├─ LICENSE
├─ note.txt
└─ README.md
```