# Imagination-backend

This repository contains the back-end server for the Imagination app. This project was created with [Flask](https://flask.palletsprojects.com/en/3.0.x/) and uses a NGINX reverse proxy along with Docker.

### Local development with only Flask server running

**Requirements**
- Python 3.12.3

1. Create virtual environment at project root
```python3 -m venv .venv```

2. Activate virtual environment
```source .venv/bin/activate```

3. Install requirements with Pip
```pip install -r flask/requirements.txt```

4. In ```flask/app``` directory create new folder ```instance``` and copy the ```config.example.json``` file into it. Replace the example values with your own.

5. Run the Flask app in debug mode on port 8000. You don't have to use port 8000 but for parity with the Docker setup it is used here.
```flask --app flask/app run --port=8000 --debug```

### Build with Flask, NGINX and Docker

**Requirements**
- Docker Desktop

1. In ```flask/app``` directory create new folder ```instance``` and copy the ```config.example.json``` file into it. Replace the example values with your own.

2. Run the Flask app and NGINX server with Docker Compose
```docker compose up -d```