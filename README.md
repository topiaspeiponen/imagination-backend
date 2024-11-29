# Imagination-backend

This repository contains the back-end server for the Imagination app. This project was created with [Flask](https://flask.palletsprojects.com/en/3.0.x/) and uses a NGINX reverse proxy along with Docker.

The frontend for this app can be seen at https://github.com/topiaspeiponen/imagination-ui

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
```flask --app flask/imagination run --port=8000 --debug```

### Build with Flask, NGINX and Docker

**Requirements**
- Docker Desktop

1. In ```flask/app``` directory create new folder ```instance``` and copy the ```config.example.json``` file into it. Replace the example values with your own.

2. Run the Flask app and NGINX server with Docker Compose
```docker compose up -d```

**Build separate containers**
In case you want to build and run the Flask app and Nginx proxy separately, do the following:

1. Build the containers (in their respective directories)

```docker build -f Dockerfile -t flask-app-image .```
```docker build -f Dockerfile -t nginx-proxy-image .```

2. Create a [Docker network](https://docs.docker.com/engine/network/)

```docker network create my-network ```

3. Run the containers (Flask in port 8000 and Nginx proxy in port 8080) using the network

```docker run -d --network=my-network -p 8000:8000 --name flask-app flask-app-image```
```docker run -d --network=my-network --env FLASK_SERVER_ADDR=http://flask-app:8000 -p 8080:8080 --name nginx-proxy nginx-proxy-image```
