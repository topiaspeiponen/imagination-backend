import os
from flask import Flask
from flask import request, abort, json
from werkzeug.exceptions import HTTPException
import app.image_processor as image_processor
import numpy as np


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.errorhandler(Exception)
    def handle_exception(e : Exception):
        # pass through HTTP errors
        if isinstance(e, HTTPException):
            return e

        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(e : HTTPException):
        response = e.get_response()
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        return response

    @app.post('/filter-mask')
    def filter_mask():
        file = request.files['image']
        if (file is None):
            abort(400, 'No file provided') 
        bts = file.stream.read()
        decoded_image = image_processor.decode_base64_image(bts)
        hsv_image = image_processor.rgb2hsv(decoded_image)
        hsv_image[:, :, 2] = image_processor.process_with_mask(hsv_image[:, :, 2],3,3,'substituteMin',image_processor.median_filter)
        rgb_image = image_processor.hsv2rgb(hsv_image)
        encoded_image = image_processor.encode_image_base64(rgb_image)
        print(rgb_image .shape)
        return {
            'image': encoded_image
        }
    @app.post('/equalize-histogram')
    def equalize_histogram():
        file = request.files['image']
        if (file is None):
            abort(400, 'No file provided') 
        bts = file.stream.read()
        decoded_image = image_processor.decode_base64_image(bts)
        hsv_image = image_processor.rgb2hsv(decoded_image)
        hsv_image[:, :, 2] = image_processor.equalize_hsv_intensity_histogram(hsv_image)
        rgb_image = image_processor.hsv2rgb(hsv_image)
        encoded_image = image_processor.encode_image_base64(rgb_image)
        print(rgb_image.shape)
        return {
            'image': encoded_image
        }
        

    return app