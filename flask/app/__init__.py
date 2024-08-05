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
        print(e)
        response.data = json.dumps({
            "code": e.code,
            "name": e.name,
            "description": e.description,
        })
        response.content_type = "application/json"
        response.status = e.code
        return response

    @app.post('/filter-mask')
    def filter_mask():
        file = request.files['image']
        form = request.form
        mask_width = int(form.get('mask_width'))
        mask_height = int(form.get('mask_height'))
        corner_handling_mode = form.get('corner_handling')
        filter_type = form.get('filter_type')

        if (file is None):
            abort(400, 'No file provided')
        if mask_width is None:
            abort(400, 'No mask width was provided')
        if mask_height is None:
            abort(400, 'No mask height was provided')
        if corner_handling_mode is None or isinstance(corner_handling_mode, str) is False:
            abort(400, 'Corner handling mode was not provided or was incorrect')
        if filter_type is None or filter_type not in image_processor.allowed_filter_types:
            abort(400, 'No filter type was provided or was incorrect')
        bts = file.stream.read()
        decoded_image = image_processor.decode_base64_image(bts)
        hsv_image = image_processor.rgb2hsv(decoded_image)

        filter_func = image_processor.median_filter
        match filter_type:
            case 'mean':
                filter_func = image_processor.mean_filter
            case 'median':
                filter_func = image_processor.median_filter

        processed_layer = image_processor.process_with_mask(
            hsv_image[:, :, 2],
            mask_width,
            mask_height,
            corner_handling_mode,
            filter_func)
        if len(processed_layer) < len(hsv_image[:, :, 2][1]) or len(processed_layer) < len(hsv_image[:, :, 2][0]):
            mask_width_half = np.floor(mask_width/2).astype(int)
            mask_height_half = np.floor(mask_height/2).astype(int)
            
            resized_hsv_image = hsv_image[
                mask_height_half:-mask_height_half, mask_width_half:-mask_width_half, :
            ]
            resized_hsv_image[:,:,2] = processed_layer
            hsv_image = resized_hsv_image
        else:
            hsv_image[:,:,2] = processed_layer
        rgb_image = image_processor.hsv2rgb(hsv_image)
        encoded_image = image_processor.encode_image_base64(rgb_image)
        print(rgb_image .shape)
        return {
            'image': encoded_image
        }
    @app.post('/histogram-equalization')
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
        return {
            'image': encoded_image
        }
        

    return app