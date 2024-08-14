import os
from flask import Flask
from flask import request, abort, json
from werkzeug.exceptions import HTTPException
from app.image_processor import allowed_filter_types, decode_base64_image, rgb2hsv, median_filter, mean_filter, process_with_mask, hsv2rgb, encode_image_base64, equalize_hsv_intensity_histogram
import numpy as np
import json

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    app.config.from_file('config.json', load=json.load)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.before_request
    def authenticate():
        print('before req ', request.headers)
        secret_key = app.config["SECRET_KEY"]
        xapikey = request.headers.get("X-Api-Key")
        if (secret_key != xapikey):
            abort(403)

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
    
    @app.get('/flask-health-check')
    def health_check():
        return 'success!'

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
        if filter_type is None or filter_type not in allowed_filter_types:
            abort(400, 'No filter type was provided or was incorrect')
        bts = file.stream.read()
        decoded_image = decode_base64_image(bts)
        hsv_image = rgb2hsv(decoded_image)

        filter_func = median_filter
        match filter_type:
            case 'mean':
                filter_func = mean_filter
            case 'median':
                filter_func = median_filter

        processed_layer = process_with_mask(
            hsv_image[:, :, 2],
            mask_width,
            mask_height,
            corner_handling_mode,
            filter_func)
        if processed_layer.shape[1] < hsv_image[:, :, 2].shape[1] or processed_layer.shape[0] < hsv_image[:, :, 2].shape[0]:
            mask_width_half = np.floor(mask_width/2).astype(int)
            mask_height_half = np.floor(mask_height/2).astype(int)
            
            resized_hsv_image = hsv_image[
                mask_height_half:-mask_height_half, mask_width_half:-mask_width_half, :
            ]
            resized_hsv_image[:,:,2] = processed_layer
            hsv_image = resized_hsv_image
        else:
            hsv_image[:,:,2] = processed_layer
        rgb_image = hsv2rgb(hsv_image)
        encoded_image = encode_image_base64(rgb_image)
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
        decoded_image = decode_base64_image(bts)
        hsv_image = rgb2hsv(decoded_image)
        hsv_image[:, :, 2] = equalize_hsv_intensity_histogram(hsv_image)
        rgb_image = hsv2rgb(hsv_image)
        encoded_image = encode_image_base64(rgb_image)
        return {
            'image': encoded_image
        }
        

    return app