import os
from flask import Flask
from flask import request, abort
import app.image_processor as image_processor
import cv2 as cv


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

    @app.post('/equalize-histogram')
    def equalize_histogram():
        file = request.files['image']
        if (file is None):
            abort(400, 'No file provided') 
        print(file)
        bts = file.stream.read()
        decoded_image = image_processor.decode_base64_image(bts)
        hsv_image = cv.cvtColor(decoded_image, cv.COLOR_RGB2HSV)
        hsv_image[:, :, 2] = image_processor.equalize_hsv_intensity_histogram(hsv_image)
        rgb_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2RGB)
        return {
            'image': image_processor.encode_image_base64(rgb_image)
        }
        

    return app