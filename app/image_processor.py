import base64
import numpy as np
import imageio.v3 as iio
import io

def decode_base64_image(base64_image : str) -> np.ndarray:
    #base64_image = base64.b64decode(base64_image)
    base64_image = io.BytesIO(base64_image)
    decoded_image = iio.imread(base64_image)
    return decoded_image

def encode_image_base64(image : np.ndarray) -> bytes:
    image = iio.imwrite(uri='<bytes>', image=image, extension='.jpg')
    base64_bytes = base64.b64encode(image)
    base64_image = base64_bytes.decode('ascii')
    base64_image_str = 'data:image/jpeg;base64,' + base64_image
    return base64_image_str

def equalize_hsv_intensity_histogram(hsv_image : np.ndarray) -> np.ndarray:
    # Equalize the intensity histogram of the HSI image
    brightness_value = hsv_image[:, :, 2]
    # Calculate the histogram of the intensity channel
    histogram, _ = np.histogram(brightness_value, bins=256, range=(0, 255))
    
    def calc_cdf(histogram):
        cdf = np.zeros(histogram.size)
        cdf[0] = histogram[0]
        for i in range(1, histogram.size):
            cdf[i] = cdf[i - 1] + histogram[i]
        return cdf
    # Calculate the cumulative distribution function
    cdf = calc_cdf(histogram)
    # Normalize the cdf
    cdf = cdf / cdf[-1]
    # Map the intensity values to the new values
    new_intensity = np.floor(cdf[brightness_value.astype(int)] * 255).astype(int)
    return new_intensity