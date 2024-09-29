
import scipy
import scipy.ndimage
import base64
import numpy as np
import imageio.v3 as iio
import io
from numpy.lib.stride_tricks import sliding_window_view
from typing import Callable, Literal
from flask import abort

type FilterType = Literal['median', 'mean']
type CornerHandlingType = Literal['mirror', 'resize', 'substituteMin', 'substituteMax']

allowed_corner_handling_types: CornerHandlingType = ['mirror', 'resize', 'substituteMin', 'substituteMax']
allowed_filter_types : FilterType = ['median', 'mean']

def decode_base64_image(base64_image : str) -> np.ndarray:
    base64_image = io.BytesIO(base64_image)
    decoded_image = iio.imread(base64_image)
    return decoded_image

def encode_image_base64(image : np.ndarray) -> bytes:
    image = image.astype(np.uint8)
    written_image = iio.imwrite(uri='<bytes>', image=image, extension='.jpg')
    base64_bytes = base64.b64encode(written_image)
    base64_image = base64_bytes.decode('ascii')
    base64_image_str = 'data:image/jpeg;base64,' + base64_image
    return base64_image_str

def equalize_hsv_intensity_histogram(hsv_image : np.ndarray) -> np.ndarray:
    # Equalize the intensity histogram of the HSI image
    brightness_value = hsv_image[:, :, 2]
    # Calculate the histogram of the intensity channel
    histogram, _ = np.histogram(brightness_value, bins=256)
    
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

def process_with_mask(
        layer: np.ndarray,
        mask_width: int,
        mask_height: int,
        corner_handling: CornerHandlingType,
        filter_type: FilterType
        ) -> np.ndarray:
    """Process image layer with a mask filter of size M x N

    Parameters
    -------
    layer : the provided image layer (should be 2-dimensional larger than 10x10 layer)
    mask_width : width of the mask (should be 3 or bigger odd integer)
    mask_height : height of the mask (should be 3 or bigger odd integer)
    corner_handling : specifies the way corners should be handled with the filter
        - 'mirror': the layer is padded with mirrored values from the inside of the layer (small impact to the edge pixels)
        - 'resize': the filter only processes pixels where it fits completely (meaning layer will be smaller)
        - 'substituteMin': layer is padded with zeros so that the mask filter can process all original pixels
        - 'substituteMax': layer is padded with the max value of the layer so that the mask filter can process all original pixels
    filter_type: how the mask/filter should calculate the return value for a pixel
        
    Returns
    -------
    layer : (..., ...) ndarray
        The image layer as ndarray
    """
    if mask_width < 3 or mask_height < 3:
        abort(400,'Mask dimensions should be integers 3 or higher')
    if mask_width % 2 != 1 or mask_height % 2 != 1:
        abort(400, 'Mask dimensions should be odd integers')
    if layer.shape < (10,10):
        abort(400,'Layer/image size should be 10x10 or greater')
    if corner_handling not in allowed_corner_handling_types:
        abort(400, 'Corner handling type is not one of the allowed values')
    filtered_layer = layer.copy()

    mask_width_half = mask_width // 2
    mask_height_half = mask_height // 2
    
    match filter_type:
        case 'mean':
            if corner_handling == 'substituteMax':
                filtered_layer = scipy.ndimage.uniform_filter(layer, (mask_height, mask_width), mode='constant', cval=np.max(layer))
            elif (corner_handling == 'substituteMin'):
                filtered_layer = scipy.ndimage.uniform_filter(layer, (mask_height, mask_width), mode='constant', cval=0)
            elif (corner_handling == 'mirror'):
                filtered_layer = scipy.ndimage.uniform_filter(layer, (mask_height, mask_width), mode='mirror')
            elif (corner_handling == 'resize'):
                filtered = scipy.ndimage.uniform_filter(layer, (mask_height, mask_width), mode='mirror')
                # Only take the area where the mask could be completely fitted
                filtered_layer = filtered[mask_height_half:-mask_height_half, mask_width_half:-mask_width_half]
        case 'median':
            if corner_handling == 'substituteMax':
                filtered_layer = scipy.ndimage.median_filter(layer, (mask_height, mask_width), mode='constant', cval=np.max(layer))
            elif (corner_handling == 'substituteMin'):
                filtered_layer = scipy.ndimage.median_filter(layer, (mask_height, mask_width), mode='constant', cval=0)
            elif (corner_handling == 'mirror'):
                filtered_layer = scipy.ndimage.median_filter(layer, (mask_height, mask_width), mode='mirror')
            elif (corner_handling == 'resize'):
                filtered = scipy.ndimage.median_filter(layer, (mask_height, mask_width), mode='mirror')
                # Only take the area where the mask could be completely fitted
                filtered_layer = filtered[mask_height_half:-mask_height_half, mask_width_half:-mask_width_half]
    return filtered_layer
    

def _prepare_colorarray(arr : np.ndarray, channel_axis=-1):
    """NOTE: This is a slightly modified version of _prepare_colorarray from scikit-image (https://github.com/scikit-image/scikit-image/blob/v0.23.1/skimage/color/colorconv.py)
    Check the shape of the array and convert it to
    floating point representation.
    """

    if arr.shape[channel_axis] != 3:
        msg = (
            f'the input array must have size 3 along `channel_axis`, '
            f'got {arr.shape}'
        )
        raise ValueError(msg)

    float_arr = arr.astype(float)
    return float_arr;

def rgb2hsv(rgb):
    """ NOTE: This is a slightly modified version of scikit-image rgb2hsv(https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2hsv)
    RGB to HSV color space conversion.

    Parameters
    ----------
    rgb : (..., C=3, ...) array_like
        The image in RGB format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in HSV format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `rgb` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV

    """
    input_is_one_pixel = rgb.ndim == 1
    if input_is_one_pixel:
        rgb = rgb[np.newaxis, ...]

    arr = _prepare_colorarray(rgb, channel_axis=-1)
    out = np.empty_like(arr)

    # -- V channel
    out_v = arr.max(-1)

    # -- S channel
    delta = np.ptp(arr, axis=-1)
    # Ignore warning for zero divided by zero
    old_settings = np.seterr(invalid='ignore')
    out_s = delta / out_v
    out_s[delta == 0.0] = 0.0

    # -- H channel
    # red is max
    idx = arr[..., 0] == out_v
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = arr[..., 1] == out_v
    out[idx, 0] = 2.0 + (arr[idx, 2] - arr[idx, 0]) / delta[idx]

    # blue is max
    idx = arr[..., 2] == out_v
    out[idx, 0] = 4.0 + (arr[idx, 0] - arr[idx, 1]) / delta[idx]
    out_h = (out[..., 0] / 6.0) % 1.0
    out_h[delta == 0.0] = 0.0

    np.seterr(**old_settings)

    # -- output
    out[..., 0] = out_h
    out[..., 1] = out_s
    out[..., 2] = out_v

    # # remove NaN
    out[np.isnan(out)] = 0

    if input_is_one_pixel:
        out = np.squeeze(out, axis=0)

    return out

def hsv2rgb(hsv):
    """NOTE: This is a slightly modified version of scikit-image hsv2rgb(https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.hsv2rgb)
    HSV to RGB color space conversion.

    Parameters
    ----------
    hsv : (..., C=3, ...) array_like
        The image in HSV format. By default, the final dimension denotes
        channels.
    channel_axis : int, optional
        This parameter indicates which axis of the array corresponds to
        channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.

    Returns
    -------
    out : (..., C=3, ...) ndarray
        The image in RGB format. Same dimensions as input.

    Raises
    ------
    ValueError
        If `hsv` is not at least 2-D with shape (..., C=3, ...).

    Notes
    -----
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/HSL_and_HSV

    """
    arr = _prepare_colorarray(hsv, channel_axis=-1)

    hi = np.floor(arr[..., 0] * 6)
    f = arr[..., 0] * 6 - hi
    p = arr[..., 2] * (1 - arr[..., 1])
    q = arr[..., 2] * (1 - f * arr[..., 1])
    t = arr[..., 2] * (1 - (1 - f) * arr[..., 1])
    v = arr[..., 2]

    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    out = np.choose(
        hi,
        np.stack(
            [
                np.stack((v, t, p), axis=-1),
                np.stack((q, v, p), axis=-1),
                np.stack((p, v, t), axis=-1),
                np.stack((p, q, v), axis=-1),
                np.stack((t, p, v), axis=-1),
                np.stack((v, p, q), axis=-1),
            ]
        ),
    )

    return out