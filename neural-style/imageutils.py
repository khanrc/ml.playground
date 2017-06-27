import cv2
import PIL
import numpy as np


def load(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = float(max_size) / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    # Convert to numpy floating-point array.
    return np.float32(image)

def save(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

def crop(image): # center crop
    h, w = image.shape[:2]
    if h > w:
        sth = (h-w)/2
        edh = h-sth
        image = image[sth:edh, :]
    else:
        stw = (w-h)/2
        edw = w-stw
        image = image[:, stw:edw]

    return image

def resize(image, size=(256, 256)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def preproc(image, size=(256, 256)):
    if image.ndim == 4:
        print('already batch image')
        return image

    return Image.resize(Image.crop(image))

def to_batch(image):
    if image.ndim != 3:
        print("dim error")
        return None
    
    return np.expand_dims(image, axis=0)