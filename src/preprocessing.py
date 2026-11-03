import numpy as np
from PIL import Image, ImageOps


def prepare_digit_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.autocontrast(image)

    pixels = np.asarray(image)
    # MNIST digits are bright on a dark background. Invert if the uploaded
    # image looks like dark ink on a light background.
    if pixels.mean() > 127:
        image = ImageOps.invert(image)

    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype="float32") / 255.0
    array = array.reshape(1, 28, 28, 1)
    return image, array
