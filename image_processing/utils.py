from PIL import Image
import tempfile
import cv2
import numpy as np


def crop_to_square(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def convert_to_jpeg(image):
    img = Image.open(image)
    temp_jpeg_file = tempfile.NamedTemporaryFile(suffix='.jpeg')
    img.convert('RGB').save(temp_jpeg_file.name, format='JPEG', quality=95)
    return temp_jpeg_file


def kmeans(image, k):
    # k = 10
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)
    # gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    # gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    # gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # threshold = 50
    # _, edges = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # edges = cv2.bitwise_not(edges)
    # white_background = np.ones_like(image) * 255
    # masked_image = cv2.bitwise_and(white_background, edges)
    # return masked_image
    return segmented_image


def edges(segmented_image, threshold):
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, edges = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges = cv2.bitwise_not(edges)
    white_background = np.ones_like(segmented_image) * 255
    masked_image = cv2.bitwise_and(white_background, edges)
    return masked_image