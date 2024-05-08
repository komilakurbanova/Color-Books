from PIL import Image
import tempfile

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
