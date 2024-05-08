from django.db import models
from .utils import crop_to_square
from django.core.files.base import ContentFile
from PIL import Image as PILImage
import tempfile
import os

# Create your models here.
# class UploadedImage(models.Model):
#     image = models.ImageField(upload_to='images/')
#     uploaded_at = models.DateTimeField(auto_now_add=True)


# class ProcessedImage(models.Model):
#     uploaded_image = models.OneToOneField(UploadedImage, on_delete=models.CASCADE)
#     processed_image = models.ImageField(upload_to='processed/')
#     processed_at = models.DateTimeField(auto_now_add=True)

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def process_image(self):
        if not self.processedimage_set.exists():
            with tempfile.NamedTemporaryFile(suffix='.jpeg') as temp_jpeg_file:
                image = PILImage.open(self.image)
                cropped_image = crop_to_square(image)
                cropped_image.save(temp_jpeg_file.name, format='JPEG')

                with open(temp_jpeg_file.name, 'rb') as f:
                    file_name = os.path.splitext(self.image.name)[0].split('/')[-1] + '.jpeg'
                    content_file = ContentFile(f.read(), name=os.path.basename(file_name))

                processed_image = ProcessedImage(uploaded_image=self)
                processed_image.image.save(content_file.name, content_file)
                processed_image.save()


class ProcessedImage(models.Model):
    uploaded_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='processed/', null=True)