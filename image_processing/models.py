from django.db import models
from .utils import crop_to_square, kmeans_edges
from django.core.files.base import ContentFile
from PIL import Image as PILImage
import tempfile
import os
import numpy as np


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    num_clusters = models.IntegerField(default=4)
    threshold = models.IntegerField(default=70)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def process_image(self):
        if not self.processedimage_set.exists():
            with tempfile.NamedTemporaryFile(suffix='.jpeg') as temp_jpeg_file:
                image = PILImage.open(self.image)

                numpy_image = np.array(image)
                
                clustered_image = kmeans_edges(numpy_image, self.num_clusters, self.threshold)
                clustered_image_pil = PILImage.fromarray(clustered_image)
                clustered_image_pil.save(temp_jpeg_file.name, format='JPEG')

                with open(temp_jpeg_file.name, 'rb') as f:
                    file_name = os.path.splitext(self.image.name)[0].split('/')[-1] + f'_n{self.num_clusters}' + '.jpeg'
                    content_file = ContentFile(f.read(), name=os.path.basename(file_name))

                processed_image = ProcessedImage(uploaded_image=self)
                processed_image.image.save(content_file.name, content_file)
                processed_image.save()


class ProcessedImage(models.Model):
    uploaded_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='processed/', null=True)
    