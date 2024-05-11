from django.db import models
from .utils import crop_to_square, kmeans, edges
from django.core.files.base import ContentFile
from TEED.main import process_test_single_image
from PIL import Image as PILImage
import tempfile
import os
import numpy as np
import cv2


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    num_clusters = models.IntegerField(default=4)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def process_image(self):
        if not self.processedimage_set.exists():
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_png_file:
                image = cv2.imread(self.image.path)
                
                clustered_image = kmeans(image, self.num_clusters)
                clustered_image_pil = PILImage.fromarray(clustered_image)
                clustered_image_pil.save(temp_png_file.name, format='PNG')

                with open(temp_png_file.name, 'rb') as f:
                    file_name = os.path.splitext(self.image.name)[0].split('/')[-1] + f'_n{self.num_clusters}' + '.png'
                    content_file = ContentFile(f.read(), name=os.path.basename(file_name))

                processed_image = ProcessedImage(uploaded_image=self)
                processed_image.processed_image.save(content_file.name, content_file)
                processed_image.save()

        

class ProcessedImage(models.Model):
    uploaded_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)
    processed_image = models.ImageField(upload_to='processed/', null=True)

    def edges_image(self, threshold=70):
        # if not self.edgesimage_set.exists():
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_png_file:
            image = cv2.imread(self.processed_image.path)

            edges_image_array = edges(image, threshold)
            edges_image_pil = PILImage.fromarray(edges_image_array)
            edges_image_pil.save(temp_png_file.name, format='PNG')

            with open(temp_png_file.name, 'rb') as f:
                file_name = os.path.splitext(self.processed_image.name)[0].split('/')[-1] + f'_t{threshold}.png'
                content_file = ContentFile(f.read(), name=os.path.basename(file_name))

            edged_image = EdgesImage(image=self, threshold=threshold)
            edged_image.edged_image.save(content_file.name, content_file)
            edged_image.save()


class EdgesImage(models.Model):
    image = models.ForeignKey(ProcessedImage, on_delete=models.CASCADE)
    threshold = models.IntegerField(default=70)
    edged_image = models.ImageField(upload_to='edged/', null=True)


class UploadedImageAuto(models.Model):
    image = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def process_image(self):
        if not self.processedimageauto_set.exists():
            with tempfile.NamedTemporaryFile(suffix='.png') as temp_png_file:
                process_test_single_image(self.image.path,
                              "/Users/komilakurbanova/Desktop/codes/django-web/TEED/checkpoints/BIPED/19/19_model.pth",
                              temp_png_file.name)

                with open(temp_png_file.name, 'rb') as f:
                    file_name = os.path.splitext(self.image.name)[0].split('/')[-1] + '.png'
                    content_file = ContentFile(f.read(), name=os.path.basename(file_name))

                processed_image = ProcessedImageAuto(uploaded_image=self)
                processed_image.processed_image.save(content_file.name, content_file)
                processed_image.save()


class ProcessedImageAuto(models.Model):
    uploaded_image = models.ForeignKey(UploadedImageAuto, on_delete=models.CASCADE)
    processed_image = models.ImageField(upload_to='edged/', null=True)