from django import forms
from .models import UploadedImage

class ImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image', 'num_clusters', 'threshold']