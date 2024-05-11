from django import forms
from .models import UploadedImage, UploadedImageAuto

class ImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image', 'num_clusters']
    

class ImageFormAuto(forms.ModelForm):
    class Meta:
        model = UploadedImageAuto
        fields = ['image']