from django import forms
from .models import UploadedImage, EdgesImage

class ImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image', 'num_clusters']

# class ThresholdForm(forms.ModelForm):
#     class Meta:
#         model = EdgesImage
#         fields = ['threshold']