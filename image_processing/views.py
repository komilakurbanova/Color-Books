from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageForm
from .models import UploadedImage
from .utils import convert_to_jpeg
import os

# Create your views here.
def say_hello(request):
    return render(request, "hello.html", {"name": "Komila"})


def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('processed_results')
    else:
        form = ImageForm()
    return render(request, 'upload_image.html', {'form': form})


# def processed_results(request):
#     uploaded_image = UploadedImage.objects.last()
#     if uploaded_image:
#         uploaded_image.process_image()
#         processed_image = uploaded_image.processedimage_set.last()
#         return render(request, 'cluster_results.html', {'uploaded_image': uploaded_image, 'processed_image': processed_image})
#     else:
#         return redirect('upload_image')


def processed_results(request):
    uploaded_image = UploadedImage.objects.last()
    if uploaded_image:
        file_name, file_extension = os.path.splitext(uploaded_image.image.name)[:2]
        if file_extension.lower() != '.jpeg':
            jpeg_data = convert_to_jpeg(uploaded_image.image)
            file_name = file_name.split('/')[-1] + '.jpeg'
            uploaded_image.image.save(file_name, jpeg_data, save=False)

        uploaded_image.process_image()
        processed_image = uploaded_image.processedimage_set.last()
        return render(request, 'cluster_results.html', {'uploaded_image': uploaded_image, 'processed_image': processed_image})
    else:
        return redirect('upload_image')