from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageForm, ImageFormAuto
from .models import UploadedImage, ProcessedImage, UploadedImageAuto
from .utils import convert_to_jpeg
import os


def main_page(request):
    return render(request, 'main_page.html')


def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()
            return redirect('processed_results', image_id=image.id)
    else:
        form = ImageForm()
    return render(request, 'upload_image.html', {'form': form})


def upload_image_auto(request):
    if request.method == 'POST':
        form = ImageFormAuto(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()
            return redirect('processed_results_auto', image_id=image.id)
    else:
        form = ImageForm()
    return render(request, 'upload_image_auto.html', {'form': form})


def processed_results_auto(request, image_id):
    uploaded_image = UploadedImageAuto.objects.get(id=image_id)
    print(uploaded_image)
    if uploaded_image:
        file_name, file_extension = os.path.splitext(uploaded_image.image.name)[:2]
        if file_extension.lower() != '.png':
            png_data = convert_to_jpeg(uploaded_image.image)
            file_name = file_name.split('/')[-1] + '.png'
            uploaded_image.image.save(file_name, png_data, save=False)

        uploaded_image.process_image()
        processed_image = uploaded_image.processedimageauto_set.last()
        return render(request, 'edges_auto_results.html', {'uploaded_image': uploaded_image, 'processed_image': processed_image})
    else:
        return redirect('upload_image')
    

def processed_results(request, image_id):
    uploaded_image = UploadedImage.objects.get(id=image_id)
    if uploaded_image:
        file_name, file_extension = os.path.splitext(uploaded_image.image.name)[:2]
        if file_extension.lower() != '.png':
            png_data = convert_to_jpeg(uploaded_image.image)
            file_name = file_name.split('/')[-1] + '.png'
            uploaded_image.image.save(file_name, png_data, save=False)

        uploaded_image.process_image()
        processed_image = uploaded_image.processedimage_set.last()
        return render(request, 'cluster_results.html', {'uploaded_image': uploaded_image, 'processed_image': processed_image})
    else:
        return redirect('upload_image')
    

def edge_results(request, image_id):
    clustered_image = ProcessedImage.objects.get(id=image_id)
    if clustered_image:
        if request.method == 'POST':
            threshold = int(request.POST.get('threshold'))
        else:
            threshold = 70

        clustered_image.edges_image(threshold)
        edged_image = clustered_image.edgesimage_set.last()
        return render(request, 'edges_result.html', {'edged_image': edged_image})
    else:
        return redirect('upload_image')
    
