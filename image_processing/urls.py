from django.urls import path
from . import views


urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('results/<int:image_id>/', views.processed_results, name='processed_results'),
    path('results/edged/<int:image_id>/', views.edge_results, name='edge_results'),
    path('auto/', views.upload_image_auto, name='upload_image_auto'),
    path('auto/edged/<int:image_id>/', views.processed_results_auto, name='processed_results_auto'),
]
