from django.urls import path
from . import views


urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('results/', views.processed_results, name='processed_results'),
    path('results/edged/', views.edge_results, name='edge_results')
]
